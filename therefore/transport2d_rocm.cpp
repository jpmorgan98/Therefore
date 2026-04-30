// transport2d_rocm.cpp  —  GPU-resident one-cell-inversion solver
//
// Design overview
// ---------------
// The CPU solver (transport2d.cpp) uses a "thread-local scratch" model:
//   per iteration → assemble n×n matrix → LU-factor → solve → discard matrix
// Memory: O(nthreads × n²) instead of O(ncells × n²).
//
// This file mirrors that model on the GPU via *chunked batched processing*:
//   per iteration → for each chunk of C cells:
//     ① assemble_cell_matrices_chunk_kernel  (new; GPU-side assembly)
//     ② rocsolver_dgetrf_strided_batched     (factor C matrices)
//     ③ rocsolver_dgetrs_strided_batched     (solve C systems, in-place on d_rhs)
//   LU factors are discarded after each chunk; peak allocation is C × n².
//
// The full source-iteration convergence loop stays on the GPU exactly as
// before (d_rhs_const copy → upwind inflow kernel → chunk solve → rocblas
// nrm2 check → swap).  The only CPU↔GPU traffic is the one-time cell-data
// upload and the per-timestep flux download for I/O.
//
// Chunk size is chosen automatically at first call to keep the d_lu_chunk
// buffer at ≤ kTargetChunkMemoryMB MB, capped at ncells.  For the 500×100
// S8 problem (n=128, ncells=50,000): 512 MB → C ≈ 4,096 cells per chunk
// versus the 6.6 GB the old approach required.
//
// The strided-batched API is retained exactly.  Assembly uses one HIP thread
// block per cell-in-chunk, with threads cooperating to zero-initialise,
// fill streaming/absorption blocks, and fill the scatter coupling.

#include "transport2d.hpp"
#include "output.hpp"

#include <filesystem>
#include <iostream>

#ifdef THEREFORE2D_ENABLE_ROCM

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace therefore2d {
namespace {

// ---------------------------------------------------------------------------
// Handle casting helpers
// ---------------------------------------------------------------------------

rocblas_handle as_handle(void* h) {
    return reinterpret_cast<rocblas_handle>(h);
}

void hip_check(hipError_t s, const char* msg) {
    if (s != hipSuccess) throw std::runtime_error(msg);
}

void rocblas_check(rocblas_status s, const char* msg) {
    if (s != rocblas_status_success) throw std::runtime_error(msg);
}

// ---------------------------------------------------------------------------
// Index helpers  (device-callable; mirror the host inline helpers)
// ---------------------------------------------------------------------------

__device__ __forceinline__
int packed_base(int cell, int group, int dir, int groups, int num_dirs) {
    return cell * (groups * num_dirs * kDofsPerAngleGroup2D)
         + ((group * num_dirs + dir) * kDofsPerAngleGroup2D);
}

__device__ __forceinline__
int west_east_face_base(int bj, int group, int dir, int groups, int num_dirs) {
    return (((bj * groups + group) * num_dirs + dir) * 4);
}

__device__ __forceinline__
int south_north_face_base(int bi, int group, int dir, int groups, int num_dirs) {
    return (((bi * groups + group) * num_dirs + dir) * 4);
}

// ---------------------------------------------------------------------------
// Streaming matrix scalar entry for one 4×4 block
// ---------------------------------------------------------------------------
// These reproduce x_stream_block / y_stream_block from transport2d.cpp.
// Both functions return the (r, c) entry (row-major block, c ∈ [0,4)) of the
// 4×4 streaming matrix used for the low and high temporal moment sub-blocks.

__device__ double x_stream_entry(double ax, bool pos_x, int r, int c) {
    if (r == c) return ax;                                // diagonal always ax
    if (pos_x) return (c % 2 == 0 && r == c + 1) ? -ax : 0.0;
    else       return (c % 2 == 1 && r == c - 1) ? -ax : 0.0;
}

__device__ double y_stream_entry(double ay, bool pos_y, int r, int c) {
    if (r == c) return ay;
    if (pos_y) return (c < 2 && r == c + 2) ? -ay : 0.0;
    else       return (c >= 2 && r == c - 2) ? -ay : 0.0;
}

// ---------------------------------------------------------------------------
// Single 8×8 diagonal block entry
// ---------------------------------------------------------------------------
// Returns the (r8, c8) entry (r8, c8 ∈ [0,8)) of the 8×8 block that lives at
// (row0, row0) in the full n×n cell matrix.  Matches assemble_angle_group_block
// in transport2d.cpp exactly.

__device__ double block8x8_entry(int r8, int c8,
                                  double ax, double ay,
                                  double gamma, double tau,
                                  bool pos_x, bool pos_y) {
    double v = 0.0;
    const int r4 = r8 & 3;                               // r8 % 4
    const int c4 = c8 & 3;
    const bool r_lo = (r8 < 4);
    const bool c_lo = (c8 < 4);

    if (r_lo && c_lo) {
        // Low–Low: Sx + Sy + gamma * I
        v += x_stream_entry(ax, pos_x, r4, c4)
           + y_stream_entry(ay, pos_y, r4, c4);
        if (r4 == c4) v += gamma;
    } else if (r_lo && !c_lo) {
        // Low–High: tau_half * I
        if (r4 == (c8 - 4)) v += 0.5 * tau;
    } else if (!r_lo && c_lo) {
        // High–Low: -tau * I
        if ((r8 - 4) == c4) v -= tau;
    } else {
        // High–High: Sx + Sy + (gamma + tau) * I
        v += x_stream_entry(ax, pos_x, r4, c4)
           + y_stream_entry(ay, pos_y, r4, c4);
        if (r4 == c4) v += gamma + tau;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Cell-matrix assembly kernel
// ---------------------------------------------------------------------------
// Grid : (chunk_count,)          — one block per cell in the chunk
// Block: kAssemblyBlockDim       — flat threads cooperating over the n×n matrix
//
// Each block writes one n×n cell matrix into A_chunk (column-major) and works
// in three synchronised phases:
//
//   Phase 1 — Zero-initialise:  all threads zero n²/blockDim entries each.
//   Phase 2 — Diagonal blocks:  threads cover all (gd, r8, c8) triples
//              (gd = group*num_dirs+dir); each triple maps to exactly one
//              matrix entry (no race condition).
//   Phase 3 — Scatter coupling: threads cover all (gd_to, gd_from) pairs;
//              each pair writes 8 diagonal DOF entries in the (gd_to, gd_from)
//              column-block (each entry written by exactly one thread; no
//              atomics required).

static constexpr int kAssemblyBlockDim = 256;

__global__ void assemble_cell_matrices_chunk_kernel(
    double* __restrict__ A_chunk,     // [chunk_count, n, n] column-major
    int chunk_start,
    int chunk_count,
    int n,                            // cell_block_size = groups*num_dirs*8
    int groups,
    int num_dirs,
    const double* __restrict__ cell_dx,
    const double* __restrict__ cell_dy,
    const double* __restrict__ cell_dt,
    const double* __restrict__ cell_velocity,   // [ncells * groups]
    const double* __restrict__ cell_sigma_t,    // [ncells * groups]
    const double* __restrict__ cell_sigma_s,    // [ncells * groups * groups]
    const double* __restrict__ dir_mu,
    const double* __restrict__ dir_eta,
    const double* __restrict__ dir_weight)
{
    const int lc  = static_cast<int>(blockIdx.x);
    if (lc >= chunk_count) return;
    const int cell = chunk_start + lc;

    // Pointer to this cell's n×n column-major matrix in the chunk buffer.
    double* A = A_chunk + static_cast<long long>(lc) * n * n;

    const int tid  = static_cast<int>(threadIdx.x);
    const int bdim = kAssemblyBlockDim;

    // ------------------------------------------------------------------
    // Phase 1: zero-initialise
    // ------------------------------------------------------------------
    for (int k = tid; k < n * n; k += bdim) A[k] = 0.0;
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: streaming + absorption + temporal coupling (diagonal blocks)
    //
    // Each (group × dir) pair contributes one 8×8 sub-block at position
    // (row0, row0) with row0 = gd * kDofsPerAngleGroup2D.
    // We flatten over (gd, r8, c8): total = gd_total * 64 entries.
    // No two threads write the same (A row, A col) pair here.
    // ------------------------------------------------------------------
    const int gd_total   = groups * num_dirs;
    const int diag_total = gd_total * 64;           // 64 = 8*8

    const double dx_c  = cell_dx[cell];
    const double dy_c  = cell_dy[cell];
    const double dt_c  = cell_dt[cell];

    for (int idx = tid; idx < diag_total; idx += bdim) {
        const int gd  = idx / 64;
        const int rc  = idx % 64;
        const int r8  = rc / 8;
        const int c8  = rc % 8;

        const int g   = gd / num_dirs;
        const int d   = gd % num_dirs;
        const int row0 = gd * kDofsPerAngleGroup2D;

        const double v_g = cell_velocity[cell * groups + g];
        const double s_t = cell_sigma_t [cell * groups + g];
        const double mu  = dir_mu [d];
        const double eta = dir_eta[d];
        const double vol = dx_c * dy_c;

        const double gamma = 0.25 * vol * s_t;
        const double tau   = vol / (v_g * dt_c);
        const double ax    = 0.5 * fabs(mu)  * dy_c;
        const double ay    = 0.5 * fabs(eta) * dx_c;

        const double entry = block8x8_entry(r8, c8, ax, ay, gamma, tau,
                                             mu >= 0.0, eta >= 0.0);
        // Column-major write: A[col * n + row]
        A[static_cast<long long>(row0 + c8) * n + (row0 + r8)] += entry;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: scatter coupling
    //
    // For each (gd_to, gd_from) pair and each DOF dof ∈ [0,8):
    //   A[(col0+dof)*n + (row0+dof)] += beta
    // where beta = -(vol/4) * sigma_s[g_to,g_from] * weight[d_from].
    //
    // Entry (col0+dof, row0+dof) is unique to this (gd_to,gd_from,dof)
    // triple, so no atomics are needed.
    // ------------------------------------------------------------------
    const double vol_c = dx_c * dy_c;
    const int scatter_pairs = gd_total * gd_total;

    for (int idx = tid; idx < scatter_pairs; idx += bdim) {
        const int gd_to   = idx / gd_total;
        const int gd_from = idx % gd_total;
        const int g_to    = gd_to   / num_dirs;
        const int g_from  = gd_from / num_dirs;
        const int d_from  = gd_from % num_dirs;

        const double sigma_s =
            cell_sigma_s[(static_cast<long long>(cell) * groups + g_to) * groups + g_from];
        if (sigma_s == 0.0) continue;

        const double beta = -(vol_c * 0.25) * sigma_s * dir_weight[d_from];
        const int row0 = gd_to   * kDofsPerAngleGroup2D;
        const int col0 = gd_from * kDofsPerAngleGroup2D;

        for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
            A[static_cast<long long>(col0 + dof) * n + (row0 + dof)] += beta;
        }
    }
    // No final sync needed: each block writes its own A slice.
}

// ---------------------------------------------------------------------------
// Upwind inflow RHS kernel  (unchanged from original)
// ---------------------------------------------------------------------------
__global__ void add_upwind_inflow_rhs_kernel(
    double* rhs,
    const double* iterate_flux,
    int nx, int ny, int groups, int num_dirs,
    const double* cell_dx, const double* cell_dy,
    const double* dir_mu,  const double* dir_eta,
    const double* boundary_west,  const double* boundary_east,
    const double* boundary_south, const double* boundary_north,
    int has_west, int has_east, int has_south, int has_north)
{
    const int tid   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nx * ny * groups * num_dirs;
    if (tid >= total) return;

    const int dir   = tid % num_dirs;
    const int tmp0  = tid / num_dirs;
    const int group = tmp0 % groups;
    const int cell  = tmp0 / groups;
    const int i     = cell % nx;
    const int j     = cell / nx;

    const double mu  = dir_mu [dir];
    const double eta = dir_eta[dir];
    const double ax  = 0.5 * fabs(mu)  * cell_dy[cell];
    const double ay  = 0.5 * fabs(eta) * cell_dx[cell];
    const int    dst = packed_base(cell, group, dir, groups, num_dirs);

    // X inflow
    if (mu >= 0.0) {
        if (i > 0) {
            const int src = packed_base(cell - 1, group, dir, groups, num_dirs);
            rhs[dst+0] += ax * iterate_flux[src+1];
            rhs[dst+2] += ax * iterate_flux[src+3];
            rhs[dst+4] += ax * iterate_flux[src+5];
            rhs[dst+6] += ax * iterate_flux[src+7];
        } else if (has_west) {
            const int off = west_east_face_base(j, group, dir, groups, num_dirs);
            rhs[dst+0] += ax * boundary_west[off+0];
            rhs[dst+2] += ax * boundary_west[off+1];
            rhs[dst+4] += ax * boundary_west[off+2];
            rhs[dst+6] += ax * boundary_west[off+3];
        }
    } else {
        if (i + 1 < nx) {
            const int src = packed_base(cell + 1, group, dir, groups, num_dirs);
            rhs[dst+1] += ax * iterate_flux[src+0];
            rhs[dst+3] += ax * iterate_flux[src+2];
            rhs[dst+5] += ax * iterate_flux[src+4];
            rhs[dst+7] += ax * iterate_flux[src+6];
        } else if (has_east) {
            const int off = west_east_face_base(j, group, dir, groups, num_dirs);
            rhs[dst+1] += ax * boundary_east[off+0];
            rhs[dst+3] += ax * boundary_east[off+1];
            rhs[dst+5] += ax * boundary_east[off+2];
            rhs[dst+7] += ax * boundary_east[off+3];
        }
    }

    // Y inflow
    if (eta >= 0.0) {
        if (j > 0) {
            const int src = packed_base(cell - nx, group, dir, groups, num_dirs);
            rhs[dst+0] += ay * iterate_flux[src+2];
            rhs[dst+1] += ay * iterate_flux[src+3];
            rhs[dst+4] += ay * iterate_flux[src+6];
            rhs[dst+5] += ay * iterate_flux[src+7];
        } else if (has_south) {
            const int off = south_north_face_base(i, group, dir, groups, num_dirs);
            rhs[dst+0] += ay * boundary_south[off+0];
            rhs[dst+1] += ay * boundary_south[off+1];
            rhs[dst+4] += ay * boundary_south[off+2];
            rhs[dst+5] += ay * boundary_south[off+3];
        }
    } else {
        if (j + 1 < ny) {
            const int src = packed_base(cell + nx, group, dir, groups, num_dirs);
            rhs[dst+2] += ay * iterate_flux[src+0];
            rhs[dst+3] += ay * iterate_flux[src+1];
            rhs[dst+6] += ay * iterate_flux[src+4];
            rhs[dst+7] += ay * iterate_flux[src+5];
        } else if (has_north) {
            const int off = south_north_face_base(i, group, dir, groups, num_dirs);
            rhs[dst+2] += ay * boundary_north[off+0];
            rhs[dst+3] += ay * boundary_north[off+1];
            rhs[dst+6] += ay * boundary_north[off+2];
            rhs[dst+7] += ay * boundary_north[off+3];
        }
    }
}

// ---------------------------------------------------------------------------
// Constant RHS kernel  (unchanged from original)
// ---------------------------------------------------------------------------
__global__ void build_constant_rhs_kernel(
    double* rhs_const,
    const double* flux_previous,
    int num_cells, int groups, int num_dirs,
    const double* cell_dx, const double* cell_dy, const double* cell_dt,
    const double* cell_velocity, const double* cell_source)
{
    const int tid   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = num_cells * groups * num_dirs;
    if (tid >= total) return;

    const int dir   = tid % num_dirs;
    const int tmp0  = tid / num_dirs;
    const int group = tmp0 % groups;
    const int cell  = tmp0 / groups;

    const double volume   = cell_dx[cell] * cell_dy[cell];
    const double tau_half = 0.5 * volume /
                            (cell_velocity[cell * groups + group] * cell_dt[cell]);
    const double src_scale = volume / 4.0;
    const int base = packed_base(cell, group, dir, groups, num_dirs);

    for (int corner = 0; corner < 4; ++corner) {
        rhs_const[base + corner] =
            src_scale * cell_source[base + corner]
            + tau_half * flux_previous[base + corner];
        rhs_const[base + 4 + corner] =
            src_scale * cell_source[base + 4 + corner];
    }
}

// ---------------------------------------------------------------------------
// Chunk size heuristic
// ---------------------------------------------------------------------------
// Target: keep d_lu_chunk below kTargetChunkBytes.
// Falls back to a minimum of 32 cells if the GPU reports no free memory.

static constexpr std::size_t kTargetChunkBytes = 512ULL * 1024 * 1024; // 512 MB

int compute_chunk_size(int ncells, int n) {
    // bytes needed per cell in the chunk buffer:
    //   n*n doubles (LU matrix) + n ints (pivots) + 1 int (info)
    const std::size_t bytes_per_cell =
        static_cast<std::size_t>(n) * n * sizeof(double)
        + static_cast<std::size_t>(n) * sizeof(int) + sizeof(int);

    std::size_t free_bytes = 0, total_bytes = 0;
    std::size_t budget = kTargetChunkBytes;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
        // Use at most 55% of currently free device memory for chunk buffers.
        const std::size_t headroom = (free_bytes * 55) / 100;
        budget = headroom; //std::min(headroom); // peak LU buffer
    }

    const int chunk = static_cast<int>(budget / bytes_per_cell);
    return std::max(32, std::min(chunk, ncells));
}

// ---------------------------------------------------------------------------
// Cell data upload  (called once; replaces ensure_sweep_data_rocm)
// ---------------------------------------------------------------------------
// Uploads all cell geometry and cross-section data needed for GPU-side
// matrix assembly.  Also uploads boundary conditions and direction arrays.

void ensure_cell_data_rocm(const SolverState2D& state, RocmLUCache& cache) {
    if (cache.sweep_data_valid) return;

    const Problem2D& p = state.problem;
    const int nc = p.num_cells();
    const int G  = p.groups;
    const int D  = p.num_dirs();

    // Build flat host arrays ------------------------------------------
    std::vector<double> h_dx(nc), h_dy(nc), h_dt(nc);
    std::vector<double> h_velocity(nc * G);
    std::vector<double> h_sigma_t (nc * G);
    std::vector<double> h_sigma_s (nc * G * G);
    std::vector<double> h_source  (p.total_unknowns());
    std::vector<double> h_mu(D), h_eta(D), h_weight(D);

    for (int c = 0; c < nc; ++c) {
        const Cell2D& cell = state.cells[c];
        h_dx[c] = cell.dx;
        h_dy[c] = cell.dy;
        h_dt[c] = cell.dt;
        for (int g = 0; g < G; ++g) {
            h_velocity[c * G + g] = cell.velocity[g];
            h_sigma_t [c * G + g] = cell.sigma_t [g];
        }
        for (int g_to = 0; g_to < G; ++g_to)
            for (int g_from = 0; g_from < G; ++g_from)
                h_sigma_s[(c * G + g_to) * G + g_from] =
                    cell.sigma_s[g_to * G + g_from];
        const double* src = cell.source.data();
        std::copy(src, src + p.cell_block_size(),
                  h_source.begin() + c * p.cell_block_size());
    }
    for (int d = 0; d < D; ++d) {
        h_mu    [d] = p.directions[d].mu;
        h_eta   [d] = p.directions[d].eta;
        h_weight[d] = p.directions[d].weight;
    }

    // Allocate device arrays -----------------------------------------
    auto alloc = [](void** ptr, std::size_t bytes, const char* msg) {
        hip_check(hipMalloc(ptr, bytes), msg);
    };

#define ALLOC_D(field, vec) \
    alloc(reinterpret_cast<void**>(&cache.field), \
          sizeof(*cache.field) * (vec).size(), "hipMalloc " #field)
#define COPY_D(field, vec) \
    hip_check(hipMemcpy(cache.field, (vec).data(), \
                        sizeof(*cache.field) * (vec).size(), \
                        hipMemcpyHostToDevice), "hipMemcpy " #field)

    ALLOC_D(d_cell_dx,       h_dx);
    ALLOC_D(d_cell_dy,       h_dy);
    ALLOC_D(d_cell_dt,       h_dt);
    ALLOC_D(d_cell_velocity, h_velocity);
    ALLOC_D(d_cell_sigma_t,  h_sigma_t);
    ALLOC_D(d_cell_sigma_s,  h_sigma_s);
    ALLOC_D(d_cell_source,   h_source);
    ALLOC_D(d_dir_mu,        h_mu);
    ALLOC_D(d_dir_eta,       h_eta);
    ALLOC_D(d_dir_weight,    h_weight);

    COPY_D(d_cell_dx,       h_dx);
    COPY_D(d_cell_dy,       h_dy);
    COPY_D(d_cell_dt,       h_dt);
    COPY_D(d_cell_velocity, h_velocity);
    COPY_D(d_cell_sigma_t,  h_sigma_t);
    COPY_D(d_cell_sigma_s,  h_sigma_s);
    COPY_D(d_cell_source,   h_source);
    COPY_D(d_dir_mu,        h_mu);
    COPY_D(d_dir_eta,       h_eta);
    COPY_D(d_dir_weight,    h_weight);

#undef ALLOC_D
#undef COPY_D

    // Boundary conditions (only if non-empty) -------------------------
    auto upload_boundary = [](double** d_ptr, const std::vector<double>& h_vec,
                               const char* msg_alloc, const char* msg_copy) {
        if (h_vec.empty()) return;
        hip_check(hipMalloc(d_ptr, sizeof(double) * h_vec.size()), msg_alloc);
        hip_check(hipMemcpy(*d_ptr, h_vec.data(),
                            sizeof(double) * h_vec.size(),
                            hipMemcpyHostToDevice), msg_copy);
    };
    upload_boundary(&cache.d_boundary_west,  p.boundary.west,
                    "hipMalloc west",  "hipMemcpy west");
    upload_boundary(&cache.d_boundary_east,  p.boundary.east,
                    "hipMalloc east",  "hipMemcpy east");
    upload_boundary(&cache.d_boundary_south, p.boundary.south,
                    "hipMalloc south", "hipMemcpy south");
    upload_boundary(&cache.d_boundary_north, p.boundary.north,
                    "hipMalloc north", "hipMemcpy north");

    cache.sweep_data_valid = true;
}

// ---------------------------------------------------------------------------
// Allocate chunk-sized temporary buffers and flux state buffers
// ---------------------------------------------------------------------------
void allocate_device_buffers(const SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p   = state.problem;
    cache.n              = p.cell_block_size();
    cache.batch_count    = p.num_cells();
    cache.stride_a       = static_cast<std::size_t>(p.cell_block_elems());
    cache.stride_b       = static_cast<std::size_t>(p.cell_block_size());
    cache.stride_p       = static_cast<std::size_t>(p.cell_block_size());

    // rocBLAS handle
    if (!cache.rocblas_handle) {
        rocblas_handle h = nullptr;
        rocblas_check(rocblas_create_handle(&h), "rocBLAS handle creation failed.");
        cache.rocblas_handle = reinterpret_cast<void*>(h);
    }
    rocblas_check(rocblas_set_pointer_mode(as_handle(cache.rocblas_handle),
                                            rocblas_pointer_mode_host),
                  "rocBLAS pointer mode setup failed.");

    // Chunk-sized LU workspace
    if (!cache.d_lu_chunk) {
        cache.chunk_size = compute_chunk_size(p.num_cells(), cache.n);
        std::cout << "ROCm chunk size: " << cache.chunk_size
                  << " cells  (n=" << cache.n
                  << ", peak LU buffer = "
                  << (static_cast<double>(cache.chunk_size) *
                      cache.stride_a * sizeof(double) / (1024.0 * 1024.0))
                  << " MB)\n";

        const std::size_t a_bytes = sizeof(double) * cache.stride_a * cache.chunk_size;
        const std::size_t p_bytes = sizeof(int)    * cache.stride_p * cache.chunk_size;
        const std::size_t i_bytes = sizeof(int)    * cache.chunk_size;
        hip_check(hipMalloc(&cache.d_lu_chunk,     a_bytes), "hipMalloc d_lu_chunk");
        hip_check(hipMalloc(&cache.d_pivots_chunk, p_bytes), "hipMalloc d_pivots_chunk");
        hip_check(hipMalloc(&cache.d_info_chunk,   i_bytes), "hipMalloc d_info_chunk");
    }

    // Flux state buffers (total_unknowns each)
    if (!cache.d_rhs) {
        const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
        hip_check(hipMalloc(&cache.d_rhs,       b_bytes), "hipMalloc d_rhs");
        hip_check(hipMalloc(&cache.d_flux_last,  b_bytes), "hipMalloc d_flux_last");
        hip_check(hipMalloc(&cache.d_rhs_const,  b_bytes), "hipMalloc d_rhs_const");
        hip_check(hipMalloc(&cache.d_work,       b_bytes), "hipMalloc d_work");
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_add_upwind_rhs(const SolverState2D& state, RocmLUCache& cache,
                            double* d_rhs, const double* d_flux)
{
    const Problem2D& p   = state.problem;
    const int total      = p.num_cells() * p.groups * p.num_dirs();
    const int block      = 256;
    const int grid       = (total + block - 1) / block;
    hipLaunchKernelGGL(add_upwind_inflow_rhs_kernel,
                       dim3(grid), dim3(block), 0, 0,
                       d_rhs, d_flux,
                       p.nx, p.ny, p.groups, p.num_dirs(),
                       cache.d_cell_dx, cache.d_cell_dy,
                       cache.d_dir_mu,  cache.d_dir_eta,
                       cache.d_boundary_west,  cache.d_boundary_east,
                       cache.d_boundary_south, cache.d_boundary_north,
                       p.boundary.west.empty()  ? 0 : 1,
                       p.boundary.east.empty()  ? 0 : 1,
                       p.boundary.south.empty() ? 0 : 1,
                       p.boundary.north.empty() ? 0 : 1);
    hip_check(hipGetLastError(), "HIP launch: add_upwind_inflow_rhs_kernel");
}

void launch_build_constant_rhs(const SolverState2D& state, RocmLUCache& cache,
                                double* d_rhs_const, const double* d_flux_prev)
{
    const Problem2D& p = state.problem;
    const int total    = p.num_cells() * p.groups * p.num_dirs();
    const int block    = 256;
    const int grid     = (total + block - 1) / block;
    hipLaunchKernelGGL(build_constant_rhs_kernel,
                       dim3(grid), dim3(block), 0, 0,
                       d_rhs_const, d_flux_prev,
                       p.num_cells(), p.groups, p.num_dirs(),
                       cache.d_cell_dx, cache.d_cell_dy, cache.d_cell_dt,
                       cache.d_cell_velocity, cache.d_cell_source);
    hip_check(hipGetLastError(), "HIP launch: build_constant_rhs_kernel");
}

// ---------------------------------------------------------------------------
// Chunk-based assemble + factor + solve
// ---------------------------------------------------------------------------
// Processes all ncells in chunks of at most cache.chunk_size:
//   ① GPU assembles C cell matrices into the chunk buffer
//   ② rocsolver_dgetrf_strided_batched factors the C matrices
//   ③ rocsolver_dgetrs_strided_batched solves in-place against d_rhs[c*n..]
//
// After each chunk, the LU factors are gone — only the solutions remain in
// d_rhs.  This keeps peak device memory at chunk_size × n² instead of
// ncells × n².

void solve_all_cells_chunked(const SolverState2D& state, RocmLUCache& cache,
                              double* d_rhs)
{
    const Problem2D& p  = state.problem;
    const int n         = cache.n;
    const int ncells    = p.num_cells();
    const int chunk     = cache.chunk_size;
    rocblas_handle h    = as_handle(cache.rocblas_handle);

    for (int c_start = 0; c_start < ncells; c_start += chunk) {
        const int c_len = std::min(chunk, ncells - c_start);

        // ① Assemble cell matrices for this chunk
        hipLaunchKernelGGL(assemble_cell_matrices_chunk_kernel,
                           dim3(c_len), dim3(kAssemblyBlockDim), 0, 0,
                           cache.d_lu_chunk,
                           c_start, c_len, n,
                           p.groups, p.num_dirs(),
                           cache.d_cell_dx, cache.d_cell_dy, cache.d_cell_dt,
                           cache.d_cell_velocity,
                           cache.d_cell_sigma_t,
                           cache.d_cell_sigma_s,
                           cache.d_dir_mu, cache.d_dir_eta, cache.d_dir_weight);
        hip_check(hipGetLastError(), "HIP launch: assemble_cell_matrices_chunk_kernel");

        // ② Factor
        rocblas_check(
            rocsolver_dgetrf_strided_batched(
                h, n, n,
                cache.d_lu_chunk, n,
                static_cast<rocblas_stride>(cache.stride_a),
                cache.d_pivots_chunk,
                static_cast<rocblas_stride>(cache.stride_p),
                cache.d_info_chunk,
                c_len),
            "rocsolver_dgetrf_strided_batched (chunk)");

        // ③ Solve (modifies d_rhs[c_start*n .. (c_start+c_len)*n - 1] in place)
        rocblas_check(
            rocsolver_dgetrs_strided_batched(
                h, rocblas_operation_none, n, 1,
                cache.d_lu_chunk, n,
                static_cast<rocblas_stride>(cache.stride_a),
                cache.d_pivots_chunk,
                static_cast<rocblas_stride>(cache.stride_p),
                d_rhs + static_cast<long long>(c_start) * n, n,
                static_cast<rocblas_stride>(cache.stride_b),
                c_len),
            "rocsolver_dgetrs_strided_batched (chunk)");
        // d_lu_chunk is now implicitly discarded — no free needed.
    }
    // Sync once after all chunks so the caller sees the complete solution.
    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize after chunked solve");
}

} // namespace

// ---------------------------------------------------------------------------
// Public API  —  factor_cells_rocm
// ---------------------------------------------------------------------------
// In the new design this function uploads cell data and allocates buffers.
// No LU factorisation happens here; it is fused into every solve iteration.
// The function keeps its original name so call sites in example drivers do
// not need to change.

void factor_cells_rocm(const SolverState2D& state, RocmLUCache& cache) {
    ensure_cell_data_rocm(state, cache);
    allocate_device_buffers(state, cache);
    cache.valid = true;
}

// ---------------------------------------------------------------------------
// Public API  —  solve_cells_rocm
// ---------------------------------------------------------------------------
// Convenience wrapper that does a single assemble+factor+solve pass over all
// cells.  Usable from test drivers or TRT outer loops that want to call the
// GPU solver directly with a pre-built host RHS.

void solve_cells_rocm(const SolverState2D& state, RocmLUCache& cache,
                       std::vector<double>& rhs) {
    if (!cache.valid) factor_cells_rocm(state, cache);

    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    hip_check(hipMemcpy(cache.d_rhs, rhs.data(), b_bytes, hipMemcpyHostToDevice),
              "hipMemcpy: RHS upload");

    solve_all_cells_chunked(state, cache, cache.d_rhs);

    hip_check(hipMemcpy(rhs.data(), cache.d_rhs, b_bytes, hipMemcpyDeviceToHost),
              "hipMemcpy: RHS download");
}

// ---------------------------------------------------------------------------
// One time step  —  entire source-iteration loop on the GPU
// ---------------------------------------------------------------------------

IterationStats run_one_timestep_rocm(SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;

    if (!cache.valid) factor_cells_rocm(state, cache);

    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    rocblas_handle h = as_handle(cache.rocblas_handle);

    // ------------------------------------------------------------------
    // Build the constant RHS on the GPU from flux_previous
    // ------------------------------------------------------------------
    hip_check(hipMemcpy(cache.d_flux_last, state.flux_previous.data(),
                        b_bytes, hipMemcpyHostToDevice),
              "hipMemcpy: flux_previous upload");

    launch_build_constant_rhs(state, cache, cache.d_rhs_const, cache.d_flux_last);

    // ------------------------------------------------------------------
    // Initialise the source-iteration starting point
    // ------------------------------------------------------------------
    if (p.initialize_from_previous) {
        hip_check(hipMemcpy(cache.d_flux_last, state.flux_previous.data(),
                            b_bytes, hipMemcpyHostToDevice),
                  "hipMemcpy: initial iterate (previous)");
    } else {
        hip_check(hipMemset(cache.d_flux_last, 0, b_bytes),
                  "hipMemset: initial iterate (zero)");
    }

    // ------------------------------------------------------------------
    // Source-iteration loop (entirely on GPU)
    // ------------------------------------------------------------------
    IterationStats stats{};
    const double neg_one = -1.0;

    for (int it = 0; it < p.max_iters; ++it) {
        // ① Reset d_rhs to the timestep-constant part
        hip_check(hipMemcpy(cache.d_rhs, cache.d_rhs_const,
                            b_bytes, hipMemcpyDeviceToDevice),
                  "hipMemcpy: rhs_const copy");

        // ② Add upwind inflow from the previous iterate (Jacobi-in-space)
        launch_add_upwind_rhs(state, cache, cache.d_rhs, cache.d_flux_last);

        // ③ Chunked assemble + factor + solve for every cell
        solve_all_cells_chunked(state, cache, cache.d_rhs);
        // d_rhs now holds the new iterate

        // ④ Convergence check:  err = ||d_rhs - d_flux_last|| / ||d_flux_last||
        //    Use d_work as scratch for the difference vector.
        hip_check(hipMemcpy(cache.d_work, cache.d_rhs,
                            b_bytes, hipMemcpyDeviceToDevice),
                  "hipMemcpy: d_work copy for convergence");
        rocblas_check(
            rocblas_daxpy(h, p.total_unknowns(),
                          &neg_one,
                          cache.d_flux_last, 1,
                          cache.d_work,      1),
            "rocblas_daxpy: convergence difference");

        double numer = 0.0, denom = 0.0;
        rocblas_check(
            rocblas_dnrm2(h, p.total_unknowns(), cache.d_work,      1, &numer),
            "rocblas_dnrm2: numerator");
        rocblas_check(
            rocblas_dnrm2(h, p.total_unknowns(), cache.d_flux_last, 1, &denom),
            "rocblas_dnrm2: denominator");

        stats.final_error    = (denom == 0.0) ? numer : (numer / denom);
        stats.iterations     = it + 1;
        stats.spectral_radius = (stats.error_previous > 0.0)
                                ? (stats.final_error / stats.error_previous)
                                : 0.0;

        // ⑤ Swap d_rhs ↔ d_flux_last so the new iterate becomes the input
        //    for the next iteration, at zero extra cost.
        std::swap(cache.d_rhs, cache.d_flux_last);

        if (stats.final_error < p.convergence_tol) break;
        stats.iterate();
    }

    // ------------------------------------------------------------------
    // Download the converged flux and update state
    // ------------------------------------------------------------------
    state.flux_last.resize(p.total_unknowns());
    hip_check(hipMemcpy(state.flux_last.data(), cache.d_flux_last,
                        b_bytes, hipMemcpyDeviceToHost),
              "hipMemcpy: converged flux download");
    state.flux_previous = state.flux_last;
    return stats;
}

// ---------------------------------------------------------------------------
// Full time loop
// ---------------------------------------------------------------------------

std::vector<TimestepRecord2D> run_time_rocm(SolverState2D& state,
                                             RocmLUCache& cache,
                                             const TransportOutputFiles2D& outputs)
{
    std::filesystem::create_directories(outputs.output_dir);

    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state),
        ParaviewSeriesConfig2D{outputs.output_dir, outputs.series_name,
                               outputs.write_pvd_every_step});

    const double dt = (state.problem.time_step > 0.0)
                    ? state.problem.time_step
                    : (state.cells.empty() ? 0.0 : state.cells.front().dt);
    double time = 0.0;
    std::vector<TimestepRecord2D> history;
    history.reserve(state.problem.num_time_steps);

    for (int step = 0; step < state.problem.num_time_steps; ++step) {
        IterationStats stats = run_one_timestep_rocm(state, cache);
        time += dt;
        history.push_back(TimestepRecord2D{step, time, stats});

        std::vector<CellScalarField2D> fields;
        if (outputs.save_flux) {
            append_fields(fields,
                make_angular_flux_group_dir_fields(state, state.flux_previous,
                                                   "angular_flux"));
            append_fields(fields,
                make_scalar_flux_group_fields(state, state.flux_previous,
                                              "scalar_flux_g"));
        }
        writer.write_step(step, time, fields);

        std::cout << "step " << step
                  << "  time="       << time
                  << "  iters="      << stats.iterations
                  << "  rho="        << stats.spectral_radius
                  << "  err="        << stats.final_error << '\n';
    }

    write_transport_summary_json(outputs.summary_json, state, history,
                                 "rocm_chunked", writer.pvd_path());

    std::cout << "\nWrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << outputs.summary_json << '\n';
    return history;
}

// ---------------------------------------------------------------------------
// Lightweight per–nonlinear-iteration refresh for TRT outer loops
// ---------------------------------------------------------------------------
// In a TRT Picard iteration sigma_t, sigma_s, and source change every
// nonlinear step (because alpha(T) changes), while geometry, velocity,
// quadrature, and boundary conditions are fixed.
//
// This function re-uploads only the three time-varying fields without
// reallocating device memory.  Call it instead of invalidating cache.valid.
//
// Precondition: factor_cells_rocm() has been called at least once so that
// the device arrays are already allocated.

void refresh_cell_opacities_rocm(const SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;
    const int nc = p.num_cells();
    const int G  = p.groups;

    std::vector<double> h_sigma_t(nc * G);
    std::vector<double> h_sigma_s(nc * G * G);
    std::vector<double> h_source (p.total_unknowns());

    for (int c = 0; c < nc; ++c) {
        const Cell2D& cell = state.cells[c];
        for (int g = 0; g < G; ++g)
            h_sigma_t[c * G + g] = cell.sigma_t[g];
        for (int g_to = 0; g_to < G; ++g_to)
            for (int g_from = 0; g_from < G; ++g_from)
                h_sigma_s[(c * G + g_to) * G + g_from] =
                    cell.sigma_s[g_to * G + g_from];
        const double* src = cell.source.data();
        std::copy(src, src + p.cell_block_size(),
                  h_source.begin() + c * p.cell_block_size());
    }

    hip_check(hipMemcpy(cache.d_cell_sigma_t, h_sigma_t.data(),
                        sizeof(double) * h_sigma_t.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy refresh: sigma_t");
    hip_check(hipMemcpy(cache.d_cell_sigma_s, h_sigma_s.data(),
                        sizeof(double) * h_sigma_s.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy refresh: sigma_s");
    hip_check(hipMemcpy(cache.d_cell_source, h_source.data(),
                        sizeof(double) * h_source.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy refresh: source");
}

// ---------------------------------------------------------------------------
// Cache teardown
// ---------------------------------------------------------------------------

void destroy_rocm_cache(RocmLUCache& cache) {
    auto safe_free = [](auto** ptr) {
        if (*ptr) { hipFree(*ptr); *ptr = nullptr; }
    };

    // Chunk buffers (new)
    safe_free(&cache.d_lu_chunk);
    safe_free(&cache.d_pivots_chunk);
    safe_free(&cache.d_info_chunk);

    // Flux state
    safe_free(&cache.d_rhs);
    safe_free(&cache.d_flux_last);
    safe_free(&cache.d_rhs_const);
    safe_free(&cache.d_work);

    // Cell data (geometry + cross-sections)
    safe_free(&cache.d_cell_dx);
    safe_free(&cache.d_cell_dy);
    safe_free(&cache.d_cell_dt);
    safe_free(&cache.d_cell_velocity);
    safe_free(&cache.d_cell_sigma_t);
    safe_free(&cache.d_cell_sigma_s);
    safe_free(&cache.d_cell_source);
    safe_free(&cache.d_dir_mu);
    safe_free(&cache.d_dir_eta);
    safe_free(&cache.d_dir_weight);

    // Boundaries
    safe_free(&cache.d_boundary_west);
    safe_free(&cache.d_boundary_east);
    safe_free(&cache.d_boundary_south);
    safe_free(&cache.d_boundary_north);

    if (cache.rocblas_handle) {
        rocblas_destroy_handle(as_handle(cache.rocblas_handle));
        cache.rocblas_handle = nullptr;
    }

    cache.chunk_size      = 0;
    cache.valid           = false;
    cache.sweep_data_valid = false;
}

} // namespace therefore2d

#endif // THEREFORE2D_ENABLE_ROCM
