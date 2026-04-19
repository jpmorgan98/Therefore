#include "transport2d.hpp"
#include "output.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
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

rocblas_handle as_handle(void* handle) {
    return reinterpret_cast<rocblas_handle>(handle);
}

void hip_check(hipError_t status, const char* msg) {
    if (status != hipSuccess) {
        throw std::runtime_error(msg);
    }
}

void rocblas_check(rocblas_status status, const char* msg) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(msg);
    }
}

__device__ __forceinline__ int packed_base(int cell, int group, int dir, int groups, int num_dirs) {
    return cell * (groups * num_dirs * kDofsPerAngleGroup2D)
         + ((group * num_dirs + dir) * kDofsPerAngleGroup2D);
}

__device__ __forceinline__ int west_east_face_base(int boundary_j, int group, int dir, int groups, int num_dirs) {
    return (((boundary_j * groups + group) * num_dirs + dir) * 4);
}

__device__ __forceinline__ int south_north_face_base(int boundary_i, int group, int dir, int groups, int num_dirs) {
    return (((boundary_i * groups + group) * num_dirs + dir) * 4);
}

__global__ void add_upwind_inflow_rhs_kernel(
    double* rhs,
    const double* iterate_flux,
    int nx,
    int ny,
    int groups,
    int num_dirs,
    const double* cell_dx,
    const double* cell_dy,
    const double* dir_mu,
    const double* dir_eta,
    const double* boundary_west,
    const double* boundary_east,
    const double* boundary_south,
    const double* boundary_north,
    int has_west,
    int has_east,
    int has_south,
    int has_north) {

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = nx * ny * groups * num_dirs;
    if (tid >= total) {
        return;
    }

    const int dir = tid % num_dirs;
    const int tmp0 = tid / num_dirs;
    const int group = tmp0 % groups;
    const int cell = tmp0 / groups;

    const int i = cell % nx;
    const int j = cell / nx;
    const double mu = dir_mu[dir];
    const double eta = dir_eta[dir];
    const double ax = 0.5 * fabs(mu) * cell_dy[cell];
    const double ay = 0.5 * fabs(eta) * cell_dx[cell];

    const int dst = packed_base(cell, group, dir, groups, num_dirs);

    if (mu >= 0.0) {
        if (i > 0) {
            const int src = packed_base(cell - 1, group, dir, groups, num_dirs);
            rhs[dst + 0] += ax * iterate_flux[src + 1];
            rhs[dst + 2] += ax * iterate_flux[src + 3];
            rhs[dst + 4] += ax * iterate_flux[src + 5];
            rhs[dst + 6] += ax * iterate_flux[src + 7];
        } else if (has_west) {
            const int off = west_east_face_base(j, group, dir, groups, num_dirs);
            rhs[dst + 0] += ax * boundary_west[off + 0];
            rhs[dst + 2] += ax * boundary_west[off + 1];
            rhs[dst + 4] += ax * boundary_west[off + 2];
            rhs[dst + 6] += ax * boundary_west[off + 3];
        }
    } else {
        if (i + 1 < nx) {
            const int src = packed_base(cell + 1, group, dir, groups, num_dirs);
            rhs[dst + 1] += ax * iterate_flux[src + 0];
            rhs[dst + 3] += ax * iterate_flux[src + 2];
            rhs[dst + 5] += ax * iterate_flux[src + 4];
            rhs[dst + 7] += ax * iterate_flux[src + 6];
        } else if (has_east) {
            const int off = west_east_face_base(j, group, dir, groups, num_dirs);
            rhs[dst + 1] += ax * boundary_east[off + 0];
            rhs[dst + 3] += ax * boundary_east[off + 1];
            rhs[dst + 5] += ax * boundary_east[off + 2];
            rhs[dst + 7] += ax * boundary_east[off + 3];
        }
    }

    if (eta >= 0.0) {
        if (j > 0) {
            const int src = packed_base(cell - nx, group, dir, groups, num_dirs);
            rhs[dst + 0] += ay * iterate_flux[src + 2];
            rhs[dst + 1] += ay * iterate_flux[src + 3];
            rhs[dst + 4] += ay * iterate_flux[src + 6];
            rhs[dst + 5] += ay * iterate_flux[src + 7];
        } else if (has_south) {
            const int off = south_north_face_base(i, group, dir, groups, num_dirs);
            rhs[dst + 0] += ay * boundary_south[off + 0];
            rhs[dst + 1] += ay * boundary_south[off + 1];
            rhs[dst + 4] += ay * boundary_south[off + 2];
            rhs[dst + 5] += ay * boundary_south[off + 3];
        }
    } else {
        if (j + 1 < ny) {
            const int src = packed_base(cell + nx, group, dir, groups, num_dirs);
            rhs[dst + 2] += ay * iterate_flux[src + 0];
            rhs[dst + 3] += ay * iterate_flux[src + 1];
            rhs[dst + 6] += ay * iterate_flux[src + 4];
            rhs[dst + 7] += ay * iterate_flux[src + 5];
        } else if (has_north) {
            const int off = south_north_face_base(i, group, dir, groups, num_dirs);
            rhs[dst + 2] += ay * boundary_north[off + 0];
            rhs[dst + 3] += ay * boundary_north[off + 1];
            rhs[dst + 6] += ay * boundary_north[off + 2];
            rhs[dst + 7] += ay * boundary_north[off + 3];
        }
    }
}

__global__ void build_constant_rhs_kernel(
    double* rhs_const,
    const double* flux_previous,
    int num_cells,
    int groups,
    int num_dirs,
    const double* cell_dx,
    const double* cell_dy,
    const double* cell_dt,
    const double* cell_velocity,
    const double* cell_source) {

    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = num_cells * groups * num_dirs;
    if (tid >= total) {
        return;
    }

    const int dir = tid % num_dirs;
    const int tmp0 = tid / num_dirs;
    const int group = tmp0 % groups;
    const int cell = tmp0 / groups;

    const double volume = cell_dx[cell] * cell_dy[cell];
    const double tau_half = 0.5 * volume / (cell_velocity[cell * groups + group] * cell_dt[cell]);
    const int base = packed_base(cell, group, dir, groups, num_dirs);
    const double source_scale = volume / 4.0;

    for (int corner = 0; corner < 4; ++corner) {
        rhs_const[base + corner] = source_scale * cell_source[base + corner]
                                 + tau_half * flux_previous[base + corner];
        rhs_const[base + 4 + corner] = source_scale * cell_source[base + 4 + corner];
    }
}

void ensure_sweep_data_rocm(const SolverState2D& state, RocmLUCache& cache) {
    if (cache.sweep_data_valid) {
        return;
    }

    const Problem2D& p = state.problem;
    std::vector<double> host_dx(p.num_cells());
    std::vector<double> host_dy(p.num_cells());
    std::vector<double> host_dt(p.num_cells());
    std::vector<double> host_velocity(p.num_cells() * p.groups);
    std::vector<double> host_source(p.total_unknowns());
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        host_dx[cell] = state.cells[cell].dx;
        host_dy[cell] = state.cells[cell].dy;
        host_dt[cell] = state.cells[cell].dt;
        for (int g = 0; g < p.groups; ++g) {
            host_velocity[cell * p.groups + g] = state.cells[cell].velocity[g];
        }
        const double* src = state.cells[cell].source.data();
        std::copy(src,
                  src + p.cell_block_size(),
                  host_source.begin() + cell * p.cell_block_size());
    }

    std::vector<double> host_mu(p.num_dirs());
    std::vector<double> host_eta(p.num_dirs());
    for (int d = 0; d < p.num_dirs(); ++d) {
        host_mu[d] = p.directions[d].mu;
        host_eta[d] = p.directions[d].eta;
    }

    hip_check(hipMalloc(&cache.d_cell_dx, sizeof(double) * host_dx.size()), "hipMalloc failed for d_cell_dx.");
    hip_check(hipMalloc(&cache.d_cell_dy, sizeof(double) * host_dy.size()), "hipMalloc failed for d_cell_dy.");
    hip_check(hipMalloc(&cache.d_cell_dt, sizeof(double) * host_dt.size()), "hipMalloc failed for d_cell_dt.");
    hip_check(hipMalloc(&cache.d_cell_velocity, sizeof(double) * host_velocity.size()), "hipMalloc failed for d_cell_velocity.");
    hip_check(hipMalloc(&cache.d_cell_source, sizeof(double) * host_source.size()), "hipMalloc failed for d_cell_source.");
    hip_check(hipMalloc(&cache.d_dir_mu, sizeof(double) * host_mu.size()), "hipMalloc failed for d_dir_mu.");
    hip_check(hipMalloc(&cache.d_dir_eta, sizeof(double) * host_eta.size()), "hipMalloc failed for d_dir_eta.");

    hip_check(hipMemcpy(cache.d_cell_dx, host_dx.data(), sizeof(double) * host_dx.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_cell_dx.");
    hip_check(hipMemcpy(cache.d_cell_dy, host_dy.data(), sizeof(double) * host_dy.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_cell_dy.");
    hip_check(hipMemcpy(cache.d_cell_dt, host_dt.data(), sizeof(double) * host_dt.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_cell_dt.");
    hip_check(hipMemcpy(cache.d_cell_velocity, host_velocity.data(), sizeof(double) * host_velocity.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_cell_velocity.");
    hip_check(hipMemcpy(cache.d_cell_source, host_source.data(), sizeof(double) * host_source.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_cell_source.");
    hip_check(hipMemcpy(cache.d_dir_mu, host_mu.data(), sizeof(double) * host_mu.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_dir_mu.");
    hip_check(hipMemcpy(cache.d_dir_eta, host_eta.data(), sizeof(double) * host_eta.size(), hipMemcpyHostToDevice),
              "hipMemcpy failed for d_dir_eta.");

    if (!p.boundary.west.empty()) {
        hip_check(hipMalloc(&cache.d_boundary_west, sizeof(double) * p.boundary.west.size()),
                  "hipMalloc failed for d_boundary_west.");
        hip_check(hipMemcpy(cache.d_boundary_west, p.boundary.west.data(), sizeof(double) * p.boundary.west.size(), hipMemcpyHostToDevice),
                  "hipMemcpy failed for d_boundary_west.");
    }
    if (!p.boundary.east.empty()) {
        hip_check(hipMalloc(&cache.d_boundary_east, sizeof(double) * p.boundary.east.size()),
                  "hipMalloc failed for d_boundary_east.");
        hip_check(hipMemcpy(cache.d_boundary_east, p.boundary.east.data(), sizeof(double) * p.boundary.east.size(), hipMemcpyHostToDevice),
                  "hipMemcpy failed for d_boundary_east.");
    }
    if (!p.boundary.south.empty()) {
        hip_check(hipMalloc(&cache.d_boundary_south, sizeof(double) * p.boundary.south.size()),
                  "hipMalloc failed for d_boundary_south.");
        hip_check(hipMemcpy(cache.d_boundary_south, p.boundary.south.data(), sizeof(double) * p.boundary.south.size(), hipMemcpyHostToDevice),
                  "hipMemcpy failed for d_boundary_south.");
    }
    if (!p.boundary.north.empty()) {
        hip_check(hipMalloc(&cache.d_boundary_north, sizeof(double) * p.boundary.north.size()),
                  "hipMalloc failed for d_boundary_north.");
        hip_check(hipMemcpy(cache.d_boundary_north, p.boundary.north.data(), sizeof(double) * p.boundary.north.size(), hipMemcpyHostToDevice),
                  "hipMemcpy failed for d_boundary_north.");
    }

    cache.sweep_data_valid = true;
}

void add_upwind_inflow_rhs_rocm(const SolverState2D& state,
                                RocmLUCache& cache,
                                double* d_rhs,
                                const double* d_iterate_flux) {
    const Problem2D& p = state.problem;
    ensure_sweep_data_rocm(state, cache);

    const int total = p.num_cells() * p.groups * p.num_dirs();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    hipLaunchKernelGGL(add_upwind_inflow_rhs_kernel,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       0,
                       d_rhs,
                       d_iterate_flux,
                       p.nx,
                       p.ny,
                       p.groups,
                       p.num_dirs(),
                       cache.d_cell_dx,
                       cache.d_cell_dy,
                       cache.d_dir_mu,
                       cache.d_dir_eta,
                       cache.d_boundary_west,
                       cache.d_boundary_east,
                       cache.d_boundary_south,
                       cache.d_boundary_north,
                       p.boundary.west.empty() ? 0 : 1,
                       p.boundary.east.empty() ? 0 : 1,
                       p.boundary.south.empty() ? 0 : 1,
                       p.boundary.north.empty() ? 0 : 1);
    hip_check(hipGetLastError(), "HIP launch failed for add_upwind_inflow_rhs_kernel.");
}

void build_constant_rhs_rocm_device(const SolverState2D& state,
                                    RocmLUCache& cache,
                                    double* d_rhs_const,
                                    const double* d_flux_previous) {
    const Problem2D& p = state.problem;
    ensure_sweep_data_rocm(state, cache);

    const int total = p.num_cells() * p.groups * p.num_dirs();
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    hipLaunchKernelGGL(build_constant_rhs_kernel,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       0,
                       d_rhs_const,
                       d_flux_previous,
                       p.num_cells(),
                       p.groups,
                       p.num_dirs(),
                       cache.d_cell_dx,
                       cache.d_cell_dy,
                       cache.d_cell_dt,
                       cache.d_cell_velocity,
                       cache.d_cell_source);
    hip_check(hipGetLastError(), "HIP launch failed for build_constant_rhs_kernel.");
}

void solve_cells_rocm_device(const SolverState2D& state, RocmLUCache& cache, double* d_rhs) {
    if (!cache.valid) {
        factor_cells_rocm(state, cache);
    }

    rocblas_check(
        rocsolver_dgetrs_strided_batched(
            as_handle(cache.rocblas_handle),
            rocblas_operation_none,
            cache.n,
            1,
            cache.d_lu,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_a),
            cache.d_pivots,
            static_cast<rocblas_stride>(cache.stride_p),
            d_rhs,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_b),
            cache.batch_count),
        "rocsolver_dgetrs_strided_batched failed.");

    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize failed after dgetrs.");
}

} // namespace

void factor_cells_rocm(const SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;
    cache.n = p.cell_block_size();
    cache.batch_count = p.num_cells();
    cache.stride_a = static_cast<std::size_t>(p.cell_block_elems());
    cache.stride_b = static_cast<std::size_t>(p.cell_block_size());
    cache.stride_p = static_cast<std::size_t>(p.cell_block_size());

    if (!cache.rocblas_handle) {
        rocblas_handle handle = nullptr;
        rocblas_check(rocblas_create_handle(&handle), "rocBLAS handle creation failed.");
        cache.rocblas_handle = reinterpret_cast<void*>(handle);
    }

    const std::size_t a_bytes = sizeof(double) * cache.stride_a * cache.batch_count;
    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    const std::size_t p_bytes = sizeof(int) * cache.stride_p * cache.batch_count;
    const std::size_t i_bytes = sizeof(int) * cache.batch_count;

    if (!cache.d_lu) {
        hip_check(hipMalloc(&cache.d_lu, a_bytes), "hipMalloc failed for d_lu.");
        hip_check(hipMalloc(&cache.d_rhs, b_bytes), "hipMalloc failed for d_rhs.");
        hip_check(hipMalloc(&cache.d_flux_last, b_bytes), "hipMalloc failed for d_flux_last.");
        hip_check(hipMalloc(&cache.d_rhs_const, b_bytes), "hipMalloc failed for d_rhs_const.");
        hip_check(hipMalloc(&cache.d_work, b_bytes), "hipMalloc failed for d_work.");
        hip_check(hipMalloc(&cache.d_pivots, p_bytes), "hipMalloc failed for d_pivots.");
        hip_check(hipMalloc(&cache.d_info, i_bytes), "hipMalloc failed for d_info.");
    }

    ensure_sweep_data_rocm(state, cache);

    hip_check(hipMemcpy(cache.d_lu, state.cell_matrices.data(), a_bytes, hipMemcpyHostToDevice), "hipMemcpy failed for LU upload.");

    rocblas_check(
        rocsolver_dgetrf_strided_batched(
            as_handle(cache.rocblas_handle),
            cache.n,
            cache.n,
            cache.d_lu,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_a),
            cache.d_pivots,
            static_cast<rocblas_stride>(cache.stride_p),
            cache.d_info,
            cache.batch_count),
        "rocsolver_dgetrf_strided_batched failed.");

    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize failed after LU factorization.");
    cache.valid = true;
}

void solve_cells_rocm(const SolverState2D& state, RocmLUCache& cache, std::vector<double>& rhs) {
    if (!cache.valid) {
        factor_cells_rocm(state, cache);
    }

    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    hip_check(hipMemcpy(cache.d_rhs, rhs.data(), b_bytes, hipMemcpyHostToDevice), "hipMemcpy failed for RHS upload.");
    solve_cells_rocm_device(state, cache, cache.d_rhs);
    hip_check(hipMemcpy(rhs.data(), cache.d_rhs, b_bytes, hipMemcpyDeviceToHost), "hipMemcpy failed for RHS download.");
}

IterationStats run_one_timestep_rocm(SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;
    if (!cache.valid || !p.reuse_factorization) {
        factor_cells_rocm(state, cache);
    }

    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    hip_check(hipMemcpy(cache.d_rhs, state.flux_previous.data(), b_bytes, hipMemcpyHostToDevice),
              "hipMemcpy failed for previous-time solution upload.");
    build_constant_rhs_rocm_device(state, cache, cache.d_rhs_const, cache.d_rhs);

    state.flux_last = p.initialize_from_previous ? state.flux_previous : std::vector<double>(p.total_unknowns(), 0.0);
    if (p.initialize_from_previous) {
        hip_check(hipMemcpy(cache.d_flux_last, cache.d_rhs, b_bytes, hipMemcpyDeviceToDevice),
                  "hipMemcpy failed for initial iterate copy.");
    } else {
        hip_check(hipMemset(cache.d_flux_last, 0, b_bytes), "hipMemset failed for initial iterate zeroing.");
    }

    IterationStats stats{};
    const double alpha = -1.0;
    rocblas_check(rocblas_set_pointer_mode(as_handle(cache.rocblas_handle), rocblas_pointer_mode_host),
                  "rocBLAS pointer mode setup failed.");

    for (int it = 0; it < p.max_iters; ++it) {
        hip_check(hipMemcpy(cache.d_rhs, cache.d_rhs_const, b_bytes, hipMemcpyDeviceToDevice),
                  "hipMemcpy failed for device RHS reset.");
        add_upwind_inflow_rhs_rocm(state, cache, cache.d_rhs, cache.d_flux_last);
        solve_cells_rocm_device(state, cache, cache.d_rhs);

        hip_check(hipMemcpy(cache.d_work, cache.d_rhs, b_bytes, hipMemcpyDeviceToDevice),
                  "hipMemcpy failed for d_work copy.");
        rocblas_check(rocblas_daxpy(as_handle(cache.rocblas_handle),
                                    p.total_unknowns(),
                                    &alpha,
                                    cache.d_flux_last,
                                    1,
                                    cache.d_work,
                                    1),
                      "rocblas_daxpy failed while forming the iteration difference.");

        double numer_norm = 0.0;
        double denom_norm = 0.0;
        rocblas_check(rocblas_dnrm2(as_handle(cache.rocblas_handle),
                                    p.total_unknowns(),
                                    cache.d_work,
                                    1,
                                    &numer_norm),
                      "rocblas_dnrm2 failed for the iteration difference norm.");
        rocblas_check(rocblas_dnrm2(as_handle(cache.rocblas_handle),
                                    p.total_unknowns(),
                                    cache.d_flux_last,
                                    1,
                                    &denom_norm),
                      "rocblas_dnrm2 failed for the previous iterate norm.");

        stats.final_error = (denom_norm == 0.0) ? numer_norm : (numer_norm / denom_norm);
        stats.iterations = it + 1;
        if (stats.error_previous != 0.0) {
            stats.spectral_radius = stats.final_error / stats.error_previous;
        }
        std::swap(cache.d_flux_last, cache.d_rhs);

        if (stats.final_error < p.convergence_tol) {
            break;
        }

        stats.iterate();
    }

    state.flux_last.resize(p.total_unknowns());
    hip_check(hipMemcpy(state.flux_last.data(), cache.d_flux_last, b_bytes, hipMemcpyDeviceToHost),
              "hipMemcpy failed for final timestep solution download.");
    state.flux_previous = state.flux_last;
    return stats;
}

std::vector<TimestepRecord2D> run_time_rocm(SolverState2D& state,
                                           RocmLUCache& cache,
                                           const TransportOutputFiles2D& outputs) {
    auto write_transport_summary_json = [](const std::string& path,
                                           const SolverState2D& state_ref,
                                           const std::vector<TimestepRecord2D>& history,
                                           const std::string& backend_name,
                                           const std::string& pvd_path) {
        const std::filesystem::path out_path(path);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }

        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("Could not open summary JSON for writing: " + path);
        }

        out << std::setprecision(16);
        out << "{\n";
        out << "  \"backend\": \"" << backend_name << "\",\n";
        out << "  \"nx\": " << state_ref.problem.nx << ",\n";
        out << "  \"ny\": " << state_ref.problem.ny << ",\n";
        out << "  \"groups\": " << state_ref.problem.groups << ",\n";
        out << "  \"num_dirs\": " << state_ref.problem.num_dirs() << ",\n";
        out << "  \"cell_block_size\": " << state_ref.problem.cell_block_size() << ",\n";
        out << "  \"total_unknowns\": " << state_ref.problem.total_unknowns() << ",\n";
        out << "  \"paraview_pvd\": \"" << pvd_path << "\",\n";
        out << "  \"time_history\": [\n";
        for (std::size_t k = 0; k < history.size(); ++k) {
            const auto& rec = history[k];
            out << "    {\"step\": " << rec.step
                << ", \"time\": " << rec.time
                << ", \"iterations\": " << rec.stats.iterations
                << ", \"final_error\": " << rec.stats.final_error
                << ", \"spectral_radius\": " << rec.stats.spectral_radius << "}";
            if (k + 1 != history.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << "  ]\n";
        out << "}\n";
    };

    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state),
        ParaviewSeriesConfig2D{outputs.output_dir, outputs.series_name, outputs.write_pvd_every_step});

    const double dt = (state.problem.time_step > 0.0)
        ? state.problem.time_step
        : (state.cells.empty() ? 0.0 : state.cells.front().dt);
    double time = 0.0;
    std::vector<TimestepRecord2D> history;
    history.reserve(state.problem.num_time_steps);

    for (int step = 0; step < state.problem.num_time_steps; ++step) {
        std::cout << " TIME STEP: " << step << std::endl;
        IterationStats stats = run_one_timestep_rocm(state, cache);
        time += dt;
        history.push_back(TimestepRecord2D{step, time, stats});

        std::vector<CellScalarField2D> fields;
        append_fields(fields, make_angular_flux_group_dir_fields(state, state.flux_previous, "angular_flux"));
        append_fields(fields, make_scalar_flux_group_fields(state, state.flux_previous, "scalar_flux_g"));
        writer.write_step(step, time, fields);

        std::cout << "step " << step
                  << "  time=" << time
                  << "  iterations=" << stats.iterations
                  << "  spectral radius=" << stats.spectral_radius
                  << "  final_error=" << stats.final_error << '\n';
    }

    write_transport_summary_json(outputs.summary_json, state, history, "rocm", writer.pvd_path());

    std::cout << "\nWrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << outputs.summary_json << '\n';
    return history;
}

void destroy_rocm_cache(RocmLUCache& cache) {
    if (cache.d_lu) { hipFree(cache.d_lu); cache.d_lu = nullptr; }
    if (cache.d_rhs) { hipFree(cache.d_rhs); cache.d_rhs = nullptr; }
    if (cache.d_flux_last) { hipFree(cache.d_flux_last); cache.d_flux_last = nullptr; }
    if (cache.d_rhs_const) { hipFree(cache.d_rhs_const); cache.d_rhs_const = nullptr; }
    if (cache.d_work) { hipFree(cache.d_work); cache.d_work = nullptr; }
    if (cache.d_cell_dx) { hipFree(cache.d_cell_dx); cache.d_cell_dx = nullptr; }
    if (cache.d_cell_dy) { hipFree(cache.d_cell_dy); cache.d_cell_dy = nullptr; }
    if (cache.d_cell_dt) { hipFree(cache.d_cell_dt); cache.d_cell_dt = nullptr; }
    if (cache.d_cell_velocity) { hipFree(cache.d_cell_velocity); cache.d_cell_velocity = nullptr; }
    if (cache.d_cell_source) { hipFree(cache.d_cell_source); cache.d_cell_source = nullptr; }
    if (cache.d_dir_mu) { hipFree(cache.d_dir_mu); cache.d_dir_mu = nullptr; }
    if (cache.d_dir_eta) { hipFree(cache.d_dir_eta); cache.d_dir_eta = nullptr; }
    if (cache.d_boundary_west) { hipFree(cache.d_boundary_west); cache.d_boundary_west = nullptr; }
    if (cache.d_boundary_east) { hipFree(cache.d_boundary_east); cache.d_boundary_east = nullptr; }
    if (cache.d_boundary_south) { hipFree(cache.d_boundary_south); cache.d_boundary_south = nullptr; }
    if (cache.d_boundary_north) { hipFree(cache.d_boundary_north); cache.d_boundary_north = nullptr; }
    if (cache.d_pivots) { hipFree(cache.d_pivots); cache.d_pivots = nullptr; }
    if (cache.d_info) { hipFree(cache.d_info); cache.d_info = nullptr; }
    if (cache.rocblas_handle) {
        rocblas_destroy_handle(as_handle(cache.rocblas_handle));
        cache.rocblas_handle = nullptr;
    }
    cache.sweep_data_valid = false;
    cache.valid = false;
}

} // namespace therefore2d

#endif
