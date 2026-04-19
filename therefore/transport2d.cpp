#include "transport2d.hpp"
#include "output.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda,
             const int* ipiv, double* b, const int* ldb, int* info);
}

#ifdef _OPENMP
#include <omp.h>
#endif

namespace therefore2d {
namespace {

// ---------------------------------------------------------------------------
// Corner indices (spatial DOF ordering within a cell-face):
//   kLL = low-x,low-y   kLR = low-x,high-y
//   kUL = high-x,low-y  kUR = high-x,high-y
// ---------------------------------------------------------------------------
constexpr int kLL = 0;
constexpr int kLR = 1;
constexpr int kUL = 2;
constexpr int kUR = 3;

inline double& cm(std::vector<double>& a, int n, int row, int col) {
    return a[col * n + row];
}
inline const double& cm(const std::vector<double>& a, int n, int row, int col) {
    return a[col * n + row];
}
// Raw-pointer overloads used by the thread-local fused assemble+factor+solve path.
inline double& cm(double* a, int n, int row, int col) {
    return a[col * n + row];
}

// ---------------------------------------------------------------------------
// Streaming matrices for the upwind bilinear-DFEM scheme
// ---------------------------------------------------------------------------
// Each 4x4 block encodes the contribution of the upstream face to the four
// corner unknowns of the current cell.  Layout (row=destination corner):
//   mu > 0: information flows from west face (columns 0,2) into left corners
//   mu < 0: information flows from east face (columns 1,3) into right corners
// Similarly for eta in y.

std::array<double, 16> x_stream_block(double ax, bool positive_x) {
    if (positive_x) {
        // mu > 0: west inflow into left corners (LL←right-of-west, UL←right-of-west)
        return {
            +ax, 0.0, 0.0, 0.0,
            -ax, +ax, 0.0, 0.0,
            0.0, 0.0, +ax, 0.0,
            0.0, 0.0, -ax, +ax
        };
    }
    // mu < 0: east inflow into right corners
    return {
        +ax, -ax, 0.0, 0.0,
        0.0, +ax, 0.0, 0.0,
        0.0, 0.0, +ax, -ax,
        0.0, 0.0, 0.0, +ax
    };
}

std::array<double, 16> y_stream_block(double ay, bool positive_y) {
    if (positive_y) {
        // eta > 0: south inflow into bottom corners
        return {
            +ay, 0.0, 0.0, 0.0,
            0.0, +ay, 0.0, 0.0,
            -ay, 0.0, +ay, 0.0,
            0.0, -ay, 0.0, +ay
        };
    }
    // eta < 0: north inflow into top corners
    return {
        +ay, 0.0, -ay, 0.0,
        0.0, +ay, 0.0, -ay,
        0.0, 0.0, +ay, 0.0,
        0.0, 0.0, 0.0, +ay
    };
}

void add_small_block(std::vector<double>& a, int n, int row0, int col0,
                     const std::array<double, 16>& block) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            cm(a, n, row0 + r, col0 + c) += block[r * 4 + c];
}
void add_small_block(double* a, int n, int row0, int col0,
                     const std::array<double, 16>& block) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            cm(a, n, row0 + r, col0 + c) += block[r * 4 + c];
}

void add_identity_scaled(std::vector<double>& a, int n,
                         int row0, int col0, int count, double value) {
    for (int k = 0; k < count; ++k)
        cm(a, n, row0 + k, col0 + k) += value;
}
void add_identity_scaled(double* a, int n,
                         int row0, int col0, int count, double value) {
    for (int k = 0; k < count; ++k)
        cm(a, n, row0 + k, col0 + k) += value;
}

// ---------------------------------------------------------------------------
// Cell-level block assembly
// ---------------------------------------------------------------------------
// Assembles the (group, direction) block at (row0, row0) of the cell matrix.
// The 8×8 block corresponds to the two temporal moments of the 4 spatial
// corners, i.e., DOFs [low_corner×4] and [high_corner×4].
//
// Eq. (5) of the scheme (schematic):
//   A_ag = [[ Sx+Sy + gamma*I ,  tau/2 * I ],
//           [   -tau * I      ,  Sx+Sy + (gamma+tau)*I ]]
//
// where gamma = (V/4)*sigma_t, tau = V/(v*dt), Sx/Sy are streaming matrices.

void assemble_angle_group_block(std::vector<double>& cell_matrix,
                                int n, int row0,
                                const Cell2D& cell,
                                const Direction2D& dir,
                                int group) {
    const double volume   = cell.dx * cell.dy;
    const double gamma    = 0.25 * volume * cell.sigma_t[group];
    const double tau      = volume / (cell.velocity[group] * cell.dt);
    const double tau_half = 0.5 * tau;
    const double ax       = 0.5 * std::abs(dir.mu)  * cell.dy;
    const double ay       = 0.5 * std::abs(dir.eta) * cell.dx;

    auto kx = x_stream_block(ax, dir.mu  >= 0.0);
    auto ky = y_stream_block(ay, dir.eta >= 0.0);

    // Low temporal moment block (corners 0–3)
    add_small_block       (cell_matrix, n, row0 + 0, row0 + 0, kx);
    add_small_block       (cell_matrix, n, row0 + 0, row0 + 0, ky);
    add_identity_scaled   (cell_matrix, n, row0 + 0, row0 + 0, 4, gamma);
    add_identity_scaled   (cell_matrix, n, row0 + 0, row0 + 4, 4, tau_half);

    // High temporal moment block (corners 4–7)
    add_small_block       (cell_matrix, n, row0 + 4, row0 + 4, kx);
    add_small_block       (cell_matrix, n, row0 + 4, row0 + 4, ky);
    add_identity_scaled   (cell_matrix, n, row0 + 4, row0 + 4, 4, gamma + tau);
    add_identity_scaled   (cell_matrix, n, row0 + 4, row0 + 0, 4, -tau);
}

// Raw-pointer overload — used by the fused assemble+factor+solve path where
// the buffer is a thread-local slice of CpuLUCache::thread_lu.
void assemble_angle_group_block(double* a, int n, int row0,
                                const Cell2D& cell,
                                const Direction2D& dir,
                                int group) {
    const double volume   = cell.dx * cell.dy;
    const double gamma    = 0.25 * volume * cell.sigma_t[group];
    const double tau      = volume / (cell.velocity[group] * cell.dt);
    const double tau_half = 0.5 * tau;
    const double ax       = 0.5 * std::abs(dir.mu)  * cell.dy;
    const double ay       = 0.5 * std::abs(dir.eta) * cell.dx;

    auto kx = x_stream_block(ax, dir.mu  >= 0.0);
    auto ky = y_stream_block(ay, dir.eta >= 0.0);

    add_small_block       (a, n, row0 + 0, row0 + 0, kx);
    add_small_block       (a, n, row0 + 0, row0 + 0, ky);
    add_identity_scaled   (a, n, row0 + 0, row0 + 0, 4, gamma);
    add_identity_scaled   (a, n, row0 + 0, row0 + 4, 4, tau_half);

    add_small_block       (a, n, row0 + 4, row0 + 4, kx);
    add_small_block       (a, n, row0 + 4, row0 + 4, ky);
    add_identity_scaled   (a, n, row0 + 4, row0 + 4, 4, gamma + tau);
    add_identity_scaled   (a, n, row0 + 4, row0 + 0, 4, -tau);
}

// ---------------------------------------------------------------------------
// Isotropic multigroup scattering coupling
// ---------------------------------------------------------------------------
// Eq. (6): sigma_s is isotropic so scattering from group g_from, direction
// d_from contributes  -(V/4)*sigma_s[g_to,g_from]*w[d_from]  to the
// diagonal-in-DOF coupling at (g_to, d_to) <- (g_from, d_from).

void assemble_scatter_coupling(std::vector<double>& cell_matrix,
                               const SolverState2D& state,
                               int cell_index) {
    const Problem2D& problem = state.problem;
    const Cell2D& cell       = state.cells[cell_index];
    const int n              = problem.cell_block_size();
    const double volume      = cell.dx * cell.dy;

    for (int g_to = 0; g_to < problem.groups; ++g_to) {
        for (int d_to = 0; d_to < problem.num_dirs(); ++d_to) {
            const int row0 = local_angle_group_offset(problem, g_to, d_to, 0);
            for (int g_from = 0; g_from < problem.groups; ++g_from) {
                const double sigma_s = cell.sigma_s[g_to * problem.groups + g_from];
                if (sigma_s == 0.0) continue;
                for (int d_from = 0; d_from < problem.num_dirs(); ++d_from) {
                    // weight encodes quadrature; sum over d_from gives phi_code = <psi>
                    const double beta = -(volume / 4.0)
                                       * sigma_s
                                       * problem.directions[d_from].weight;
                    const int col0 = local_angle_group_offset(problem, g_from, d_from, 0);
                    for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof)
                        cm(cell_matrix, n, row0 + dof, col0 + dof) += beta;
                }
            }
        }
    }
}

// Raw-pointer overload for the fused assemble+factor+solve path.
void assemble_scatter_coupling(double* a, int n,
                               const SolverState2D& state,
                               int cell_index) {
    const Problem2D& problem = state.problem;
    const Cell2D& cell       = state.cells[cell_index];
    const double volume      = cell.dx * cell.dy;

    for (int g_to = 0; g_to < problem.groups; ++g_to) {
        for (int d_to = 0; d_to < problem.num_dirs(); ++d_to) {
            const int row0 = local_angle_group_offset(problem, g_to, d_to, 0);
            for (int g_from = 0; g_from < problem.groups; ++g_from) {
                const double sigma_s = cell.sigma_s[g_to * problem.groups + g_from];
                if (sigma_s == 0.0) continue;
                for (int d_from = 0; d_from < problem.num_dirs(); ++d_from) {
                    const double beta = -(volume / 4.0)
                                       * sigma_s
                                       * problem.directions[d_from].weight;
                    const int col0 = local_angle_group_offset(problem, g_from, d_from, 0);
                    for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof)
                        cm(a, n, row0 + dof, col0 + dof) += beta;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in LU factorisation (fallback when LAPACK not available)
// ---------------------------------------------------------------------------

bool lu_factor_in_place(std::vector<double>& a, int n, int* pivots) {
    for (int k = 0; k < n; ++k) {
        int    pivot_row = k;
        double pivot_abs = std::abs(cm(a, n, k, k));
        for (int row = k + 1; row < n; ++row) {
            const double cand = std::abs(cm(a, n, row, k));
            if (cand > pivot_abs) { pivot_abs = cand; pivot_row = row; }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) return false;
        pivots[k] = pivot_row;
        if (pivot_row != k)
            for (int col = 0; col < n; ++col)
                std::swap(cm(a, n, k, col), cm(a, n, pivot_row, col));
        const double akk = cm(a, n, k, k);
        for (int row = k + 1; row < n; ++row) {
            cm(a, n, row, k) /= akk;
            const double lik = cm(a, n, row, k);
            for (int col = k + 1; col < n; ++col)
                cm(a, n, row, col) -= lik * cm(a, n, k, col);
        }
    }
    return true;
}

void lu_solve_in_place(const std::vector<double>& lu, int n, const int* pivots, double* b) {
    for (int k = 0; k < n; ++k)
        if (pivots[k] != k) std::swap(b[k], b[pivots[k]]);
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            b[i] -= cm(lu, n, i, j) * b[j];
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j)
            b[i] -= cm(lu, n, i, j) * b[j];
        b[i] /= cm(lu, n, i, i);
    }
}

// ---------------------------------------------------------------------------
// Upwind inflow helpers (neighbor and boundary)
// ---------------------------------------------------------------------------
// Face DOF connectivity:
//   West/East faces store 4 values per segment:
//     [low_bottom, low_top, high_bottom, high_top]
//   South/North faces store 4 values per segment:
//     [low_left, low_right, high_left, high_right]
//
// Inflow from west neighbor: right-face corners of src → left-face corners of dst.
//   src right-face: DOFs 1,3,5,7 (col 1 of each temporal block)
//   dst left-face : DOFs 0,2,4,6 (col 0 of each temporal block)

void add_x_inflow_from_neighbor(std::vector<double>& rhs,
                                const std::vector<double>& iterate_flux,
                                const SolverState2D& state,
                                int dst_cell, int src_cell,
                                int group, int dir,
                                double coeff, bool from_west) {
    const Problem2D& p = state.problem;
    if (from_west) {
        // src right edge → dst left edge
        rhs[global_offset(p, dst_cell, group, dir, 0)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 1)];
        rhs[global_offset(p, dst_cell, group, dir, 2)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 3)];
        rhs[global_offset(p, dst_cell, group, dir, 4)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 5)];
        rhs[global_offset(p, dst_cell, group, dir, 6)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 7)];
    } else {
        // src left edge → dst right edge
        rhs[global_offset(p, dst_cell, group, dir, 1)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 0)];
        rhs[global_offset(p, dst_cell, group, dir, 3)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 2)];
        rhs[global_offset(p, dst_cell, group, dir, 5)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 4)];
        rhs[global_offset(p, dst_cell, group, dir, 7)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 6)];
    }
}

void add_y_inflow_from_neighbor(std::vector<double>& rhs,
                                const std::vector<double>& iterate_flux,
                                const SolverState2D& state,
                                int dst_cell, int src_cell,
                                int group, int dir,
                                double coeff, bool from_south) {
    const Problem2D& p = state.problem;
    if (from_south) {
        // src top edge → dst bottom edge
        rhs[global_offset(p, dst_cell, group, dir, 0)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 2)];
        rhs[global_offset(p, dst_cell, group, dir, 1)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 3)];
        rhs[global_offset(p, dst_cell, group, dir, 4)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 6)];
        rhs[global_offset(p, dst_cell, group, dir, 5)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 7)];
    } else {
        // src bottom edge → dst top edge
        rhs[global_offset(p, dst_cell, group, dir, 2)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 0)];
        rhs[global_offset(p, dst_cell, group, dir, 3)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 1)];
        rhs[global_offset(p, dst_cell, group, dir, 6)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 4)];
        rhs[global_offset(p, dst_cell, group, dir, 7)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 5)];
    }
}

void add_x_inflow_from_boundary(std::vector<double>& rhs,
                                const SolverState2D& state,
                                int cell, int boundary_j,
                                int group, int dir,
                                double coeff, bool from_west) {
    const Problem2D& p = state.problem;
    const std::vector<double>& face = from_west ? p.boundary.west : p.boundary.east;
    if (face.empty()) return;

    const int off = face_offset_west_east(p, boundary_j, group, dir, 0);
    // Boundary data layout: [low_bottom, low_top, high_bottom, high_top]
    // Mapping to DOFs depends on which face we are loading:
    if (from_west) {
        rhs[global_offset(p, cell, group, dir, 0)] += coeff * face[off + 0];
        rhs[global_offset(p, cell, group, dir, 2)] += coeff * face[off + 1];
        rhs[global_offset(p, cell, group, dir, 4)] += coeff * face[off + 2];
        rhs[global_offset(p, cell, group, dir, 6)] += coeff * face[off + 3];
    } else {
        rhs[global_offset(p, cell, group, dir, 1)] += coeff * face[off + 0];
        rhs[global_offset(p, cell, group, dir, 3)] += coeff * face[off + 1];
        rhs[global_offset(p, cell, group, dir, 5)] += coeff * face[off + 2];
        rhs[global_offset(p, cell, group, dir, 7)] += coeff * face[off + 3];
    }
}

void add_y_inflow_from_boundary(std::vector<double>& rhs,
                                const SolverState2D& state,
                                int cell, int boundary_i,
                                int group, int dir,
                                double coeff, bool from_south) {
    const Problem2D& p = state.problem;
    const std::vector<double>& face = from_south ? p.boundary.south : p.boundary.north;
    if (face.empty()) return;

    const int off = face_offset_south_north(p, boundary_i, group, dir, 0);
    if (from_south) {
        rhs[global_offset(p, cell, group, dir, 0)] += coeff * face[off + 0];
        rhs[global_offset(p, cell, group, dir, 1)] += coeff * face[off + 1];
        rhs[global_offset(p, cell, group, dir, 4)] += coeff * face[off + 2];
        rhs[global_offset(p, cell, group, dir, 5)] += coeff * face[off + 3];
    } else {
        rhs[global_offset(p, cell, group, dir, 2)] += coeff * face[off + 0];
        rhs[global_offset(p, cell, group, dir, 3)] += coeff * face[off + 1];
        rhs[global_offset(p, cell, group, dir, 6)] += coeff * face[off + 2];
        rhs[global_offset(p, cell, group, dir, 7)] += coeff * face[off + 3];
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

double cell_average_angular_flux(const SolverState2D& state,
                                 const std::vector<double>& flux,
                                 int cell, int group, int dir) {
    // Average only the LOW temporal moment DOFs (corners 0-3, representing
    // the t_{n+1} angular flux). DOFs 4-7 are the HIGH temporal moment
    // (t_n, the previous-timestep flux) and must NOT be included — mixing
    // them with the new-time DOFs produces a diluted, incorrect cell average
    // that masks physical changes such as the Marshak wave front heating.
    double sum = 0.0;
    for (int corner = 0; corner < kSpatialCorners2D; ++corner)
        sum += flux[global_offset(state.problem, cell, group, dir, corner)];
    return sum / static_cast<double>(kSpatialCorners2D);
}

double cell_centered_scalar_flux(const SolverState2D& state,
                                 const std::vector<double>& flux,
                                 int cell, int group) {
    const Problem2D& p = state.problem;
    double value      = 0.0;
    double weight_sum = 0.0;
    for (int dir = 0; dir < p.num_dirs(); ++dir) {
        const double w  = p.directions[dir].weight;
        value      += w * cell_average_angular_flux(state, flux, cell, group, dir);
        weight_sum += w;
    }
    return (weight_sum != 0.0) ? (value / weight_sum) : 0.0;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

void validate_problem(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    require(p.nx > 0 && p.ny > 0, "Problem2D requires nx > 0 and ny > 0.");
    require(p.groups > 0, "Problem2D requires groups > 0.");
    require(!p.directions.empty(), "Problem2D requires at least one direction.");
    require(static_cast<int>(state.cells.size()) == p.num_cells(),
            "cells.size() must equal nx * ny.");

    for (int c = 0; c < p.num_cells(); ++c) {
        const Cell2D& cell = state.cells[c];
        require(static_cast<int>(cell.velocity.size()) == p.groups,
                "Cell velocity size must equal number of groups.");
        require(static_cast<int>(cell.sigma_t.size()) == p.groups,
                "Cell sigma_t size must equal number of groups.");
        require(static_cast<int>(cell.sigma_s.size()) == p.groups * p.groups,
                "Cell sigma_s size must equal groups*groups.");
        require(static_cast<int>(cell.source.size()) == p.cell_block_size(),
                "Cell source size must equal cell_block_size().");
        require(cell.dx > 0.0 && cell.dy > 0.0 && cell.dt > 0.0,
                "Cell dimensions and dt must be positive.");
        for (double v : cell.velocity)
            require(v > 0.0, "All group velocities must be positive.");
    }

    const int west_east_size   = p.ny * p.groups * p.num_dirs() * 4;
    const int south_north_size = p.nx * p.groups * p.num_dirs() * 4;
    require(p.boundary.west.empty()  || static_cast<int>(p.boundary.west.size())  == west_east_size,
            "west boundary size must be ny * groups * num_dirs * 4.");
    require(p.boundary.east.empty()  || static_cast<int>(p.boundary.east.size())  == west_east_size,
            "east boundary size must be ny * groups * num_dirs * 4.");
    require(p.boundary.south.empty() || static_cast<int>(p.boundary.south.size()) == south_north_size,
            "south boundary size must be nx * groups * num_dirs * 4.");
    require(p.boundary.north.empty() || static_cast<int>(p.boundary.north.size()) == south_north_size,
            "north boundary size must be nx * groups * num_dirs * 4.");
}

// ---------------------------------------------------------------------------
// State initialisation
// ---------------------------------------------------------------------------

void initialize_state(SolverState2D& state,
                      const std::vector<double>& initial_condition) {
    validate_problem(state);
    require(static_cast<int>(initial_condition.size()) == state.problem.total_unknowns(),
            "Initial condition must have total_unknowns() entries.");

    state.cell_matrices.assign(state.problem.num_cells()
                               * state.problem.cell_block_elems(), 0.0);
    state.rhs_const.assign(state.problem.total_unknowns(), 0.0);
    state.flux_previous = initial_condition;
    state.flux_last     = initial_condition;
    state.flux_current.assign(state.problem.total_unknowns(), 0.0);
}

// ---------------------------------------------------------------------------
// Matrix assembly
// ---------------------------------------------------------------------------

void assemble_cell_matrices(SolverState2D& state) {
    validate_problem(state);
    const Problem2D& p = state.problem;
    state.cell_matrices.assign(p.num_cells() * p.cell_block_elems(), 0.0);

    // Write each cell's matrix directly into its slice of cell_matrices using
    // the raw-pointer assembly path — no per-cell temporary vector allocation.
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        double*   a = state.cell_matrices.data() + cell * p.cell_block_elems();
        const int n = p.cell_block_size();
        for (int g = 0; g < p.groups; ++g) {
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int row0 = local_angle_group_offset(p, g, d, 0);
                assemble_angle_group_block(a, n, row0,
                                           state.cells[cell], p.directions[d], g);
            }
        }
        assemble_scatter_coupling(a, n, state, cell);
    }
}

// ---------------------------------------------------------------------------
// RHS assembly
// ---------------------------------------------------------------------------

void build_constant_rhs(SolverState2D& state) {
    const Problem2D& p = state.problem;
    state.rhs_const.assign(p.total_unknowns(), 0.0);

    // Constant source + time-derivative term from previous time level.
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const Cell2D& c = state.cells[cell];
        const double volume = c.dx * c.dy;
        for (int g = 0; g < p.groups; ++g) {
            const double tau_half = 0.5 * volume / (c.velocity[g] * c.dt);
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int base = global_offset(p, cell, g, d, 0);
                for (int corner = 0; corner < 4; ++corner) {
                    // Low temporal moment: source + tau/2 * psi_n
                    state.rhs_const[base + corner] =
                        (volume / 4.0) * c.source[local_angle_group_offset(p, g, d, corner)]
                        + tau_half * state.flux_previous[base + corner];
                    // High temporal moment: source only (psi_n enters via the -tau coupling)
                    state.rhs_const[base + 4 + corner] =
                        (volume / 4.0) * c.source[local_angle_group_offset(p, g, d, 4 + corner)];
                }
            }
        }
    }
}

void add_upwind_inflow_rhs(std::vector<double>& rhs,
                           const std::vector<double>& iterate_flux,
                           const SolverState2D& state) {
    const Problem2D& p = state.problem;
    require(static_cast<int>(rhs.size()) == p.total_unknowns(),
            "rhs must have total_unknowns() entries.");
    require(static_cast<int>(iterate_flux.size()) == p.total_unknowns(),
            "iterate_flux must have total_unknowns() entries.");

    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            const Cell2D& c = state.cells[cell];
            for (int g = 0; g < p.groups; ++g) {
                for (int d = 0; d < p.num_dirs(); ++d) {
                    const Direction2D& omega = p.directions[d];
                    const double ax = 0.5 * std::abs(omega.mu)  * c.dy;
                    const double ay = 0.5 * std::abs(omega.eta) * c.dx;

                    if (omega.mu >= 0.0) {
                        if (i > 0)
                            add_x_inflow_from_neighbor(rhs, iterate_flux, state,
                                cell, cell_id(i - 1, j, p.nx), g, d, ax, true);
                        else
                            add_x_inflow_from_boundary(rhs, state, cell, j, g, d, ax, true);
                    } else {
                        if (i + 1 < p.nx)
                            add_x_inflow_from_neighbor(rhs, iterate_flux, state,
                                cell, cell_id(i + 1, j, p.nx), g, d, ax, false);
                        else
                            add_x_inflow_from_boundary(rhs, state, cell, j, g, d, ax, false);
                    }

                    if (omega.eta >= 0.0) {
                        if (j > 0)
                            add_y_inflow_from_neighbor(rhs, iterate_flux, state,
                                cell, cell_id(i, j - 1, p.nx), g, d, ay, true);
                        else
                            add_y_inflow_from_boundary(rhs, state, cell, i, g, d, ay, true);
                    } else {
                        if (j + 1 < p.ny)
                            add_y_inflow_from_neighbor(rhs, iterate_flux, state,
                                cell, cell_id(i, j + 1, p.nx), g, d, ay, false);
                        else
                            add_y_inflow_from_boundary(rhs, state, cell, i, g, d, ay, false);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L2 error norm
// ---------------------------------------------------------------------------

double relative_l2_error(const std::vector<double>& previous,
                         const std::vector<double>& current) {
    require(previous.size() == current.size(),
            "relative_l2_error requires equal-size vectors.");
    double numer = 0.0, denom = 0.0;
    for (std::size_t i = 0; i < previous.size(); ++i) {
        const double diff = current[i] - previous[i];
        numer += diff * diff;
        denom += previous[i] * previous[i];
    }
    return (denom > 0.0) ? std::sqrt(numer / denom) : std::sqrt(numer);
}

// ---------------------------------------------------------------------------
// CPU LU solver
// ---------------------------------------------------------------------------

void factor_cells_cpu(const SolverState2D& state, CpuLUCache& cache, bool use_openmp) {
    // Allocate (or verify) per-thread scratch buffers.
    // This is the only role of factor_cells_cpu in the CPU path — the actual
    // LU factorisation is now fused into solve_cells_cpu, one cell at a time,
    // reusing these thread-local buffers without additional heap allocation.
    const Problem2D& p          = state.problem;
    const int        block_elems = p.cell_block_elems();
    const int        block_n     = p.cell_block_size();

    int nthreads = 1;
#ifdef _OPENMP
    if (use_openmp) nthreads = omp_get_max_threads();
#endif

    // Skip reallocation when the existing buffers are already the right size.
    if (cache.num_threads_cached == nthreads &&
        static_cast<int>(cache.thread_lu.size()) == nthreads * block_elems) {
        cache.valid = true;
        return;
    }

    cache.thread_lu    .assign(nthreads * block_elems, 0.0);
    cache.thread_pivots.assign(nthreads * block_n,     0);
    cache.num_threads_cached = nthreads;
    cache.valid = true;
}

void solve_cells_cpu(const SolverState2D& state, CpuLUCache& cache,
                     std::vector<double>& rhs, bool use_openmp) {
    // Fused per-cell loop: assemble A → LU-factor in place → solve A x = b.
    //
    // Memory footprint: O(nthreads × block²) instead of O(ncells × block²).
    // For a 30×30 grid with 16 groups and S4 (cell_block_size=1024):
    //   old: 900 × 1024² × 8 B = 7.5 GB
    //   new:   8 × 1024² × 8 B =  67 MB  (for 8 threads)
    //
    // Trade-off: cell matrices are re-assembled and re-factored every source
    // iteration.  For TRT (reuse_factorization=false) this was already the
    // case.  For pure transport (reuse_factorization=true) the extra work is
    // assembly+factor per iteration, but at far lower memory cost.
    const Problem2D& p     = state.problem;
    const int        n     = p.cell_block_size();
    const int        lda   = n;
    const int        nrhs  = 1;
    const int        ldb   = n;
    const char       trans = 'N';

    require(cache.valid,
            "CPU cache not initialised — call factor_cells_cpu first.");
    require(static_cast<int>(rhs.size()) == p.total_unknowns(),
            "rhs must have total_unknowns() entries.");

#ifndef _OPENMP
    (void)use_openmp;
#endif

#ifdef _OPENMP
    #pragma omp parallel for if(use_openmp)
#endif
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        int tid = 0;
#ifdef _OPENMP
        if (use_openmp) tid = omp_get_thread_num();
#endif
        // Thread-local scratch (no allocation — pre-sized by factor_cells_cpu).
        double* a    = cache.thread_lu    .data() + tid * p.cell_block_elems();
        int*    ipiv = cache.thread_pivots.data() + tid * n;

        // 1. Assemble the cell matrix.
        std::fill(a, a + p.cell_block_elems(), 0.0);
        for (int g = 0; g < p.groups; ++g) {
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int row0 = local_angle_group_offset(p, g, d, 0);
                assemble_angle_group_block(a, n, row0,
                                           state.cells[cell], p.directions[d], g);
            }
        }
        assemble_scatter_coupling(a, n, state, cell);

        // 2. LU-factor A in place.
        int info = 0;
        dgetrf_(&n, &n, a, &lda, ipiv, &info);
        if (info != 0)
            throw std::runtime_error(
                "dgetrf failed cell " + std::to_string(cell)
                + " (info=" + std::to_string(info) + ")");

        // 3. Solve A x = b; b is overwritten with the solution.
        double* b = rhs.data() + cell * n;
        dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
        if (info != 0)
            throw std::runtime_error(
                "dgetrs failed cell " + std::to_string(cell)
                + " (info=" + std::to_string(info) + ")");
    }
}

// ---------------------------------------------------------------------------
// One time step
// ---------------------------------------------------------------------------

IterationStats run_one_timestep_cpu(SolverState2D& state,
                                    CpuLUCache& cache,
                                    bool use_openmp) {
    const Problem2D& p = state.problem;
    // factor_cells_cpu now only allocates the per-thread scratch buffers.
    // Cell matrices are assembled and LU-factored inside every solve_cells_cpu
    // call, so reuse_factorization no longer affects the CPU path.
    if (!cache.valid)
        factor_cells_cpu(state, cache, use_openmp);

    state.flux_last = p.initialize_from_previous
                    ? state.flux_previous
                    : std::vector<double>(p.total_unknowns(), 0.0);

    IterationStats stats{};

    for (int it = 0; it < p.max_iters; ++it) {
        state.flux_current = state.rhs_const;
        add_upwind_inflow_rhs(state.flux_current, state.flux_last, state);
        solve_cells_cpu(state, cache, state.flux_current, use_openmp);

        const double error = relative_l2_error(state.flux_last, state.flux_current);
        // Guard spectral radius: undefined on the first iteration when
        // error_previous == 0.
        stats.spectral_radius  = (stats.error_previous > 0.0)
                                ? (error / stats.error_previous)
                                : 0.0;
        stats.final_error      = error;
        stats.iterations       = it + 1;
        state.flux_last.swap(state.flux_current);

        if (error < p.convergence_tol) break;

        stats.iterate();  // advances error_previous
    }

    state.flux_previous = state.flux_last;
    return stats;
}

// ---------------------------------------------------------------------------
// Full time loop
// ---------------------------------------------------------------------------

std::vector<TimestepRecord2D> run_time_cpu(SolverState2D& state,
                                           CpuLUCache& cpu_cache,
                                           bool use_openmp,
                                           const TransportOutputFiles2D& outputs) {
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
        build_constant_rhs(state);
        IterationStats stats = run_one_timestep_cpu(state, cpu_cache, use_openmp);
        time += dt;
        history.push_back(TimestepRecord2D{step, time, stats});

        std::vector<CellScalarField2D> fields;
        if (outputs.save_flux) {
            append_fields(fields,
                make_angular_flux_group_dir_fields(state, state.flux_previous, "angular_flux"));
            append_fields(fields,
                make_scalar_flux_group_fields(state, state.flux_previous, "scalar_flux_g"));
        }
        writer.write_step(step, time, fields);

        std::cout << "step " << step
                  << "  time=" << time
                  << "  iters=" << stats.iterations
                  << "  rho=" << stats.spectral_radius
                  << "  err=" << stats.final_error << '\n';
    }

    const std::string backend_name = use_openmp ? "omp_lapack" : "cpu_lapack";
    write_transport_summary_json(outputs.summary_json, state, history,
                                 backend_name, writer.pvd_path());

    std::cout << "Wrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << outputs.summary_json << '\n';
    return history;
}

// ---------------------------------------------------------------------------
// Quadrature sets
// ---------------------------------------------------------------------------

std::vector<Direction2D> make_tensor_product_quadrature_2d(
    const std::vector<double>& mu, const std::vector<double>& w) {
    require(mu.size() == w.size(), "mu and w must have the same size.");
    std::vector<Direction2D> dirs;
    dirs.reserve(mu.size() * mu.size());
    for (std::size_t j = 0; j < mu.size(); ++j)
        for (std::size_t i = 0; i < mu.size(); ++i)
            dirs.push_back(Direction2D{mu[i], mu[j], w[i] * w[j]});

    double sum_w = 0.0;
    for (const auto& d : dirs) sum_w += d.weight;
    if (sum_w != 0.0)
        for (auto& d : dirs) d.weight /= sum_w;
    return dirs;
}

std::vector<Direction2D> make_level_symmetric_quadrature_2d(int sn_order) {
    using OctantEntry = std::array<double, 3>;
    std::vector<OctantEntry> first_octant;

    switch (sn_order) {
        case 2:
            first_octant = {{ {0.5773502691896257, 0.5773502691896257, 0.25} }};
            break;
        case 4:
            first_octant = {{
                {0.3500211745815407, 0.8688903007222012, 0.125},
                {0.8688903007222012, 0.3500211745815407, 0.125}
            }};
            break;
        case 6:
            first_octant = {{
                {0.2666354015167047, 0.9261809355174897, 0.0738967877396407},
                {0.6815077265365469, 0.6815077265365469, 0.0522103172504727},
                {0.9261809355174897, 0.2666354015167047, 0.0738967877396407}
            }};
            break;
        case 8:
            first_octant = {{
                {0.2182178902359924, 0.9511897312113418, 0.0604938271604938},
                {0.5773502691896257, 0.7867957924694432, 0.0907407407407407},
                {0.7867957924694432, 0.5773502691896257, 0.0907407407407407},
                {0.9511897312113418, 0.2182178902359924, 0.0604938271604938}
            }};
            break;
        case 10:
            first_octant = {{
                {0.1893213264780105, 0.9624302435022339, 0.0489872391580385},
                {0.5088817555826188, 0.8606632976105324, 0.0413295978698440},
                {0.6943188875943843, 0.6943188875943843, 0.0657672860700875},
                {0.8606632976105324, 0.5088817555826188, 0.0413295978698440},
                {0.9624302435022339, 0.1893213264780105, 0.0489872391580385}
            }};
            break;
        case 12:
            first_octant = {{
                {0.1672126528227133, 0.9716377192513584, 0.0404939918855891},
                {0.4595476346425947, 0.8887176968558780, 0.0381989900133081},
                {0.6280190966421309, 0.7781849323219839, 0.0399233473099967},
                {0.7781849323219839, 0.6280190966421309, 0.0399233473099967},
                {0.8887176968558780, 0.4595476346425947, 0.0381989900133081},
                {0.9716377192513584, 0.1672126528227133, 0.0404939918855891}
            }};
            break;
        default:
            throw std::runtime_error("Unsupported level-symmetric order S"
                                     + std::to_string(sn_order) + ".");
    }

    std::vector<Direction2D> dirs;
    dirs.reserve(first_octant.size() * 4);
    for (const auto& entry : first_octant) {
        for (double sx : {-1.0, 1.0})
            for (double sy : {-1.0, 1.0})
                dirs.push_back(Direction2D{sx * entry[0], sy * entry[1], entry[2]});
    }

    double sum_w = 0.0;
    for (const auto& d : dirs) sum_w += d.weight;
    if (sum_w != 0.0)
        for (auto& d : dirs) d.weight /= sum_w;
    return dirs;
}

} // namespace therefore2d
