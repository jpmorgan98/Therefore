#include "transport2d.hpp"

#include <algorithm>
#include <array>
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

// Spatial corner ordering used everywhere:
// 0 = LL, 1 = LR, 2 = UL, 3 = UR.
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

std::array<double, 16> x_stream_block(double ax, bool positive_x) {
    // Eq. (4) in README_2D.md.
    if (positive_x) {
        return {
            +ax, +ax, 0.0, 0.0,
            -ax, +ax, 0.0, 0.0,
            0.0, 0.0, +ax, +ax,
            0.0, 0.0, -ax, +ax
        };
    }
    return {
        -ax, -ax, 0.0, 0.0,
        +ax, -ax, 0.0, 0.0,
        0.0, 0.0, -ax, -ax,
        0.0, 0.0, +ax, -ax
    };
}

std::array<double, 16> y_stream_block(double ay, bool positive_y) {
    // Eq. (4) in README_2D.md.
    if (positive_y) {
        return {
            +ay, 0.0, +ay, 0.0,
            0.0, +ay, 0.0, +ay,
            -ay, 0.0, +ay, 0.0,
            0.0, -ay, 0.0, +ay
        };
    }
    return {
        -ay, 0.0, -ay, 0.0,
        0.0, -ay, 0.0, -ay,
        +ay, 0.0, -ay, 0.0,
        0.0, +ay, 0.0, -ay
    };
}

void add_small_block(std::vector<double>& a, int n, int row0, int col0, const std::array<double, 16>& block) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            cm(a, n, row0 + r, col0 + c) += block[r * 4 + c];
        }
    }
}

void add_identity_scaled(std::vector<double>& a, int n, int row0, int col0, int count, double value) {
    for (int k = 0; k < count; ++k) {
        cm(a, n, row0 + k, col0 + k) += value;
    }
}

void assemble_angle_group_block(
    std::vector<double>& cell_matrix,
    int n,
    int row0,
    const Cell2D& cell,
    const Direction2D& dir,
    int group) {

    const double volume = cell.dx * cell.dy;
    const double gamma = 0.25 * volume * cell.sigma_t[group];
    const double tau = volume / (cell.velocity[group] * cell.dt);
    const double tau_half = 0.5 * tau;
    const double ax = 0.5 * std::abs(dir.mu) * cell.dy;
    const double ay = 0.5 * std::abs(dir.eta) * cell.dx;

    auto kx = x_stream_block(ax, dir.mu >= 0.0);
    auto ky = y_stream_block(ay, dir.eta >= 0.0);

    // Eq. (5): A_ag = [[S, tau/2 I], [-tau I, S + tau I]].
    add_small_block(cell_matrix, n, row0 + 0, row0 + 0, kx);
    add_small_block(cell_matrix, n, row0 + 0, row0 + 0, ky);
    add_small_block(cell_matrix, n, row0 + 4, row0 + 4, kx);
    add_small_block(cell_matrix, n, row0 + 4, row0 + 4, ky);

    add_identity_scaled(cell_matrix, n, row0 + 0, row0 + 0, 4, gamma);
    add_identity_scaled(cell_matrix, n, row0 + 4, row0 + 4, 4, gamma + tau);
    add_identity_scaled(cell_matrix, n, row0 + 0, row0 + 4, 4, tau_half);
    add_identity_scaled(cell_matrix, n, row0 + 4, row0 + 0, 4, -tau);
}

void assemble_scatter_coupling(std::vector<double>& cell_matrix, const SolverState2D& state, int cell_index) {
    const Problem2D& problem = state.problem;
    const Cell2D& cell = state.cells[cell_index];
    const int n = problem.cell_block_size();
    const double volume = cell.dx * cell.dy;

    // Eq. (6): isotropic multigroup scattering creates dense coupling across
    // all direction/group blocks but only matches the same local temporal-spatial dof.
    for (int g_to = 0; g_to < problem.groups; ++g_to) {
        for (int d_to = 0; d_to < problem.num_dirs(); ++d_to) {
            const int row0 = local_angle_group_offset(problem, g_to, d_to, 0);
            for (int g_from = 0; g_from < problem.groups; ++g_from) {
                const double sigma_s = cell.sigma_s[g_to * problem.groups + g_from];
                if (sigma_s == 0.0) {
                    continue;
                }
                for (int d_from = 0; d_from < problem.num_dirs(); ++d_from) {
                    const double beta = -(volume / 8.0) * sigma_s * problem.directions[d_from].weight;
                    const int col0 = local_angle_group_offset(problem, g_from, d_from, 0);
                    for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                        cm(cell_matrix, n, row0 + dof, col0 + dof) += beta;
                    }
                }
            }
        }
    }
}

bool lu_factor_in_place(std::vector<double>& a, int n, int* pivots) {
    for (int k = 0; k < n; ++k) {
        int pivot_row = k;
        double pivot_abs = std::abs(cm(a, n, k, k));
        for (int row = k + 1; row < n; ++row) {
            const double cand = std::abs(cm(a, n, row, k));
            if (cand > pivot_abs) {
                pivot_abs = cand;
                pivot_row = row;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        pivots[k] = pivot_row;
        if (pivot_row != k) {
            for (int col = 0; col < n; ++col) {
                std::swap(cm(a, n, k, col), cm(a, n, pivot_row, col));
            }
        }
        const double akk = cm(a, n, k, k);
        for (int row = k + 1; row < n; ++row) {
            cm(a, n, row, k) /= akk;
            const double lik = cm(a, n, row, k);
            for (int col = k + 1; col < n; ++col) {
                cm(a, n, row, col) -= lik * cm(a, n, k, col);
            }
        }
    }
    return true;
}

void lu_solve_in_place(const std::vector<double>& lu, int n, const int* pivots, double* b) {
    for (int k = 0; k < n; ++k) {
        if (pivots[k] != k) {
            std::swap(b[k], b[pivots[k]]);
        }
    }
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            b[i] -= cm(lu, n, i, j) * b[j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= cm(lu, n, i, j) * b[j];
        }
        b[i] /= cm(lu, n, i, i);
    }
}

void add_x_inflow_from_neighbor(
    std::vector<double>& rhs,
    const std::vector<double>& iterate_flux,
    const SolverState2D& state,
    int dst_cell,
    int src_cell,
    int group,
    int dir,
    double coeff,
    bool from_west) {

    const Problem2D& p = state.problem;
    if (from_west) {
        rhs[global_offset(p, dst_cell, group, dir, 0)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 1)];
        rhs[global_offset(p, dst_cell, group, dir, 2)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 3)];
        rhs[global_offset(p, dst_cell, group, dir, 4)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 5)];
        rhs[global_offset(p, dst_cell, group, dir, 6)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 7)];
    } else {
        rhs[global_offset(p, dst_cell, group, dir, 1)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 0)];
        rhs[global_offset(p, dst_cell, group, dir, 3)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 2)];
        rhs[global_offset(p, dst_cell, group, dir, 5)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 4)];
        rhs[global_offset(p, dst_cell, group, dir, 7)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 6)];
    }
}

void add_y_inflow_from_neighbor(
    std::vector<double>& rhs,
    const std::vector<double>& iterate_flux,
    const SolverState2D& state,
    int dst_cell,
    int src_cell,
    int group,
    int dir,
    double coeff,
    bool from_south) {

    const Problem2D& p = state.problem;
    if (from_south) {
        rhs[global_offset(p, dst_cell, group, dir, 0)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 2)];
        rhs[global_offset(p, dst_cell, group, dir, 1)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 3)];
        rhs[global_offset(p, dst_cell, group, dir, 4)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 6)];
        rhs[global_offset(p, dst_cell, group, dir, 5)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 7)];
    } else {
        rhs[global_offset(p, dst_cell, group, dir, 2)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 0)];
        rhs[global_offset(p, dst_cell, group, dir, 3)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 1)];
        rhs[global_offset(p, dst_cell, group, dir, 6)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 4)];
        rhs[global_offset(p, dst_cell, group, dir, 7)] += coeff * iterate_flux[global_offset(p, src_cell, group, dir, 5)];
    }
}

void add_x_inflow_from_boundary(
    std::vector<double>& rhs,
    const SolverState2D& state,
    int cell,
    int boundary_j,
    int group,
    int dir,
    double coeff,
    bool from_west) {

    const Problem2D& p = state.problem;
    const std::vector<double>& face = from_west ? p.boundary.west : p.boundary.east;
    if (face.empty()) {
        return;
    }
    const int off = face_offset_west_east(p, boundary_j, group, dir, 0);
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

void add_y_inflow_from_boundary(
    std::vector<double>& rhs,
    const SolverState2D& state,
    int cell,
    int boundary_i,
    int group,
    int dir,
    double coeff,
    bool from_south) {

    const Problem2D& p = state.problem;
    const std::vector<double>& face = from_south ? p.boundary.south : p.boundary.north;
    if (face.empty()) {
        return;
    }
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

void validate_problem(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    require(p.nx > 0 && p.ny > 0, "Problem2D requires nx > 0 and ny > 0.");
    require(p.groups > 0, "Problem2D requires groups > 0.");
    require(!p.directions.empty(), "Problem2D requires at least one direction.");
    require(static_cast<int>(state.cells.size()) == p.num_cells(), "cells.size() must equal nx * ny.");

    for (int c = 0; c < p.num_cells(); ++c) {
        const Cell2D& cell = state.cells[c];
        require(static_cast<int>(cell.velocity.size()) == p.groups, "Cell velocity size must equal number of groups.");
        require(static_cast<int>(cell.sigma_t.size()) == p.groups, "Cell sigma_t size must equal number of groups.");
        require(static_cast<int>(cell.sigma_s.size()) == p.groups * p.groups, "Cell sigma_s size must equal groups*groups.");
        require(static_cast<int>(cell.source.size()) == p.cell_block_size(), "Cell source size must equal cell_block_size().");
        require(cell.dx > 0.0 && cell.dy > 0.0 && cell.dt > 0.0, "Cell dimensions and dt must be positive.");
        for (double v : cell.velocity) {
            require(v > 0.0, "All group velocities must be positive.");
        }
    }

    const int west_east_size = p.ny * p.groups * p.num_dirs() * 4;
    const int south_north_size = p.nx * p.groups * p.num_dirs() * 4;
    require(p.boundary.west.empty() || static_cast<int>(p.boundary.west.size()) == west_east_size,
            "west boundary size must be ny * groups * num_dirs * 4.");
    require(p.boundary.east.empty() || static_cast<int>(p.boundary.east.size()) == west_east_size,
            "east boundary size must be ny * groups * num_dirs * 4.");
    require(p.boundary.south.empty() || static_cast<int>(p.boundary.south.size()) == south_north_size,
            "south boundary size must be nx * groups * num_dirs * 4.");
    require(p.boundary.north.empty() || static_cast<int>(p.boundary.north.size()) == south_north_size,
            "north boundary size must be nx * groups * num_dirs * 4.");
}

void initialize_state(SolverState2D& state, const std::vector<double>& initial_condition) {
    validate_problem(state);
    require(static_cast<int>(initial_condition.size()) == state.problem.total_unknowns(),
            "Initial condition must have total_unknowns() entries.");

    state.cell_matrices.assign(state.problem.num_cells() * state.problem.cell_block_elems(), 0.0);
    state.rhs_const.assign(state.problem.total_unknowns(), 0.0);
    state.flux_previous = initial_condition;
    state.flux_last = initial_condition;
    state.flux_current.assign(state.problem.total_unknowns(), 0.0);
}

void assemble_cell_matrices(SolverState2D& state) {
    validate_problem(state);
    const Problem2D& p = state.problem;
    state.cell_matrices.assign(p.num_cells() * p.cell_block_elems(), 0.0);

    for (int cell = 0; cell < p.num_cells(); ++cell) {
        std::vector<double> local(p.cell_block_elems(), 0.0);
        for (int g = 0; g < p.groups; ++g) {
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int row0 = local_angle_group_offset(p, g, d, 0);
                assemble_angle_group_block(local, p.cell_block_size(), row0, state.cells[cell], p.directions[d], g);
            }
        }
        assemble_scatter_coupling(local, state, cell);
        std::copy(local.begin(), local.end(), state.cell_matrices.begin() + cell * p.cell_block_elems());
    }
}

void build_constant_rhs(SolverState2D& state) {
    const Problem2D& p = state.problem;
    state.rhs_const.assign(p.total_unknowns(), 0.0);

    // Eq. (7): constant source + previous-time contribution.
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const Cell2D& c = state.cells[cell];
        const double volume = c.dx * c.dy;
        for (int g = 0; g < p.groups; ++g) {
            const double tau_half = 0.5 * volume / (c.velocity[g] * c.dt);
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int base = global_offset(p, cell, g, d, 0);
                for (int corner = 0; corner < 4; ++corner) {
                    state.rhs_const[base + corner] = (volume / 8.0) * c.source[local_angle_group_offset(p, g, d, corner)]
                                                   + tau_half * state.flux_previous[base + corner];
                    state.rhs_const[base + 4 + corner] = (volume / 8.0) * c.source[local_angle_group_offset(p, g, d, 4 + corner)];
                }
            }
        }
    }
}

void add_upwind_inflow_rhs(std::vector<double>& rhs, const std::vector<double>& iterate_flux, const SolverState2D& state) {
    const Problem2D& p = state.problem;
    require(static_cast<int>(rhs.size()) == p.total_unknowns(), "rhs must have total_unknowns() entries.");
    require(static_cast<int>(iterate_flux.size()) == p.total_unknowns(), "iterate_flux must have total_unknowns() entries.");

    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            const Cell2D& c = state.cells[cell];
            for (int g = 0; g < p.groups; ++g) {
                for (int d = 0; d < p.num_dirs(); ++d) {
                    const Direction2D& omega = p.directions[d];
                    const double ax = 0.5 * std::abs(omega.mu) * c.dy;
                    const double ay = 0.5 * std::abs(omega.eta) * c.dx;

                    if (omega.mu >= 0.0) {
                        if (i > 0) {
                            add_x_inflow_from_neighbor(rhs, iterate_flux, state, cell, cell_id(i - 1, j, p.nx), g, d, ax, true);
                        } else {
                            add_x_inflow_from_boundary(rhs, state, cell, j, g, d, ax, true);
                        }
                    } else {
                        if (i + 1 < p.nx) {
                            add_x_inflow_from_neighbor(rhs, iterate_flux, state, cell, cell_id(i + 1, j, p.nx), g, d, ax, false);
                        } else {
                            add_x_inflow_from_boundary(rhs, state, cell, j, g, d, ax, false);
                        }
                    }

                    if (omega.eta >= 0.0) {
                        if (j > 0) {
                            add_y_inflow_from_neighbor(rhs, iterate_flux, state, cell, cell_id(i, j - 1, p.nx), g, d, ay, true);
                        } else {
                            add_y_inflow_from_boundary(rhs, state, cell, i, g, d, ay, true);
                        }
                    } else {
                        if (j + 1 < p.ny) {
                            add_y_inflow_from_neighbor(rhs, iterate_flux, state, cell, cell_id(i, j + 1, p.nx), g, d, ay, false);
                        } else {
                            add_y_inflow_from_boundary(rhs, state, cell, i, g, d, ay, false);
                        }
                    }
                }
            }
        }
    }
}

double relative_l2_error(const std::vector<double>& previous, const std::vector<double>& current) {
    require(previous.size() == current.size(), "relative_l2_error requires equal-size vectors.");
    double numer = 0.0;
    double denom = 0.0;
    for (std::size_t i = 0; i < previous.size(); ++i) {
        const double diff = current[i] - previous[i];
        numer += diff * diff;
        denom += previous[i] * previous[i];
    }
    if (denom == 0.0) {
        return std::sqrt(numer);
    }
    return std::sqrt(numer / denom);
}

void factor_cells_cpu(const SolverState2D& state, CpuLUCache& cache) {
    const Problem2D& p = state.problem;
    cache.lu = state.cell_matrices;
    cache.pivots.assign(p.num_cells() * p.cell_block_size(), 0);

    const int n = p.cell_block_size();
    const int lda = n;
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        int info = 0;
        double* a = cache.lu.data() + cell * p.cell_block_elems();
        int* ipiv = cache.pivots.data() + cell * n;
        dgetrf_(&n, &n, a, &lda, ipiv, &info);
        if (info != 0) {
            throw std::runtime_error("CPU LAPACK dgetrf failed for a cell matrix. info=" + std::to_string(info));
        }
    }
    cache.valid = true;
}

void solve_cells_cpu(const SolverState2D& state, const CpuLUCache& cache, std::vector<double>& rhs, bool use_openmp) {
    const Problem2D& p = state.problem;
    require(cache.valid, "CPU LU cache is not valid.");
    require(static_cast<int>(rhs.size()) == p.total_unknowns(), "rhs must have total_unknowns() entries.");

    const int n = p.cell_block_size();
    const int nrhs = 1;
    const int lda = n;
    const int ldb = n;
    const char trans = 'N';
#ifndef _OPENMP
    (void)use_openmp;
#endif

#ifdef _OPENMP
    #pragma omp parallel for if(use_openmp)
#endif
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        int info = 0;
        const double* a = cache.lu.data() + cell * p.cell_block_elems();
        const int* ipiv = cache.pivots.data() + cell * n;
        double* b = rhs.data() + cell * n;
        dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("CPU LAPACK dgetrs failed for a cell RHS. info=" + std::to_string(info));
        }
    }
}

IterationStats run_one_timestep_cpu(SolverState2D& state, CpuLUCache& cache, bool use_openmp) {
    const Problem2D& p = state.problem;
    if (!cache.valid || !p.reuse_factorization) {
        factor_cells_cpu(state, cache);
    }

    state.flux_last = p.initialize_from_previous ? state.flux_previous : std::vector<double>(p.total_unknowns(), 0.0);

    IterationStats stats{};
    for (int it = 0; it < p.max_iters; ++it) {
        state.flux_current = state.rhs_const;
        add_upwind_inflow_rhs(state.flux_current, state.flux_last, state);
        solve_cells_cpu(state, cache, state.flux_current, use_openmp);

        stats.final_error = relative_l2_error(state.flux_last, state.flux_current);
        stats.iterations = it + 1;
        state.flux_last.swap(state.flux_current);

        if (stats.final_error < p.convergence_tol) {
            break;
        }

        stats.spectral_radius = stats.final_error / stats.error_previous;
        stats.iterate();
    }

    state.flux_previous = state.flux_last;
    return stats;
}

std::vector<Direction2D> make_tensor_product_quadrature_2d(const std::vector<double>& mu, const std::vector<double>& w) {
    require(mu.size() == w.size(), "mu and w must have the same size.");
    std::vector<Direction2D> dirs;
    dirs.reserve(mu.size() * mu.size());
    for (std::size_t j = 0; j < mu.size(); ++j) {
        for (std::size_t i = 0; i < mu.size(); ++i) {
            dirs.push_back(Direction2D{mu[i], mu[j], w[i] * w[j]});
        }
    }

    double sum_w = 0.0;
    for (const auto& d : dirs) {
        sum_w += d.weight;
    }
    if (sum_w != 0.0) {
        for (auto& d : dirs) {
            d.weight /= sum_w;
        }
    }
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
            throw std::runtime_error("Unsupported level-symmetric order S" + std::to_string(sn_order) + ".");
    }

    std::vector<Direction2D> dirs;
    dirs.reserve(first_octant.size() * 4);
    for (const auto& entry : first_octant) {
        for (double sx : {-1.0, 1.0}) {
            for (double sy : {-1.0, 1.0}) {
                dirs.push_back(Direction2D{sx * entry[0], sy * entry[1], entry[2]});
            }
        }
    }

    double sum_w = 0.0;
    for (const auto& d : dirs) {
        sum_w += d.weight;
    }
    if (sum_w != 0.0) {
        for (auto& d : dirs) {
            d.weight /= sum_w;
        }
    }
    return dirs;
}

} // namespace therefore2d
