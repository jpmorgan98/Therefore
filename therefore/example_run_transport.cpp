#include "transport2d.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace therefore2d;

    const int num_time_steps = 24;
#ifdef THEREFORE2D_EXAMPLE_USE_OPENMP
    const bool use_openmp = true;
#else
    const bool use_openmp = false;
#endif
#if defined(THEREFORE2D_ENABLE_ROCM) && defined(THEREFORE2D_EXAMPLE_USE_ROCM)
    const bool use_rocm = true;
#else
    const bool use_rocm = false;
#endif

    Problem2D problem;
    problem.nx           = 48;
    problem.ny           = 32;
    problem.Lx           = 1.0;
    problem.Ly           = 1.0;
    problem.groups       = 2;
    problem.max_iters    = 2500;
    problem.num_time_steps = num_time_steps;
    problem.time_step    = 0.10;
    problem.convergence_tol       = 1.0e-10;
    problem.initialize_from_previous = true;
    problem.reuse_factorization      = true;
    problem.directions = make_level_symmetric_quadrature_2d(8);

    const double dx = problem.Lx / problem.nx;
    const double dy = problem.Ly / problem.ny;

    std::vector<Cell2D> cells(problem.num_cells());
    for (int j = 0; j < problem.ny; ++j) {
        for (int i = 0; i < problem.nx; ++i) {
            const int c  = cell_id(i, j, problem.nx);
            Cell2D& cell = cells[c];
            cell.x_left    = static_cast<double>(i) * dx;  // BUG FIX: was (double)i
            cell.y_bottom  = static_cast<double>(j) * dy;
            cell.dx        = dx;
            cell.dy        = dy;
            cell.dt        = 0.10;
            cell.velocity  = {1.0, 0.5};
            cell.sigma_t   = {1.0, 1.0};
            cell.sigma_s   = {0.99, 0.99,
                              0.0,  0.99};
            cell.source.assign(problem.cell_block_size(), 0.0);
        }
    }

    // Fixed incident inflow on the west boundary for group 0, vacuum elsewhere.
    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    for (int j = 0; j < problem.ny; ++j) {
        for (int dir = 0; dir < problem.num_dirs(); ++dir) {
            if (problem.directions[dir].mu > 0.0) {
                const int off = face_offset_west_east(problem, j, 0, dir, 0);
                for (int k = 0; k < 4; ++k)
                    problem.boundary.west[off + k] = 1.0;
            }
        }
    }

    SolverState2D state;
    state.problem = problem;
    state.cells   = cells;

    std::vector<double> initial_condition(problem.total_unknowns(), 0.0);
    initialize_state(state, initial_condition);
    assemble_cell_matrices(state);

    CpuLUCache cpu_cache;
#ifdef THEREFORE2D_ENABLE_ROCM
    RocmLUCache rocm_cache;
#endif

    TransportOutputFiles2D outputs;
    outputs.output_dir   = "results/example_run_transport";
    outputs.series_name  = "transport";
    outputs.summary_json = "results/example_run_transport_summary.json";
    outputs.save_flux    = true;

#ifdef THEREFORE2D_ENABLE_ROCM
    if (use_rocm) {
        run_time_rocm(state, rocm_cache, outputs);
        destroy_rocm_cache(rocm_cache);
    } else
#endif
    {
        run_time_cpu(state, cpu_cache, use_openmp, outputs);
    }

    return 0;
}
