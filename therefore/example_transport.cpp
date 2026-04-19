#include "transport2d.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace therefore2d;

    const int num_time_steps = 1;
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
    problem.nx = 48;
    problem.ny = 32;
    problem.Lx = 48 * 0.5;   // cell dx = 0.5, so Lx = nx * dx
    problem.Ly = 32 * 0.5;   // cell dy = 0.5, so Ly = ny * dy
    problem.groups = 2;
    problem.max_iters = 1000;
    problem.num_time_steps = num_time_steps;
    problem.time_step = 1.0;
    problem.convergence_tol = 1.0e-10;
    problem.initialize_from_previous = true;
    problem.reuse_factorization = true;
    problem.directions = make_level_symmetric_quadrature_2d(8);

    const double psi_iso = 1;

    std::vector<Cell2D> cells(problem.num_cells());
    for (int j = 0; j < problem.ny; ++j) {
        for (int i = 0; i < problem.nx; ++i) {
            const int c = cell_id(i, j, problem.nx);
            Cell2D& cell = cells[c];
            cell.dx = 0.5;
            cell.dy = 0.5;
            cell.x_left   = i * cell.dx;
            cell.y_bottom = j * cell.dy;
            cell.dt = 1.0;
            cell.velocity = {1.0, 1.0};
            cell.sigma_t  = {2.0, 2.0};
            cell.sigma_s  = {
                0.00, 0.00,
                0.00, 0.00
            };

            cell.source.assign(problem.cell_block_size(), 0.0);
            for (int dir = 0; dir < problem.num_dirs(); ++dir) {
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                    cell.source[local_angle_group_offset(problem, 0, dir, dof)] = psi_iso;
                    cell.source[local_angle_group_offset(problem, 1, dir, dof)] = psi_iso;
                }
            }
        }
    }

    // Vacuum on the west boundary; all other boundaries default vacuum.
    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);

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
    outputs.output_dir   = "results/example_transport";
    outputs.series_name  = "transport";
    outputs.summary_json = "results/example_transport_summary.json";
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
