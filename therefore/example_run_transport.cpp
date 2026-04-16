#include "output.hpp"
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
    const std::string backend_name = use_rocm ? "rocm" : (use_openmp ? "omp_lapack" : "cpu_lapack");

    Problem2D problem;
    problem.nx = 48;
    problem.ny = 32;
    problem.Lx = 1.0;
    problem.Ly = 1.0;
    problem.groups = 2;
    problem.max_iters = 2500;
    problem.convergence_tol = 1.0e-10;
    problem.initialize_from_previous = true;
    problem.reuse_factorization = true;
    problem.directions = make_level_symmetric_quadrature_2d(8);

    std::vector<Cell2D> cells(problem.num_cells());
    for (int j = 0; j < problem.ny; ++j) {
        for (int i = 0; i < problem.nx; ++i) {
            const int c = cell_id(i, j, problem.nx);
            Cell2D& cell = cells[c];
            cell.x_left = static_cast<double>(i);
            cell.y_bottom = static_cast<double>(j);
            cell.dx = problem.Lx/problem.nx;
            cell.dy = problem.Ly/problem.ny;
            cell.dt = 0.10;
            cell.velocity = {1.0, 0.5};

            // if (i < problem.nx / 2) {
            //     cell.sigma_t = {1.1, 0.7};
            //     cell.sigma_s = {
            //         0.7, 0.0,
            //         0.0, 0.7
            //     };
            // } else {
            cell.sigma_t = {1, 1};
            cell.sigma_s = {
                0.99, 0.99,
                0.0, 0.99
            };
            // }

            cell.source.assign(problem.cell_block_size(), 0.0);
            // const bool source_patch = (i >= 2 && i <= 5 && j >= 5 && j <= 10);
            // if (source_patch) {
            //     for (int dir = 0; dir < problem.num_dirs(); ++dir) {
            //         for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
            //             cell.source[local_angle_group_offset(problem, 0, dir, dof)] = 1.0;
            //             cell.source[local_angle_group_offset(problem, 1, dir, dof)] = 1.0;
            //         }
            //     }
            // }
        }
    }

    // Fixed incident inflow on the west boundary for group 0, vacuum elsewhere.
    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    for (int j = 0; j < problem.ny; ++j) {
        for (int dir = 0; dir < problem.num_dirs(); ++dir) {
            if (problem.directions[dir].mu > 0.0) {
                const int off = face_offset_west_east(problem, j, 0, dir, 0);
                problem.boundary.west[off + 0] = 1;
                problem.boundary.west[off + 1] = 1;
                problem.boundary.west[off + 2] = 1;
                problem.boundary.west[off + 3] = 1;
            }
        }
    }

    SolverState2D state;
    state.problem = problem;
    state.cells = cells;

    std::vector<double> initial_condition(problem.total_unknowns(), 0.0);
    for (int j = 0; j < problem.ny; ++j) {
        for (int i = 0; i < problem.nx; ++i) {
            const int c = cell_id(i, j, problem.nx);
            const double x = cells[c].x_left + 0.5 * cells[c].dx;
            const double y = cells[c].y_bottom + 0.5 * cells[c].dy;
            const double pulse = 0; //0.2 * std::exp(-0.05 * ((x - 6.0) * (x - 6.0) + (y - 8.0) * (y - 8.0)));
            for (int dir = 0; dir < problem.num_dirs(); ++dir) {
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                    initial_condition[global_offset(problem, c, 0, dir, dof)] = pulse;
                    initial_condition[global_offset(problem, c, 1, dir, dof)] = 0.5 * pulse;
                }
            }
        }
    }

    initialize_state(state, initial_condition);
    assemble_cell_matrices(state);

    CpuLUCache cpu_cache;
#ifdef THEREFORE2D_ENABLE_ROCM
    RocmLUCache rocm_cache;
#endif
    OutputFiles2D outputs;
    initialize_output_files(outputs);

    const double dt = cells.front().dt;
    double time = 0.0;
    std::vector<TimestepRecord2D> history;
    history.reserve(num_time_steps);

    for (int step = 0; step < num_time_steps; ++step) {
        build_constant_rhs(state);
        IterationStats stats{};
#ifdef THEREFORE2D_ENABLE_ROCM
        if (use_rocm) {
            stats = run_one_timestep_rocm(state, rocm_cache);
        } else
#endif
        {
            stats = run_one_timestep_cpu(state, cpu_cache, use_openmp);
        }
        time += dt;
        history.push_back(TimestepRecord2D{step, time, stats});

        append_angular_flux_csv(outputs.angular_flux_csv, step, time, state.flux_previous);
        append_scalar_flux_csv(outputs.scalar_flux_csv, step, time, state, state.flux_previous);

        std::cout << "step " << step
                  << "  time=" << time
                  << "  iterations=" << stats.iterations
                  << "  final_error=" << stats.final_error << '\n';
    }

    write_summary_json(outputs.summary_json, state, history, backend_name, outputs);
#ifdef THEREFORE2D_ENABLE_ROCM
    if (use_rocm) {
        destroy_rocm_cache(rocm_cache);
    }
#endif

    std::cout << "\nWrote:\n"
              << "  " << outputs.angular_flux_csv << '\n'
              << "  " << outputs.scalar_flux_csv << '\n'
              << "  " << outputs.summary_json << '\n';
    return 0;
}
