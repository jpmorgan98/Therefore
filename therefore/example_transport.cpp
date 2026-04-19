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
    problem.nx = int(48);
    problem.ny = int(32);
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
            cell.x_left = i * cell.dx;
            cell.y_bottom = j * cell.dy;
            cell.dt = 1.0;
            cell.velocity = {1.0, 1.0};
            cell.sigma_t = {2.0, 2.0};
            cell.sigma_s = {
                     0.00, 0.00,
                     0.00, 0.00
                 };

            // if (i < problem.nx / 2) {
            //     cell.sigma_t = {1.1, 0.7};
            //     cell.sigma_s = {
            //         0.96, 0.08,
            //         0.12, 0.30
            //     };
            // } else {
            //     cell.sigma_t = {1.8, 1.1};
            //     cell.sigma_s = {
            //         0.20, 0.03,
            //         0.08, 0.79
            //     };
            // }

            cell.source.assign(problem.cell_block_size(), 0.0);
            // const bool source_patch = (i >= 2 && i <= 5 && j >= 5 && j <= 10);
            for (int dir = 0; dir < problem.num_dirs(); ++dir) {
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                    cell.source[local_angle_group_offset(problem, 0, dir, dof)] = psi_iso;
                    cell.source[local_angle_group_offset(problem, 1, dir, dof)] = psi_iso;
                }
            }
        }
    }

    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    //problem.boundary.east.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    //problem.boundary.north.assign(problem.nx * problem.groups * problem.num_dirs() * 4, 0.0);
    //problem.boundary.south.assign(problem.nx * problem.groups * problem.num_dirs() * 4, 0.0);

    for (int j = 0; j < problem.ny; ++j) {
        for (int g = 0; g < problem.groups; ++g) {
            for (int dir = 0; dir < problem.num_dirs(); ++dir) {
                if (problem.directions[dir].mu > 0.0) {
                    const int off_we = face_offset_west_east(problem, j, g, dir, 0);
                    // problem.boundary.west[off_we + 0] = psi_iso;
                    // problem.boundary.west[off_we + 1] = psi_iso;
                    // problem.boundary.west[off_we + 2] = psi_iso;
                    // problem.boundary.west[off_we + 3] = psi_iso;
                    //for (int dof=0; dof<kDofsPerAngleGroup2D; ++dof){
                    //    problem.boundary.west[off_we + dof] = psi_iso;
                    //    //problem.boundary.east[off_we + dof] = psi_iso;
                    //}
                }
            }
        }
    }

    // for (int i = 0; i < problem.nx; ++i) {
    //     for (int g = 0; g < problem.groups; ++g) {
    //         for (int dir = 0; dir < problem.num_dirs(); ++dir) {
    //             if (problem.directions[dir].mu > 0.0) {
    //                 const int off_ns = face_offset_south_north(problem, i, g, dir, 0);
    //                 for (int dof=0; dof<kDofsPerAngleGroup2D; ++dof){
    //                     problem.boundary.north[off_ns + dof] = psi_iso;
    //                     problem.boundary.south[off_ns + dof] = psi_iso;
    //                 }
    //             }
    //         }
    //     }
    // }

    SolverState2D state;
    state.problem = problem;
    state.cells = cells;

    std::vector<double> initial_condition(problem.total_unknowns(), 0.0);
    // for (int j = 0; j < problem.ny; ++j) {
    //     for (int i = 0; i < problem.nx; ++i) {
    //         const int c = cell_id(i, j, problem.nx);
    //         const double x = cells[c].x_left + 0.5 * cells[c].dx;
    //         const double y = cells[c].y_bottom + 0.5 * cells[c].dy;
    //         const double pulse = 0.2 * std::exp(-0.05 * ((x - 6.0) * (x - 6.0) + (y - 8.0) * (y - 8.0)));
    //         for (int dir = 0; dir < problem.num_dirs(); ++dir) {
    //             for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
    //                 initial_condition[global_offset(problem, c, 0, dir, dof)] = pulse;
    //                 initial_condition[global_offset(problem, c, 1, dir, dof)] = 0.5 * pulse;
    //             }
    //         }
    //     }
    // }

    initialize_state(state, initial_condition);
    assemble_cell_matrices(state);

    CpuLUCache cpu_cache;
#ifdef THEREFORE2D_ENABLE_ROCM
    RocmLUCache rocm_cache;
#endif
    TransportOutputFiles2D outputs;
    outputs.output_dir = "results/example_transport";
    outputs.series_name = "transport";
    outputs.summary_json = "results/example_transport_summary.json";

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
