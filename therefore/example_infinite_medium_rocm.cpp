#include "transport2d.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace therefore2d;

constexpr int kNx = 500;
constexpr int kNy = 100;
constexpr int kSN = 8;
constexpr int kNumTimeSteps = 50;
constexpr int kWarmupSteps = 200;
constexpr double kWarmupTol = 1.0e-13;

const double PI = 3.14159265359;

constexpr double kLx = 1.0;
constexpr double kLy = 1.0;
constexpr double kDt = 0.1;
constexpr double kVelocity = 1.0;
constexpr double kSigmaT = 1.0;
constexpr double kSigmaS = 0.95;
constexpr double kSourceIso = 0.25;

void fill_homogeneous_single_group_cells(Problem2D& problem, std::vector<Cell2D>& cells) {
    const double dx = problem.Lx / problem.nx;
    const double dy = problem.Ly / problem.ny;

    for (int j = 0; j < problem.ny; ++j) {
        for (int i = 0; i < problem.nx; ++i) {
            const int c = cell_id(i, j, problem.nx);
            Cell2D& cell = cells[c];

            cell.dx = dx;
            cell.dy = dy;
            cell.x_left = i * dx;
            cell.y_bottom = j * dy;
            cell.dt = kDt;
            cell.velocity = {kVelocity};
            cell.sigma_t = {kSigmaT};
            cell.sigma_s = {kSigmaS};

            cell.source.assign(problem.cell_block_size(), 0.0);
            for (int dir = 0; dir < problem.num_dirs(); ++dir) {
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                    cell.source[local_angle_group_offset(problem, 0, dir, dof)] = kSourceIso;
                }
            }
        }
    }
}

void set_isotropic_inflow_boundaries(Problem2D& problem, double psi_in) {
    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    problem.boundary.east.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    problem.boundary.south.assign(problem.nx * problem.groups * problem.num_dirs() * 4, 0.0);
    problem.boundary.north.assign(problem.nx * problem.groups * problem.num_dirs() * 4, 0.0);

    for (int j = 0; j < problem.ny; ++j) {
        for (int dir = 0; dir < problem.num_dirs(); ++dir) {
            if (problem.directions[dir].mu > 0.0) {
                const int off = face_offset_west_east(problem, j, 0, dir, 0);
                for (int k = 0; k < 4; ++k) problem.boundary.west[off + k] = psi_in;
            }
            if (problem.directions[dir].mu < 0.0) {
                const int off = face_offset_west_east(problem, j, 0, dir, 0);
                for (int k = 0; k < 4; ++k) problem.boundary.east[off + k] = psi_in;
            }
        }
    }

    for (int i = 0; i < problem.nx; ++i) {
        for (int dir = 0; dir < problem.num_dirs(); ++dir) {
            if (problem.directions[dir].eta > 0.0) {
                const int off = face_offset_south_north(problem, i, 0, dir, 0);
                for (int k = 0; k < 4; ++k) problem.boundary.south[off + k] = psi_in;
            }
            if (problem.directions[dir].eta < 0.0) {
                const int off = face_offset_south_north(problem, i, 0, dir, 0);
                for (int k = 0; k < 4; ++k) problem.boundary.north[off + k] = psi_in;
            }
        }
    }
}

SolverState2D make_state() {
    Problem2D problem;
    problem.nx = kNx;
    problem.ny = kNy;
    problem.Lx = kLx;
    problem.Ly = kLy;
    problem.groups = 1;
    problem.max_iters = 500;
    problem.num_time_steps = kNumTimeSteps;
    problem.time_step = kDt;
    problem.convergence_tol = 1.0e-12;
    problem.initialize_from_previous = true;
    problem.reuse_factorization = true;
    problem.directions = make_level_symmetric_quadrature_2d(kSN);

    const double sigma_a = kSigmaT - kSigmaS;
    const double psi_inf = (sigma_a > 0.0) ? (kSourceIso / sigma_a) : 0.0;

    std::vector<Cell2D> cells(problem.num_cells());
    fill_homogeneous_single_group_cells(problem, cells);
    set_isotropic_inflow_boundaries(problem, psi_inf);

    SolverState2D state;
    state.problem = problem;
    state.cells = cells;
    return state;
}

std::vector<double> compute_discrete_equilibrium(bool use_openmp) {
    SolverState2D state = make_state();
    std::vector<double> initial(state.problem.total_unknowns(), 0.0);
    initialize_state(state, initial);
    assemble_cell_matrices(state);

    CpuLUCache cache;
    std::vector<double> previous = state.flux_previous;
    for (int step = 0; step < kWarmupSteps; ++step) {
        build_constant_rhs(state);
        run_one_timestep_cpu(state, cache, use_openmp);
        const double change = relative_l2_error(previous, state.flux_previous);
        if (change < kWarmupTol) {
            std::cout << "Computed discrete equilibrium in " << (step + 1)
                      << " warmup steps with change=" << change << '\n';
            return state.flux_previous;
        }
        previous = state.flux_previous;
    }

    std::cout << "Warmup hit the limit; using the last state as the discrete equilibrium guess.\n";
    return state.flux_previous;
}

} // namespace

int main(int argc, char** argv) {
    using namespace therefore2d;

    const std::string init_mode = (argc > 1) ? std::string(argv[1]) : "equilibrium";
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

    SolverState2D state = make_state();

    std::vector<double> initial_condition(state.problem.total_unknowns(), 0.0);
    if (init_mode == "equilibrium") {
        initial_condition = compute_discrete_equilibrium(use_openmp);
    } else if (init_mode != "zero") {
        std::cerr << "Unknown init mode '" << init_mode << "'. Use: equilibrium or zero\n";
        return 1;
    }

    initialize_state(state, initial_condition);
    assemble_cell_matrices(state);

    CpuLUCache cpu_cache;
#ifdef THEREFORE2D_ENABLE_ROCM
    RocmLUCache rocm_cache;
#endif

    TransportOutputFiles2D outputs;
    outputs.output_dir = "results/example_infinite_medium_rocm";
    outputs.series_name = "transport";
    outputs.summary_json = "results/example_infinite_medium_rocm_summary.json";

#ifdef THEREFORE2D_ENABLE_ROCM
    if (use_rocm) {
        run_time_rocm(state, rocm_cache, outputs);
        destroy_rocm_cache(rocm_cache);
    } else
#endif
    {
        run_time_cpu(state, cpu_cache, use_openmp, outputs);
    }

    const double sigma_a = kSigmaT - kSigmaS;
    std::cout << "\nSingle-group homogeneous medium\n"
              << "  isotropic source = " << kSourceIso << '\n'
              << "  sigma_t = " << kSigmaT << '\n'
              << "  sigma_s = " << kSigmaS << '\n'
              << "  sigma_a = " << sigma_a << '\n';
    if (sigma_a > 0.0) {
        std::cout << "  continuous infinite-medium scalar = " << (kSourceIso / sigma_a) << '\n';
    }
    std::cout << "  init_mode = " << init_mode << '\n'
              << "Wrote:\n"
              << "  " << outputs.output_dir + "/" + outputs.series_name + ".pvd" << '\n'
              << "  " << outputs.summary_json << '\n';
    return 0;
}
