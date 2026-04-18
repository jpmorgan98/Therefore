#include "trt2d.hpp"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace therefore2d;

    TrtConfig2D config;
#ifdef THEREFORE2D_EXAMPLE_USE_OPENMP
    const bool use_openmp = true;
#else
    const bool use_openmp = false;
#endif

    if (argc > 1) config.num_time_steps = std::stoi(argv[1]);
    if (argc > 2) config.dt = std::stod(argv[2]);

    std::filesystem::create_directories("results");

    TrtState2D state = make_figure24a_lattice_problem(config);
    initialize_trt_state(state);

    TrtOutputFiles2D outputs;
    initialize_trt_output_files(outputs);

    CpuLUCache cache;
    double time = 0.0;
    for (int step = 0; step < config.num_time_steps; ++step) {
        std::cout << "TRT STEP " << step << '\n';
        TrtTimestepStats2D stats = run_one_timestep_trt_cpu(state, cache, use_openmp);
        time += config.dt;
        stats.step = step;
        stats.time = time;
        state.history.push_back(stats);

        append_trt_timestep_outputs(state, outputs, step, time);

        std::cout << "  time=" << time
                  << "  nonlinear_iters=" << stats.nonlinear_iterations
                  << "  max_dT_rel=" << stats.max_temperature_change
                  << "  transport_iters=" << stats.transport_stats.iterations
                  << "  transport_error=" << stats.transport_stats.final_error
                  << '\n';
    }

    write_trt_outputs(state, outputs);
    std::cout << "Wrote:\n"
              << "  " << outputs.scalar_flux_csv << '\n'
              << "  " << outputs.radiation_temperature_csv << '\n'
              << "  " << outputs.material_temperature_csv << '\n'
              << "  " << outputs.summary_json << '\n';
    return 0;
}
