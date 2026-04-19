#include "trt2d.hpp"

#include <filesystem>
#if __has_include(<gperftools/profiler.h>)
#include <gperftools/profiler.h>
#else
inline void ProfilerStart(const char*) {}
inline void ProfilerStop() {}
#endif
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
    outputs.output_dir = "results/example_trt_lattice_cpu";
    outputs.series_name = "trt";
    outputs.summary_json = "results/example_trt_lattice_cpu_summary.json";

    CpuLUCache cache;
    ProfilerStart("cpu.prof");
    run_time_trt_cpu(state, cache, use_openmp, outputs);
    ProfilerStop();
    return 0;
}
