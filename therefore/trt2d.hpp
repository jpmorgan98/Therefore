#ifndef THEREFORE_TRT2D_HPP
#define THEREFORE_TRT2D_HPP

#include "output.hpp"
#include "transport2d.hpp"

#include <string>
#include <vector>

namespace therefore2d {

struct AnalyticOpacityParams {
    double epsilon_min = 0.0;
    double epsilon_edge = 0.0;
    double C0 = 0.0;
    double C1 = 0.0;
    double C2 = 0.0;
    int num_lines = 0;
    double delta_w = 0.0;
    double delta_s = 0.0;
};

struct MaterialModelTRT {
    std::string name;
    double density = 0.0;
    double cv = 0.0;
    double initial_temperature = 0.0;
    AnalyticOpacityParams opacity;
};

struct TrtCellState2D {
    int material_id = -1;
    double temperature = 0.0;
    double previous_temperature = 0.0;
};

struct TrtConfig2D {
    int nx = 18;
    int ny = 18;
    int sn_order = 8;
    int num_time_steps = 40;
    int max_nonlinear_iters = 10;
    int max_transport_iters = 300;
    double dt = 1.0e-11;
    double transport_tol = 1.0e-10;
    double nonlinear_tol = 1.0e-6;
    double cold_boundary_temperature = 1.0e-3;
    double hot_boundary_temperature = 1.0;
    double temperature_floor = 1.0e-3;
    double Lx = 3.0;
    double Ly = 3.0;
};

struct TrtTimestepStats2D {
    int step = 0;
    double time = 0.0;
    int nonlinear_iterations = 0;
    double max_temperature_change = 0.0;
    IterationStats transport_stats;
};

struct TrtOutputFiles2D {
    std::string scalar_flux_csv = "results/trt_scalar_flux_history.csv";
    std::string radiation_temperature_csv = "results/trt_radiation_temperature_history.csv";
    std::string material_temperature_csv = "results/trt_material_temperature_history.csv";
    std::string summary_json = "results/trt_run_summary.json";
};

struct TrtState2D {
    TrtConfig2D config;
    SolverState2D transport;
    std::vector<MaterialModelTRT> materials;
    std::vector<TrtCellState2D> trt_cells;
    std::vector<double> group_edges;
    std::vector<TrtTimestepStats2D> history;
};

std::vector<double> make_table5_group_edges();
std::vector<MaterialModelTRT> make_trt_materials();
TrtState2D make_figure24a_lattice_problem(const TrtConfig2D& config = TrtConfig2D{});
void initialize_trt_state(TrtState2D& state);
TrtTimestepStats2D run_one_timestep_trt_cpu(TrtState2D& state, CpuLUCache& cache, bool use_openmp);
void initialize_trt_output_files(const TrtOutputFiles2D& files);
void append_trt_timestep_outputs(const TrtState2D& state,
                                 const TrtOutputFiles2D& files,
                                 int time_step,
                                 double time);
void write_trt_outputs(const TrtState2D& state, const TrtOutputFiles2D& files);

} // namespace therefore2d

#endif
