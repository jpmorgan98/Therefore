#ifndef THEREFORE_OUTPUT_HPP
#define THEREFORE_OUTPUT_HPP

#include "transport2d.hpp"

#include <string>
#include <vector>

namespace therefore2d {

struct TimestepRecord2D {
    int step = 0;
    double time = 0.0;
    IterationStats stats;
};

struct OutputFiles2D {
    std::string angular_flux_csv = "results/angular_flux_history.csv";
    std::string scalar_flux_csv = "results/scalar_flux_history.csv";
    std::string summary_json = "results/run_summary.json";
};

double cell_centered_scalar_flux(const SolverState2D& state, const std::vector<double>& flux, int cell, int group);

void initialize_scalar_flux_csv(const std::string& path);
void initialize_cell_field_csv(const std::string& path,
                               const std::string& value_name,
                               bool include_material_name = false);
void initialize_output_files(const OutputFiles2D& files);
void append_angular_flux_csv(const std::string& path, int time_step, double time, const std::vector<double>& flux);
void append_scalar_flux_csv(const std::string& path, int time_step, double time, const SolverState2D& state, const std::vector<double>& flux);
void append_cell_field_csv(const std::string& path,
                           int time_step,
                           double time,
                           const SolverState2D& state,
                           const std::vector<double>& values,
                           const std::vector<std::string>* material_names = nullptr);
void write_summary_json(const std::string& path,
                        const SolverState2D& state,
                        const std::vector<TimestepRecord2D>& history,
                        const std::string& backend_name,
                        const OutputFiles2D& files);

} // namespace therefore2d

#endif
