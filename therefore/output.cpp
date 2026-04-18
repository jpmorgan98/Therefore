#include "output.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace therefore2d {

namespace {

double average_8(const SolverState2D& state, const std::vector<double>& flux, int cell, int group, int dir) {
    double sum = 0.0;
    for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
        sum += flux[global_offset(state.problem, cell, group, dir, dof)];
    }
    return sum / static_cast<double>(kDofsPerAngleGroup2D);
}

} // namespace

double cell_centered_scalar_flux(const SolverState2D& state, const std::vector<double>& flux, int cell, int group) {
    const Problem2D& p = state.problem;
    double value = 0.0;
    double weight_sum = 0.0;
    for (int dir = 0; dir < p.num_dirs(); ++dir) {
        const double w = p.directions[dir].weight;
        value += w * average_8(state, flux, cell, group, dir);
        weight_sum += w;
    }
    return (weight_sum != 0.0) ? (value / weight_sum) : 0.0;
}

void initialize_scalar_flux_csv(const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open scalar flux CSV for writing: " + path);
    }
    out << "time_step,time,cell,i,j,group,x_center,y_center,value\n";
}

void initialize_cell_field_csv(const std::string& path,
                               const std::string& value_name,
                               bool include_material_name) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open cell field CSV for writing: " + path);
    }
    out << "time_step,time,cell,i,j,x_center,y_center";
    if (include_material_name) {
        out << ",material";
    }
    out << ',' << value_name << '\n';
}

void initialize_output_files(const OutputFiles2D& files) {
    {
        std::ofstream out(files.angular_flux_csv);
        if (!out) {
            throw std::runtime_error("Could not open angular flux CSV for writing: " + files.angular_flux_csv);
        }
        out << "time_step,time,index,value\n";
    }
    initialize_scalar_flux_csv(files.scalar_flux_csv);
}

void append_angular_flux_csv(const std::string& path, int time_step, double time, const std::vector<double>& flux) {
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Could not append angular flux CSV: " + path);
    }
    out << std::setprecision(16);
    for (std::size_t i = 0; i < flux.size(); ++i) {
        out << time_step << ',' << time << ',' << i << ',' << flux[i] << '\n';
    }
}

void append_scalar_flux_csv(const std::string& path, int time_step, double time, const SolverState2D& state, const std::vector<double>& flux) {
    const Problem2D& p = state.problem;
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Could not append scalar flux CSV: " + path);
    }
    out << std::setprecision(16);
    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            const Cell2D& c = state.cells[cell];
            const double x_center = c.x_left + 0.5 * c.dx;
            const double y_center = c.y_bottom + 0.5 * c.dy;
            for (int g = 0; g < p.groups; ++g) {
                out << time_step << ',' << time << ',' << cell << ',' << i << ',' << j << ',' << g << ','
                    << x_center << ',' << y_center << ',' << cell_centered_scalar_flux(state, flux, cell, g) << '\n';
            }
        }
    }
}

void append_cell_field_csv(const std::string& path,
                           int time_step,
                           double time,
                           const SolverState2D& state,
                           const std::vector<double>& values,
                           const std::vector<std::string>* material_names) {
    const Problem2D& p = state.problem;
    if (static_cast<int>(values.size()) != p.num_cells()) {
        throw std::runtime_error("append_cell_field_csv expected one value per cell.");
    }
    if (material_names != nullptr && static_cast<int>(material_names->size()) != p.num_cells()) {
        throw std::runtime_error("append_cell_field_csv material_names must match num_cells.");
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Could not append cell field CSV: " + path);
    }
    out << std::setprecision(16);
    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            const Cell2D& c = state.cells[cell];
            const double x_center = c.x_left + 0.5 * c.dx;
            const double y_center = c.y_bottom + 0.5 * c.dy;
            out << time_step << ',' << time << ',' << cell << ',' << i << ',' << j << ','
                << x_center << ',' << y_center;
            if (material_names != nullptr) {
                out << ',' << (*material_names)[cell];
            }
            out << ',' << values[cell] << '\n';
        }
    }
}

void write_summary_json(const std::string& path,
                        const SolverState2D& state,
                        const std::vector<TimestepRecord2D>& history,
                        const std::string& backend_name,
                        const OutputFiles2D& files) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open summary JSON for writing: " + path);
    }

    out << std::setprecision(16);
    out << "{\n";
    out << "  \"backend\": \"" << backend_name << "\",\n";
    out << "  \"nx\": " << state.problem.nx << ",\n";
    out << "  \"ny\": " << state.problem.ny << ",\n";
    out << "  \"groups\": " << state.problem.groups << ",\n";
    out << "  \"num_dirs\": " << state.problem.num_dirs() << ",\n";
    out << "  \"cell_block_size\": " << state.problem.cell_block_size() << ",\n";
    out << "  \"total_unknowns\": " << state.problem.total_unknowns() << ",\n";
    out << "  \"angular_flux_csv\": \"" << files.angular_flux_csv << "\",\n";
    out << "  \"scalar_flux_csv\": \"" << files.scalar_flux_csv << "\",\n";
    out << "  \"time_history\": [\n";
    for (std::size_t k = 0; k < history.size(); ++k) {
        const auto& rec = history[k];
        out << "    {\"step\": " << rec.step
            << ", \"time\": " << rec.time
            << ", \"iterations\": " << rec.stats.iterations
            << ", \"final_error\": " << rec.stats.final_error << "}";
        if (k + 1 != history.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace therefore2d
