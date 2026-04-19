#include "output.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace therefore2d {
namespace {

void require_grid_valid(const RectilinearGrid2D& grid) {
    require(grid.nx > 0, "RectilinearGrid2D.nx must be positive.");
    require(grid.ny > 0, "RectilinearGrid2D.ny must be positive.");
    require(static_cast<int>(grid.x_edges.size()) == grid.nx + 1,
            "RectilinearGrid2D.x_edges must contain nx + 1 entries.");
    require(static_cast<int>(grid.y_edges.size()) == grid.ny + 1,
            "RectilinearGrid2D.y_edges must contain ny + 1 entries.");
}

void require_fields_valid(const RectilinearGrid2D& grid,
                          const std::vector<CellScalarField2D>& cell_fields) {
    const int expected = grid.nx * grid.ny;
    for (const auto& field : cell_fields) {
        if (static_cast<int>(field.values.size()) != expected) {
            throw std::runtime_error("Cell field '" + field.name
                                     + "' does not have nx * ny values.");
        }
    }
}

std::string xml_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out.push_back(ch); break;
        }
    }
    return out;
}

template <class Stream>
void write_ascii_array(Stream& out, const std::vector<double>& values) {
    out << std::setprecision(16);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ' ';
        out << values[i];
    }
    out << '\n';
}

std::vector<double> x_edges_from_state(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    std::vector<double> edges(p.nx + 1, 0.0);
    for (int i = 0; i < p.nx; ++i) {
        const int c = cell_id(i, 0, p.nx);
        edges[i]     = state.cells[c].x_left;
        edges[i + 1] = state.cells[c].x_left + state.cells[c].dx;
    }
    return edges;
}

std::vector<double> y_edges_from_state(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    std::vector<double> edges(p.ny + 1, 0.0);
    for (int j = 0; j < p.ny; ++j) {
        const int c = cell_id(0, j, p.nx);
        edges[j]     = state.cells[c].y_bottom;
        edges[j + 1] = state.cells[c].y_bottom + state.cells[c].dy;
    }
    return edges;
}

} // namespace

// ---------------------------------------------------------------------------
// ParaviewSeriesWriter2D
// ---------------------------------------------------------------------------

ParaviewSeriesWriter2D::ParaviewSeriesWriter2D(RectilinearGrid2D grid,
                                               ParaviewSeriesConfig2D config)
    : grid_(std::move(grid)), config_(std::move(config)) {
    require_grid_valid(grid_);
    std::filesystem::create_directories(config_.output_dir);
}

void ParaviewSeriesWriter2D::write_step(int step,
                                        double time,
                                        const std::vector<CellScalarField2D>& cell_fields) {
    require_fields_valid(grid_, cell_fields);

    const std::string relative_name = make_step_relative_filename(step);
    const std::string full_name     = make_step_filename(step);
    write_vtr_file(full_name, cell_fields);

    records_.push_back(StepRecord{step, time, relative_name});
    if (config_.write_pvd_every_step) {
        write_pvd_file();
    }
}

std::string ParaviewSeriesWriter2D::pvd_path() const {
    return config_.output_dir + "/" + config_.series_name + ".pvd";
}

std::string ParaviewSeriesWriter2D::make_step_filename(int step) const {
    return config_.output_dir + "/" + make_step_relative_filename(step);
}

std::string ParaviewSeriesWriter2D::make_step_relative_filename(int step) const {
    std::ostringstream name;
    name << config_.series_name << '_'
         << std::setw(6) << std::setfill('0') << step << ".vtr";
    return name.str();
}

void ParaviewSeriesWriter2D::write_vtr_file(
    const std::string& path,
    const std::vector<CellScalarField2D>& cell_fields) const {

    std::ofstream out(path);
    if (!out) throw std::runtime_error("Could not open VTK file for writing: " + path);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <RectilinearGrid WholeExtent=\"0 " << grid_.nx
        << " 0 " << grid_.ny << " 0 0\">\n";
    out << "    <Piece Extent=\"0 " << grid_.nx
        << " 0 " << grid_.ny << " 0 0\">\n";

    if (!cell_fields.empty()) {
        out << "      <CellData Scalars=\"" << xml_escape(cell_fields.front().name) << "\">\n";
        for (const auto& field : cell_fields) {
            out << "        <DataArray type=\"Float64\" Name=\""
                << xml_escape(field.name) << "\" format=\"ascii\">\n          ";
            write_ascii_array(out, field.values);
            out << "        </DataArray>\n";
        }
        out << "      </CellData>\n";
    } else {
        out << "      <CellData/>\n";
    }

    out << "      <PointData/>\n";
    out << "      <Coordinates>\n";

    out << "        <DataArray type=\"Float64\" Name=\"XCoordinates\" format=\"ascii\">\n          ";
    write_ascii_array(out, grid_.x_edges);
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"YCoordinates\" format=\"ascii\">\n          ";
    write_ascii_array(out, grid_.y_edges);
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"ZCoordinates\" format=\"ascii\">\n"
           "          0\n"
           "        </DataArray>\n";

    out << "      </Coordinates>\n";
    out << "    </Piece>\n";
    out << "  </RectilinearGrid>\n";
    out << "</VTKFile>\n";
}

void ParaviewSeriesWriter2D::write_pvd_file() const {
    std::ofstream out(pvd_path());
    if (!out) throw std::runtime_error("Could not open PVD file for writing: " + pvd_path());

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <Collection>\n";
    out << std::setprecision(16);
    for (const auto& record : records_) {
        out << "    <DataSet timestep=\"" << record.time
            << "\" group=\"\" part=\"0\" file=\""
            << xml_escape(record.relative_file) << "\"/>\n";
    }
    out << "  </Collection>\n";
    out << "</VTKFile>\n";
}

// ---------------------------------------------------------------------------
// Grid / field factory helpers
// ---------------------------------------------------------------------------

RectilinearGrid2D make_rectilinear_grid(const SolverState2D& state) {
    RectilinearGrid2D grid;
    grid.nx      = state.problem.nx;
    grid.ny      = state.problem.ny;
    grid.x_edges = x_edges_from_state(state);
    grid.y_edges = y_edges_from_state(state);
    return grid;
}

CellScalarField2D make_cell_scalar_field(const std::string& name,
                                         const std::vector<double>& values) {
    return CellScalarField2D{name, values};
}

std::vector<CellScalarField2D> make_scalar_flux_group_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix) {

    const Problem2D& p = state.problem;
    std::vector<CellScalarField2D> fields;
    fields.reserve(p.groups);

    for (int g = 0; g < p.groups; ++g) {
        std::vector<double> values(p.num_cells(), 0.0);
        for (int j = 0; j < p.ny; ++j) {
            for (int i = 0; i < p.nx; ++i) {
                const int c = cell_id(i, j, p.nx);
                values[c]   = cell_centered_scalar_flux(state, flux, c, g);
            }
        }
        fields.push_back(CellScalarField2D{prefix + std::to_string(g), std::move(values)});
    }
    return fields;
}

std::vector<CellScalarField2D> make_angular_flux_group_dir_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix) {

    const Problem2D& p = state.problem;
    std::vector<CellScalarField2D> fields;
    fields.reserve(p.groups * p.num_dirs());

    for (int g = 0; g < p.groups; ++g) {
        for (int d = 0; d < p.num_dirs(); ++d) {
            std::vector<double> values(p.num_cells(), 0.0);
            for (int j = 0; j < p.ny; ++j) {
                for (int i = 0; i < p.nx; ++i) {
                    const int c = cell_id(i, j, p.nx);
                    values[c]   = cell_average_angular_flux(state, flux, c, g, d);
                }
            }
            fields.push_back(CellScalarField2D{
                prefix + "_g" + std::to_string(g) + "_dir" + std::to_string(d),
                std::move(values)});
        }
    }
    return fields;
}

std::vector<CellScalarField2D> make_angle_averaged_flux_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix) {
    return make_scalar_flux_group_fields(state, flux, prefix);
}

void append_fields(std::vector<CellScalarField2D>& dst,
                   const std::vector<CellScalarField2D>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

// ---------------------------------------------------------------------------
// JSON summary writers
// ---------------------------------------------------------------------------

void write_transport_summary_json(
    const std::string&                   path,
    const SolverState2D&                 state,
    const std::vector<TimestepRecord2D>& history,
    const std::string&                   backend_name,
    const std::string&                   pvd_path) {

    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path())
        std::filesystem::create_directories(out_path.parent_path());

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Could not open transport summary JSON: " + path);

    out << std::setprecision(16);
    out << "{\n";
    out << "  \"backend\": \""          << backend_name               << "\",\n";
    out << "  \"nx\": "                 << state.problem.nx           << ",\n";
    out << "  \"ny\": "                 << state.problem.ny           << ",\n";
    out << "  \"groups\": "             << state.problem.groups       << ",\n";
    out << "  \"num_dirs\": "           << state.problem.num_dirs()   << ",\n";
    out << "  \"cell_block_size\": "    << state.problem.cell_block_size() << ",\n";
    out << "  \"total_unknowns\": "     << state.problem.total_unknowns()  << ",\n";
    out << "  \"paraview_pvd\": \""     << pvd_path                   << "\",\n";
    out << "  \"time_history\": [\n";
    for (std::size_t k = 0; k < history.size(); ++k) {
        const auto& rec = history[k];
        out << "    {\"step\": "            << rec.step
            << ", \"time\": "              << rec.time
            << ", \"iterations\": "        << rec.stats.iterations
            << ", \"final_error\": "       << rec.stats.final_error
            << ", \"spectral_radius\": "   << rec.stats.spectral_radius << "}";
        if (k + 1 != history.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

void write_trt_summary_json(
    const std::string& path,
    const TrtState2D&  state,
    const std::string& pvd_path) {

    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path())
        std::filesystem::create_directories(out_path.parent_path());

    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("Could not open TRT summary JSON: " + path);

    const Problem2D& p = state.transport.problem;
    out << std::setprecision(16);
    out << "{\n";
    out << "  \"nx\": "               << p.nx                         << ",\n";
    out << "  \"ny\": "               << p.ny                         << ",\n";
    out << "  \"groups\": "           << p.groups                     << ",\n";
    out << "  \"num_dirs\": "         << p.num_dirs()                  << ",\n";
    out << "  \"dt\": "               << state.config.dt              << ",\n";
    out << "  \"num_time_steps\": "   << state.config.num_time_steps  << ",\n";
    out << "  \"paraview_pvd\": \""   << pvd_path                     << "\",\n";
    out << "  \"history\": [\n";
    for (std::size_t k = 0; k < state.history.size(); ++k) {
        const auto& rec = state.history[k];
        out << "    {\"step\": "                      << rec.step
            << ", \"time\": "                        << rec.time
            << ", \"nonlinear_iterations\": "        << rec.nonlinear_iterations
            << ", \"max_temperature_change\": "      << rec.max_temperature_change
            << ", \"transport_iterations\": "        << rec.transport_stats.iterations
            << ", \"transport_final_error\": "       << rec.transport_stats.final_error
            << ", \"transport_spectral_radius\": "   << rec.transport_stats.spectral_radius << "}";
        if (k + 1 != state.history.size()) out << ',';
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace therefore2d
