#ifndef THEREFORE_PARAVIEW_OUTPUT_HPP
#define THEREFORE_PARAVIEW_OUTPUT_HPP

// output.hpp depends on both core solver headers.
#include "transport2d.hpp"
#include "trt2d.hpp"

#include <string>
#include <vector>

namespace therefore2d {

// ---------------------------------------------------------------------------
// VTK rectilinear grid field types
// ---------------------------------------------------------------------------

struct CellScalarField2D {
    std::string         name;
    std::vector<double> values;
};

struct RectilinearGrid2D {
    int                 nx = 0;
    int                 ny = 0;
    std::vector<double> x_edges;
    std::vector<double> y_edges;
};

// ---------------------------------------------------------------------------
// ParaView time-series writer
// ---------------------------------------------------------------------------

struct ParaviewSeriesConfig2D {
    std::string output_dir   = "results/paraview";
    std::string series_name  = "solution";
    bool write_pvd_every_step = true;
};

class ParaviewSeriesWriter2D {
public:
    ParaviewSeriesWriter2D(RectilinearGrid2D grid,
                           ParaviewSeriesConfig2D config = ParaviewSeriesConfig2D{});

    void write_step(int step, double time,
                    const std::vector<CellScalarField2D>& cell_fields);

    std::string pvd_path() const;

private:
    struct StepRecord {
        int    step = 0;
        double time = 0.0;
        std::string relative_file;
    };

    RectilinearGrid2D      grid_;
    ParaviewSeriesConfig2D config_;
    std::vector<StepRecord> records_;

    std::string make_step_filename(int step) const;
    std::string make_step_relative_filename(int step) const;
    void write_vtr_file(const std::string& path,
                        const std::vector<CellScalarField2D>& cell_fields) const;
    void write_pvd_file() const;
};

// ---------------------------------------------------------------------------
// Grid and field helpers
// ---------------------------------------------------------------------------

RectilinearGrid2D make_rectilinear_grid(const SolverState2D& state);

CellScalarField2D make_cell_scalar_field(const std::string& name,
                                         const std::vector<double>& values);

/// One scalar-flux field per energy group, named prefix + group_index.
std::vector<CellScalarField2D> make_scalar_flux_group_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix = "scalar_flux_g");

/// One cell-average angular-flux field per (group, direction).
std::vector<CellScalarField2D> make_angular_flux_group_dir_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix = "angular_flux");

/// Alias for make_scalar_flux_group_fields (kept for back-compat).
std::vector<CellScalarField2D> make_angle_averaged_flux_fields(
    const SolverState2D& state,
    const std::vector<double>& flux,
    const std::string& prefix = "angle_avg_flux_g");

void append_fields(std::vector<CellScalarField2D>& dst,
                   const std::vector<CellScalarField2D>& src);

// ---------------------------------------------------------------------------
// JSON summary writers
// ---------------------------------------------------------------------------

/// Write a JSON summary for a standalone neutron-transport run.
/// Called by run_time_cpu / run_time_rocm.
void write_transport_summary_json(
    const std::string&                   path,
    const SolverState2D&                 state,
    const std::vector<TimestepRecord2D>& history,
    const std::string&                   backend_name,
    const std::string&                   pvd_path);

/// Write a JSON summary for a TRT run.
/// Called by run_time_trt_cpu.
void write_trt_summary_json(
    const std::string&                       path,
    const TrtState2D&                        state,
    const std::string&                       pvd_path);

} // namespace therefore2d

#endif // THEREFORE_PARAVIEW_OUTPUT_HPP
