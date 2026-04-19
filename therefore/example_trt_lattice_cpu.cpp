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

// ---------------------------------------------------------------------------
// Brunner (2023) test problem definition
// ---------------------------------------------------------------------------
// This file owns all problem-specific data: group boundaries (Table 5),
// material parameters (Tables 3–4), and the Figure 24a lattice geometry.
// The generic TRT solver in trt2d.hpp/cpp has no knowledge of any of this.
// ---------------------------------------------------------------------------

namespace {

using namespace therefore2d;


// ---- Anderson mixing acceleration ---
// Set kUseAnderson = true to enable; compare with false to benchmark.
constexpr bool   kUseAnderson  = true;
constexpr int    kAndersonM    = 5;     // history window (3-5 optimal)
constexpr double kAndersonDamp = 0.8;   // 0.8 empirically better than 1.0

// ---- Group boundaries (Table 5 of Brunner 2023) ---------------------------

std::vector<double> make_table5_group_edges() {
    return {
        1.0e-4, 3.0e-3,
        1.095445115010333e-2, 4.0e-2, 5.0e-2,
        7.825422900366437e-2, 1.224744871391589e-1, 1.916829312738817e-1,
        3.0e-1, 6.708203932499368e-1,
        1.5, 3.240370349203930,
        7.0, 1.114619555925213e1, 1.774823934929885e1, 2.826076380281411e1,
        4.5e1
    };
}

// ---- Material models (Tables 3–4 of Brunner 2023) -------------------------
// Index: 0 = foam, 1 = carbon, 2 = cold_iron, 3 = hot_iron

std::vector<MaterialModelTRT> make_brunner2023_materials() {
    return {
        //             name          rho     cv           T0      {eps_min  eps_edge  C0    C1      C2   Nl dw    ds}
        MaterialModelTRT{"foam",       0.2, 2.41213e14, 1.0e-3, {0.04, 0.3,  2.0,  4.0e2,  0.0,  0, 0.0, 0.0}},
        MaterialModelTRT{"carbon",     2.0, 2.41213e14, 1.0e-3, {0.04, 1.5,  0.77, 1.2e3, 30.0,  1, 0.01, 1.2}},
        MaterialModelTRT{"cold_iron",  6.0, 5.4273e14,  1.0e-3, {0.05, 7.0, 20.1,  1.2e3, 1.2e3, 5, 0.01, 0.2}},
        MaterialModelTRT{"hot_iron",   8.0, 5.4273e14,  5.0e-1, {0.05, 7.0, 20.1,  1.2e3, 1.2e3, 5, 0.01, 0.2}},
    };
}

// ---- Figure 24a geometry --------------------------------------------------
// 3 × 3 cm box.  Material layout (cell-centre rule):
//
//   y\x  [0,1)        [1,2)       [2,3)
//   [2,3) carbon      carbon      carbon
//   [1,2) carbon      HOT IRON    carbon
//   [0,1) carbon      carbon      carbon
//
// (Per Brunner 2023 Sec. 5.1: one iron block at centre is initialised at
//  Thot/2 = 0.5 keV; all other blocks are "cold" at Tcold = 0.001 keV.
//  For the 3×3 mini-lattice (Figure 24a) we place hot iron at centre and
//  carbon at the eight surrounding cells, with foam as background fill if
//  the domain were larger.)

int material_id_figure24a(double x, double y) {
    auto in = [](double v, double lo, double hi) { return v >= lo && v < hi; };

    // Centre block: hot iron
    if (in(x, 1.0, 2.0) && in(y, 1.0, 2.0)) return 3;  // hot_iron

    // Eight surrounding blocks: carbon
    if (in(x, 0.0, 1.0) && in(y, 0.0, 1.0)) return 1;  // carbon  SW
    if (in(x, 1.0, 2.0) && in(y, 0.0, 1.0)) return 1;  // carbon  S
    if (in(x, 2.0, 3.0) && in(y, 0.0, 1.0)) return 1;  // carbon  SE
    if (in(x, 0.0, 1.0) && in(y, 1.0, 2.0)) return 1;  // carbon  W
    if (in(x, 2.0, 3.0) && in(y, 1.0, 2.0)) return 1;  // carbon  E
    if (in(x, 0.0, 1.0) && in(y, 2.0, 3.0)) return 1;  // carbon  NW
    if (in(x, 1.0, 2.0) && in(y, 2.0, 3.0)) return 1;  // carbon  N
    if (in(x, 2.0, 3.0) && in(y, 2.0, 3.0)) return 1;  // carbon  NE

    return 0;  // foam (background if domain were larger)
}

void assign_figure24a_geometry(TrtState2D& state) {
    Problem2D& p = state.transport.problem;
    state.transport.cells.assign(p.num_cells(), Cell2D{});
    state.trt_cells.assign(p.num_cells(), TrtCellState2D{});

    const double dx = p.Lx / static_cast<double>(p.nx);
    const double dy = p.Ly / static_cast<double>(p.ny);

    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int    cell = cell_id(i, j, p.nx);
            const double xc   = (static_cast<double>(i) + 0.5) * dx;
            const double yc   = (static_cast<double>(j) + 0.5) * dy;
            const int    mat  = material_id_figure24a(xc, yc);

            Cell2D& c    = state.transport.cells[cell];
            c.x_left     = static_cast<double>(i) * dx;
            c.y_bottom   = static_cast<double>(j) * dy;
            c.dx         = dx;
            c.dy         = dy;
            c.dt         = state.config.dt;

            state.trt_cells[cell].material_id          = mat;
            state.trt_cells[cell].temperature          = state.materials[mat].initial_temperature;
            state.trt_cells[cell].previous_temperature = state.materials[mat].initial_temperature;
        }
    }
}

TrtState2D make_figure24a_lattice_problem(const TrtConfig2D& config) {
    TrtState2D state;
    state.config      = config;
    state.group_edges = make_table5_group_edges();
    state.materials   = make_brunner2023_materials();

    Problem2D& p  = state.transport.problem;
    p.nx          = config.nx;
    p.ny          = config.ny;
    p.Lx          = config.Lx;
    p.Ly          = config.Ly;
    p.groups      = static_cast<int>(state.group_edges.size()) - 1;
    p.max_iters   = config.max_transport_iters;
    p.num_time_steps = config.num_time_steps;
    p.time_step   = config.dt;
    p.convergence_tol       = config.transport_tol;
    p.initialize_from_previous = true;
    p.reuse_factorization      = false;  // coefficients change each NL iteration
    p.directions = make_level_symmetric_quadrature_2d(config.sn_order);

    assign_figure24a_geometry(state);
    return state;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    using namespace therefore2d;

    TrtConfig2D config;
#ifdef THEREFORE2D_EXAMPLE_USE_OPENMP
    const bool use_openmp = true;
#else
    const bool use_openmp = false;
#endif

    if (argc > 1) config.num_time_steps = std::stoi(argv[1]);
    if (argc > 2) config.dt             = std::stod(argv[2]);

    std::filesystem::create_directories("results");

    TrtState2D state = make_figure24a_lattice_problem(config);
    initialize_trt_state(state);

    TrtOutputFiles2D outputs;
    outputs.output_dir   = "results/example_trt_lattice_cpu";
    outputs.series_name  = "trt";
    outputs.summary_json = "results/example_trt_lattice_cpu_summary.json";
    outputs.save_flux    = false;  // only write temperature fields

    CpuLUCache cache;
    ProfilerStart("cpu.prof");
    run_time_trt_cpu(state, cache, use_openmp, outputs);
    ProfilerStop();
    return 0;
}
