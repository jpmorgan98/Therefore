/// example_trt_hotwall_vacuum_rocm.cpp
///
/// Grey TRT Marshak wave — ROCm GPU back-end.
///
/// Physics identical to example_trt_hotwall_vacuum_cpu.cpp:
///   hot wall (left) + vacuum BCs on all four edges, cold initial state.
///   Constant grey opacity, grey Stefan-Boltzmann Planck functions.
///   Picard (nonlinear) iteration with optional Anderson acceleration.
///
/// GPU offload:
///   All source-iteration work (assembly, LU factorisation, triangular
///   solve, convergence check) runs on the GPU.  The only CPU work in
///   the inner loop is the temperature update and Anderson mixing.
///
/// Build (requires USE_ROCM=1):
///   make USE_ROCM=1 example_trt_hotwall_vacuum_rocm
///
/// Run:
///   build/example_trt_hotwall_vacuum_rocm [nsteps]

#include "anderson.hpp"
#include "output.hpp"
#include "transport2d.hpp"
#include "trt2d.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace therefore2d;

// ---------------------------------------------------------------------------
// Problem parameters  (mirror the CPU example defaults)
// ---------------------------------------------------------------------------
constexpr int    kNx               = 40;
constexpr int    kNy               = 40;
constexpr int    kSN               = 12;       // Sn order → 8 directions
constexpr int    kNumTimeSteps     = 2;
constexpr int    kMaxNonlinearIters = 300;
constexpr int    kMaxTransportIters = 200;
constexpr double kDt               = 1.0e-12; // s
constexpr double kTransportTol     = 1.0e-10;
constexpr double kNonlinearTol     = 1.0e-4;
constexpr double kTfloor           = 1.0e-3;  // keV
constexpr double kTcold            = 1.0e-3;  // keV  (initial)
constexpr double kThot             = 1.0;     // keV  (hot-wall BC)
constexpr double kLx               = 1.0;     // cm
constexpr double kLy               = 1.0;     // cm
constexpr double kRho              = 0.2;     // g/cm^3  (Brunner foam)
constexpr double kCv               = 2.41213e14; // erg/(g keV)
constexpr double kSigma            = 10.0;    // /cm

// Anderson acceleration (same defaults as CPU example)
constexpr bool   kUseAnderson      = true;
constexpr int    kAndersonM        = 5;
constexpr double kAndersonDamp     = 0.8;

// ---------------------------------------------------------------------------
// Grey Stefan-Boltzmann helpers
// ---------------------------------------------------------------------------
inline double grey_B(double T) {
    const double t = std::max(T, kTfloor);
    return kTrtSpeedOfLight * kTrtRadiationConstant * t*t*t*t / (4.0 * M_PI);
}
inline double grey_dBdT(double T) {
    const double t = std::max(T, kTfloor);
    return kTrtSpeedOfLight * kTrtRadiationConstant * t*t*t / M_PI;
}
inline double grey_alpha(double T, double dt) {
    const double d = 4.0 * M_PI * kSigma * grey_dBdT(T);
    return d / (kRho * kCv / dt + d);
}

// ---------------------------------------------------------------------------
// Boundary and cell fill
// ---------------------------------------------------------------------------
void set_hotwall_vacuum_bc(Problem2D& p, double psi_left) {
    // All four faces default to vacuum (zero inflow); the hot wall
    // overrides the westward-facing DOFs on the west boundary.
    p.boundary.west.assign(p.ny * p.groups * p.num_dirs() * 4, 0.0);
    p.boundary.east.clear();
    p.boundary.south.clear();
    p.boundary.north.clear();

    for (int j = 0; j < p.ny; ++j)
        for (int g = 0; g < p.groups; ++g)
            for (int d = 0; d < p.num_dirs(); ++d) {
                if (p.directions[d].mu <= 0.0) continue;   // only rightward dirs
                const int off = face_offset_west_east(p, j, g, d, 0);
                for (int k = 0; k < 4; ++k)
                    p.boundary.west[off + k] = psi_left;
            }
}

// Rebuild cell cross-sections and sources from the current temperature.
// Called once per nonlinear iteration.
void fill_cells(SolverState2D& state, const std::vector<double>& Tlag) {
    Problem2D& p = state.problem;
    const double dx = p.Lx / p.nx;
    const double dy = p.Ly / p.ny;

    // Boundary: hot wall on the left, vacuum on other three sides.
    // The BC is invariant across Picard iterations; we set it here so that
    // the initial factor_cells_rocm() call uploads the correct values.
    set_hotwall_vacuum_bc(p, grey_B(kThot));

    for (int j = 0; j < p.ny; ++j)
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            Cell2D&   c    = state.cells[cell];

            const double T     = std::max(Tlag[cell], kTfloor);
            const double B     = grey_B(T);
            const double alpha = grey_alpha(T, p.time_step);

            c.x_left   = i * dx;  c.y_bottom = j * dy;
            c.dx       = dx;      c.dy       = dy;
            c.dt       = p.time_step;
            c.velocity .assign(1, kTrtSpeedOfLight);
            c.sigma_t  .assign(1, kSigma);
            c.sigma_s  .assign(1, alpha * kSigma);
            c.source   .assign(p.cell_block_size(), 0.0);

            const double q = (1.0 - alpha) * kSigma * B;
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int off = local_angle_group_offset(p, 0, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof)
                    c.source[off + dof] = q;
            }
        }
}

// ---------------------------------------------------------------------------
// Temperature update and radiation temperature diagnostic
// ---------------------------------------------------------------------------
std::vector<double> update_T(const SolverState2D& state,
                              const std::vector<double>& T_old,
                              const std::vector<double>& Tlag,
                              const std::vector<double>& phi)
{
    const Problem2D& p = state.problem;
    std::vector<double> next(p.num_cells(), kTfloor);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const double T   = std::max(Tlag[cell], kTfloor);
        const double B   = grey_B(T);
        const double dB  = grey_dBdT(T);
        double lhs = kRho * kCv / p.time_step;
        double rhs = lhs  * T_old[cell];
        lhs += 4.0 * M_PI * kSigma * dB;
        rhs += 4.0 * M_PI * kSigma * (phi[cell] - B + dB * T);
        next[cell] = std::max(kTfloor, rhs / lhs);
    }
    return next;
}

std::vector<double> make_Trad(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    std::vector<double> v(p.num_cells(), kTfloor);
    for (int c = 0; c < p.num_cells(); ++c) {
        const double phi = cell_centered_scalar_flux(state, state.flux_previous, c, 0);
        const double ur  = 4.0 * M_PI * phi / kTrtSpeedOfLight;
        if (ur > 0.0)
            v[c] = std::max(kTfloor, std::pow(ur / kTrtRadiationConstant, 0.25));
    }
    return v;
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = std::max(1e-14, std::max(std::abs(a[i]), std::abs(b[i])));
        e = std::max(e, std::abs(a[i] - b[i]) / d);
    }
    return e;
}

} // anonymous namespace

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
    using namespace therefore2d;

    int nsteps = kNumTimeSteps;
    if (argc > 1) nsteps = std::stoi(argv[1]);

    // ---- Problem setup (identical to CPU version) -------------------------
    SolverState2D state;
    Problem2D&    p = state.problem;
    p.nx = kNx;  p.ny = kNy;
    p.Lx = kLx;  p.Ly = kLy;
    p.groups             = 1;
    p.max_iters          = kMaxTransportIters;
    p.num_time_steps     = nsteps;
    p.time_step          = kDt;
    p.convergence_tol    = kTransportTol;
    p.initialize_from_previous = true;
    p.reuse_factorization      = false;
    p.directions = make_level_symmetric_quadrature_2d(kSN);
    state.cells.assign(p.num_cells(), Cell2D{});

    std::vector<double> T(p.num_cells(), kTcold);

    // Initial fill: uploads the correct boundary conditions to p.boundary
    // before factor_cells_rocm() reads them.
    fill_cells(state, T);

    // ---- GPU cache initialisation ----------------------------------------
    // factor_cells_rocm:
    //   • allocates all device buffers (flux state, chunk LU workspace)
    //   • uploads geometry, initial cross-sections, quadrature, boundaries
    // This is the ONLY call that allocates; subsequent Picard iterations use
    // refresh_cell_opacities_rocm() to re-upload the changing fields only.
    RocmLUCache cache;
    factor_cells_rocm(state, cache);

    // Initialise flux to thermal equilibrium at the cold temperature
    initialize_state(state, std::vector<double>(p.total_unknowns(), grey_B(kTcold)));

    // ---- Diagnostics / output setup --------------------------------------
    const double t_total = kDt * nsteps;
    const double D       = kTrtSpeedOfLight / (3.0 * kSigma);
    std::cout << "Grey Marshak wave (ROCm GPU)  sigma=" << kSigma
              << " /cm  mfp=" << 1.0 / kSigma << " cm\n"
              << "  nx=" << kNx << "  kSN=" << kSN
              << "  dirs=" << p.num_dirs()
              << "  cell_block=" << p.cell_block_size() << "\n"
              << "  dt=" << kDt << " s  nsteps=" << nsteps
              << "  t_total=" << t_total << " s\n"
              << "  alpha(T_hot)=" << grey_alpha(kThot, kDt)
              << "  alpha(T_cold)=" << grey_alpha(kTcold, kDt) << "\n"
              << "  diffusion front: sqrt(D*t)=" << std::sqrt(D * t_total) << " cm\n";

    const std::string outdir  = "results/example_trt_hotwall_vacuum_rocm";
    const std::string jsonpath = outdir + "/summary.json";
    std::filesystem::create_directories(outdir);
    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state),
        ParaviewSeriesConfig2D{outdir, "trt_hotwall_rocm", true});

    std::vector<TrtTimestepStats2D> hist;
    hist.reserve(nsteps);
    double time = 0.0;

    // =========================================================================
    // Time loop
    // =========================================================================
    for (int step = 0; step < nsteps; ++step) {
        const std::vector<double> T_old = T;
        std::vector<double> Tlag = T;
        TrtTimestepStats2D rec;

        // Anderson accelerator — fresh per timestep (history resets each step).
        FixedPointAccelerator acc(kAndersonM, kAndersonDamp);

        // ---- Picard / nonlinear loop ------------------------------------
        for (int nl = 0; nl < kMaxNonlinearIters; ++nl) {

            // (1) Rebuild cell cross-sections and emission source on CPU.
            fill_cells(state, Tlag);

            // (2) Push the changed sigma_t, sigma_s, source to the GPU.
            //     Geometry, velocity, quadrature, and boundary conditions
            //     are invariant and are NOT re-uploaded.
            refresh_cell_opacities_rocm(state, cache);

            // (3) Run source iteration entirely on the GPU.
            //     Internally: build d_rhs_const → Jacobi sweep (assemble +
            //     LU factor + solve per chunk) → convergence check.
            //     On return, state.flux_previous holds the converged flux.
            rec.transport_stats = run_one_timestep_rocm(state, cache);

            // (4) Extract cell-centred scalar flux (CPU, from downloaded flux).
            std::vector<double> phi(p.num_cells());
            for (int c = 0; c < p.num_cells(); ++c)
                phi[c] = cell_centered_scalar_flux(state, state.flux_previous, c, 0);

            // (5) Linearised implicit temperature update.
            auto nT = update_T(state, T_old, Tlag, phi);

            // (6) Convergence check (on the raw Picard residual).
            rec.max_temperature_change = max_rel(Tlag, nT);
            rec.nonlinear_iterations   = nl + 1;

            // (7) Apply Anderson acceleration (or plain Picard).
            if (kUseAnderson) acc.apply(Tlag, nT);
            Tlag = std::move(nT);

            if (rec.max_temperature_change < kNonlinearTol) break;
        }

        T     = Tlag;
        time += kDt;
        rec.step = step;
        rec.time = time;
        hist.push_back(rec);

        // ---- VTK output ------------------------------------------------
        std::vector<CellScalarField2D> fields;
        fields.push_back(make_cell_scalar_field("radiation_temperature",
                                                make_Trad(state)));
        fields.push_back(make_cell_scalar_field("material_temperature", T));
        writer.write_step(step, time, fields);

        std::cout << "step " << step
                  << "  nl=" << rec.nonlinear_iterations
                  << (rec.nonlinear_iterations >= kMaxNonlinearIters
                          ? "(MAX)" : "     ")
                  << "  dT=" << rec.max_temperature_change
                  << "  tr=" << rec.transport_stats.iterations
                  << "  T[0]="    << T[0]
                  << "  T[nx/4]=" << T[p.nx / 4]
                  << "  T[nx/2]=" << T[p.nx / 2]
                  << "\n";
    }

    // ---- Teardown --------------------------------------------------------
    destroy_rocm_cache(cache);

    TrtState2D dummy;
    dummy.transport       = state;
    dummy.config.dt       = kDt;
    dummy.config.num_time_steps = nsteps;
    dummy.history         = hist;
    write_trt_summary_json(jsonpath, dummy, writer.pvd_path());

    std::cout << "Wrote:\n  " << writer.pvd_path() << "\n  " << jsonpath << "\n";
    return 0;
}
