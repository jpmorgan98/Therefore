#ifndef THEREFORE_TRT2D_HPP
#define THEREFORE_TRT2D_HPP

#include "transport2d.hpp"

#include <string>
#include <vector>

namespace therefore2d {

// ---------------------------------------------------------------------------
// Physical constants (CGS-keV)
// ---------------------------------------------------------------------------
// Speed of light:            c  = 2.99792458e10 cm/s
// Planck constant:           h  = 6.62607015e-27 erg s
// Boltzmann constant:        k  = 1.602176634e-9 erg/keV
// Radiation constant:        a  = 8 pi^5 k^4 / (15 c^3 h^3)  erg/(cm^3 keV^4)
//   -> also a = 4 sigma_SB / c

constexpr double kTrtSpeedOfLight = 2.99792458e10;

constexpr double kTrtBoltzmann    = 1.602176634e-9;

constexpr double kTrtPlanckH      = 6.62607015e-27;

// a = 8 pi^5 k^4 / (15 c^3 h^3)
// Computed at compile time using the constants above.
// Numerically: a ≈ 1.3720169264801069e14  erg/(cm^3 keV^4)
constexpr double kTrtRadiationConstant =
    8.0 * 3.14159265358979323846 * 3.14159265358979323846 * 3.14159265358979323846
        * 3.14159265358979323846 * 3.14159265358979323846
    * kTrtBoltzmann * kTrtBoltzmann * kTrtBoltzmann * kTrtBoltzmann
    / (15.0 * kTrtSpeedOfLight * kTrtSpeedOfLight * kTrtSpeedOfLight
       * kTrtPlanckH * kTrtPlanckH * kTrtPlanckH);

// ---------------------------------------------------------------------------
// Material / opacity model
// ---------------------------------------------------------------------------

/// Analytic frequency-dependent opacity parameters following Brunner (2023),
/// Eq. (24): sigma = C0*rho^2 / (sqrt(T) * ehat^3) * (1-exp(-ehat/T))
///                  * [1 + C1*H(ehat - epsilon_edge)
///                       + sum_l C2/(Nl-l) * gauss(ehat, center_l, delta_w)]
struct AnalyticOpacityParams {
    double epsilon_min  = 0.0;  ///< Low-frequency cutoff, ehat = max(epsilon_min, epsilon)
    double epsilon_edge = 0.0;  ///< Position of the shell edge
    double C0           = 0.0;  ///< Overall opacity scale  [cm^5 keV^3.5 / g^2]
    double C1           = 0.0;  ///< Edge jump amplitude
    double C2           = 0.0;  ///< Line amplitude
    int    num_lines    = 0;    ///< Number of Gaussian line features
    double delta_w      = 0.0;  ///< Line width [keV]
    double delta_s      = 0.0;  ///< Line separation [keV]
};

/// Complete material model: thermodynamic + opacity.
struct MaterialModelTRT {
    std::string           name;
    double                density             = 0.0;  ///< g/cm^3
    double                cv                  = 0.0;  ///< specific heat  erg/(g keV)
    double                initial_temperature = 0.0;  ///< keV
    AnalyticOpacityParams opacity;
};

// ---------------------------------------------------------------------------
// Group-averaged Planck integrals and opacities (public API)
// ---------------------------------------------------------------------------

/// Integral of B(epsilon, T) over [elo, ehi]  [erg/(cm^2 s sr)]
double planck_B_group(double elo, double ehi, double temperature);

/// Integral of dB/dT(epsilon, T) over [elo, ehi]  [erg/(cm^2 s sr keV)]
double planck_dB_dT_group(double elo, double ehi, double temperature);

/// Planck-weighted group-averaged opacity  [1/cm]
/// sigma_P,g = integral(sigma * B de) / integral(B de)
double planck_opacity_group(double elo, double ehi,
                            double temperature, double density,
                            const AnalyticOpacityParams& params);

// ---------------------------------------------------------------------------
// TRT solver state
// ---------------------------------------------------------------------------

struct TrtCellState2D {
    int    material_id           = -1;
    double temperature           = 0.0;
    double previous_temperature  = 0.0;
};

struct TrtConfig2D {
    int    nx                    = 30;
    int    ny                    = 30;
    int    sn_order              = 4;
    int    num_time_steps        = 40;
    int    max_nonlinear_iters   = 100;
    int    max_transport_iters   = 3000;
    double dt                    = 1.0e-11;
    double transport_tol         = 1.0e-10;
    double nonlinear_tol         = 1.0e-6;
    double cold_boundary_temperature = 1.0e-3;
    double hot_boundary_temperature  = 1.0;
    double temperature_floor     = 1.0e-3;
    double Lx                    = 3.0;
    double Ly                    = 3.0;

    // ---- Anderson mixing / DIIS acceleration for the outer Picard loop ---
    // When enabled, applies Anderson mixing to the temperature vector after
    // each nonlinear iteration.  Maintains anderson_m past (T, residual) pairs
    // and computes an optimal linear combination minimising the residual norm.
    //
    // Convergence is checked on the RAW (pre-acceleration) residual so the
    // tolerance criterion stays physically meaningful.  Typical speedup for
    // alpha(T) ~ 0.7-0.9: 100-200 Picard iters reduced to ~10-30.
    //
    // Empirically best defaults (grey hotwall, alpha~0.77):
    //   m=3-5, damping=0.8, regularization=1e-12
    // Larger m (≥7) often degrades because residuals become linearly dependent
    // and DIIS weights blow up, triggering the safety fallback.
    bool   use_anderson_acceleration = false;
    int    anderson_m               = 5;       ///< History window (3-5 recommended).
    double anderson_damping         = 0.8;     ///< 0.8 empirically beats 1.0 (pure Anderson).
    double anderson_regularization  = 1.0e-12; ///< Gram matrix diagonal shift.
};

struct TrtTimestepStats2D {
    int            step                   = 0;
    double         time                   = 0.0;
    int            nonlinear_iterations   = 0;
    double         max_temperature_change = 0.0;
    IterationStats transport_stats;
};

struct TrtOutputFiles2D {
    std::string output_dir   = "results/trt";
    std::string series_name  = "trt";
    std::string summary_json = "results/trt_run_summary.json";
    bool write_pvd_every_step = true;
    /// If true, also write scalar and angular flux to each VTK step.
    /// Default is false because TRT produces many groups and many time steps;
    /// radiation and material temperatures are usually sufficient for diagnostics.
    bool save_flux = false;
};

struct TrtState2D {
    TrtConfig2D                    config;
    SolverState2D                  transport;
    std::vector<MaterialModelTRT>  materials;
    std::vector<TrtCellState2D>    trt_cells;
    std::vector<double>            group_edges;
    std::vector<TrtTimestepStats2D> history;
};

// ---------------------------------------------------------------------------
// Transport-coefficient assembly (called inside the nonlinear loop)
// ---------------------------------------------------------------------------

/// Populate Cell2D velocity/sigma_t/sigma_s/source and boundary conditions
/// for the current nonlinear temperature iterate Tlag, following the
/// linearisation of Brunner (2023), Eqs. (28)-(30).
void fill_transport_coefficients(TrtState2D& state,
                                 const std::vector<double>& Tlag,
                                 bool use_openmp = false);

// ---------------------------------------------------------------------------
// Post-processing fields
// ---------------------------------------------------------------------------

/// Radiation energy density temperature  Tr = (ur/a)^{1/4}  [keV]
std::vector<double> radiation_temperature_field(const TrtState2D& state);

/// Material temperature field  [keV]
std::vector<double> material_temperature_field(const TrtState2D& state);

// ---------------------------------------------------------------------------
// Initializer and time-stepping
// ---------------------------------------------------------------------------

/// Set initial transport state consistent with the initial temperatures stored
/// in state.trt_cells (psi_g = B_g(T) at each cell).
void initialize_trt_state(TrtState2D& state);

/// Advance one time step with nonlinear (Picard) iteration.
TrtTimestepStats2D run_one_timestep_trt_cpu(TrtState2D& state,
                                            CpuLUCache& cache,
                                            bool use_openmp);

/// Run all time steps, writing VTK output and a JSON summary.
std::vector<TrtTimestepStats2D> run_time_trt_cpu(
    TrtState2D& state,
    CpuLUCache& cache,
    bool use_openmp,
    const TrtOutputFiles2D& outputs = TrtOutputFiles2D{});

} // namespace therefore2d

#endif // THEREFORE_TRT2D_HPP
