#include "trt2d.hpp"
#include "anderson.hpp"
#include "output.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace therefore2d {
namespace {

// ---------------------------------------------------------------------------
// Physical constants (same values as in the header, kept here for brevity)
// ---------------------------------------------------------------------------

constexpr double kPi = 3.1415926535897932384626433832795;

// Planck function prefactor: 2 k^4 / (c^2 h^3)  [erg/(cm^2 s sr keV^4 / keV^3)]
constexpr double kPlanckPrefactor =
    2.0 * kTrtBoltzmann * kTrtBoltzmann * kTrtBoltzmann * kTrtBoltzmann
    / (kTrtSpeedOfLight * kTrtSpeedOfLight
       * kTrtPlanckH * kTrtPlanckH * kTrtPlanckH);

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature (8-point) for energy integration
// ---------------------------------------------------------------------------

std::array<double, 8> gauss8_nodes() {
    return {-0.9602898564975363, -0.7966664774136267,
            -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498,  0.5255324099163290,
             0.7966664774136267,  0.9602898564975363};
}

std::array<double, 8> gauss8_weights() {
    return {0.1012285362903763, 0.2223810344533745,
            0.3137066458778873, 0.3626837833783620,
            0.3626837833783620, 0.3137066458778873,
            0.2223810344533745, 0.1012285362903763};
}

/// Integrate f over a piecewise-smooth interval by subdividing at the
/// feature points given in `cuts` and applying 8-point Gauss-Legendre
/// on each sub-interval.
double integrate_pieces(const std::vector<double>& cuts,
                        const std::function<double(double)>& f) {
    static const auto n = gauss8_nodes();
    static const auto w = gauss8_weights();
    double total = 0.0;
    for (std::size_t k = 1; k < cuts.size(); ++k) {
        const double a = cuts[k - 1];
        const double b = cuts[k];
        if (!(b > a)) continue;
        const double half = 0.5 * (b - a);
        const double mid  = 0.5 * (a + b);
        for (int q = 0; q < 8; ++q)
            total += half * w[q] * f(mid + half * n[q]);
    }
    return total;
}

/// Build a sorted list of integration sub-interval boundaries for group
/// [elo, ehi] by inserting the opacity feature points (epsilon_min,
/// epsilon_edge, and the Gaussian line centres).
std::vector<double> cuts_for_group(double elo, double ehi,
                                   const AnalyticOpacityParams& p) {
    std::vector<double> cuts{elo, ehi};
    auto push = [&](double x) {
        if (x > elo && x < ehi) cuts.push_back(x);
    };
    push(p.epsilon_min);
    push(p.epsilon_edge);
    if (p.num_lines > 0 && p.delta_w > 0.0) {
        for (int l = 0; l < p.num_lines; ++l) {
            const double center = p.epsilon_edge
                                 - (static_cast<double>(l) + 1.0) * p.delta_s;
            push(center - 5.0 * p.delta_w);
            push(center);
            push(center + 5.0 * p.delta_w);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](double a, double b){ return std::abs(a-b) < 1.0e-13; }),
               cuts.end());
    return cuts;
}

// ---------------------------------------------------------------------------
// Spectral functions (private implementation)
// ---------------------------------------------------------------------------

double B_epsilon(double epsilon, double temperature) {
    const double T = std::max(temperature, 1.0e-12);
    const double x = epsilon / T;
    // For very large x, exp(x)→∞; numerator ε³/∞ = 0 gracefully.
    // For very small x, exp(x)-1 underflows toward 0; guard to avoid /0.
    if (x > 700.0) return 0.0;
    const double denom = std::exp(x) - 1.0;
    if (denom <= 0.0) return 0.0;
    return kPlanckPrefactor * epsilon * epsilon * epsilon / denom;
}

double dB_epsilon_dT(double epsilon, double temperature) {
    const double T  = std::max(temperature, 1.0e-12);
    const double x  = epsilon / T;
    // For large x: dB/dT ≈ (ε/T)² * B → 0.  Guard BEFORE computing exp to
    // avoid inf/inf² = NaN when exp overflows.
    if (x > 700.0) return 0.0;
    const double ex = std::exp(x);
    const double denom = ex - 1.0;
    if (denom <= 0.0) return 0.0;
    return kPlanckPrefactor * epsilon * epsilon * epsilon * epsilon
           * ex / (T * T * denom * denom);
}

double sigma_epsilon(double epsilon, double temperature, double density,
                     const AnalyticOpacityParams& p) {
    const double T    = std::max(temperature, 1.0e-12);
    const double ehat = std::max(p.epsilon_min, epsilon);
    const double base = p.C0 * density * density
                       / (std::sqrt(T) * ehat * ehat * ehat)
                       * (1.0 - std::exp(-ehat / T));

    double feature = 1.0;
    if (ehat >= p.epsilon_edge) feature += p.C1;
    if (p.num_lines > 0 && p.delta_w > 0.0) {
        for (int l = 0; l < p.num_lines; ++l) {
            const double center = p.epsilon_edge
                                 - (static_cast<double>(l) + 1.0) * p.delta_s;
            const double z = (ehat - center) / p.delta_w;
            feature += p.C2 / static_cast<double>(p.num_lines - l)
                      * std::exp(-0.5 * z * z);
        }
    }
    return base * feature;
}

// ---------------------------------------------------------------------------
// Boundary condition helper
// ---------------------------------------------------------------------------

/// Set all four boundaries: left = incoming Planck at hot T, all others =
/// incoming Planck at cold T (vacuum = leave empty for those faces).
/// The problem per Brunner (2023) Sec. 5.1: left = hot, top/right = cold,
/// bottom = cold (reflecting in RZ; we use cold Planck for 2-D Cartesian).
void set_trt_boundaries(SolverState2D& state,
                        const std::vector<double>& left_B,
                        const std::vector<double>& cold_B) {
    Problem2D& p = state.problem;
    p.boundary.west.assign (p.ny * p.groups * p.num_dirs() * 4, 0.0);
    p.boundary.east.assign (p.ny * p.groups * p.num_dirs() * 4, 0.0);
    p.boundary.south.assign(p.nx * p.groups * p.num_dirs() * 4, 0.0);
    p.boundary.north.assign(p.nx * p.groups * p.num_dirs() * 4, 0.0);

    for (int j = 0; j < p.ny; ++j) {
        for (int g = 0; g < p.groups; ++g) {
            for (int d = 0; d < p.num_dirs(); ++d) {
                if (p.directions[d].mu > 0.0) {
                    const int off = face_offset_west_east(p, j, g, d, 0);
                    for (int k = 0; k < 4; ++k) p.boundary.west[off + k] = left_B[g];
                }
                if (p.directions[d].mu < 0.0) {
                    const int off = face_offset_west_east(p, j, g, d, 0);
                    for (int k = 0; k < 4; ++k) p.boundary.east[off + k] = cold_B[g];
                }
            }
        }
    }
    for (int i = 0; i < p.nx; ++i) {
        for (int g = 0; g < p.groups; ++g) {
            for (int d = 0; d < p.num_dirs(); ++d) {
                if (p.directions[d].eta > 0.0) {
                    const int off = face_offset_south_north(p, i, g, d, 0);
                    for (int k = 0; k < 4; ++k) p.boundary.south[off + k] = cold_B[g];
                }
                if (p.directions[d].eta < 0.0) {
                    const int off = face_offset_south_north(p, i, g, d, 0);
                    for (int k = 0; k < 4; ++k) p.boundary.north[off + k] = cold_B[g];
                }
            }
        }
    }
}

/// Compute Planck integrals for each group.
std::vector<double> planck_groups_vec(const std::vector<double>& edges,
                                      double temperature) {
    std::vector<double> out(edges.size() - 1, 0.0);
    for (std::size_t g = 0; g + 1 < edges.size(); ++g)
        out[g] = planck_B_group(edges[g], edges[g + 1], temperature);
    return out;
}

// ---------------------------------------------------------------------------
// Group scalar flux helper
// ---------------------------------------------------------------------------

/// Returns a flat array of size num_cells * groups with the cell-centred
/// scalar flux (angle-average = sum_d w_d * psi_d) for each (cell, group).
std::vector<double> scalar_flux_by_group(const TrtState2D& state) {
    const SolverState2D& s = state.transport;
    const Problem2D&     p = s.problem;
    std::vector<double> flux(p.num_cells() * p.groups, 0.0);
    for (int cell = 0; cell < p.num_cells(); ++cell)
        for (int g = 0; g < p.groups; ++g)
            flux[cell * p.groups + g] =
                cell_centered_scalar_flux(s, s.flux_previous, cell, g);
    return flux;
}

// ---------------------------------------------------------------------------
// Temperature update (Newton linearisation of material energy equation)
// ---------------------------------------------------------------------------
// Discretised energy equation (backward Euler):
//   rho*cv/dt * (T^{n+1} - T^n) = sum_g sigma_g * (phi_g - 4*pi*B_g)
// Linearising around T_lag:
//   (rho*cv/dt + sum_g sigma_g * 4*pi * dB_g/dT) * T^{n+1}
//     = rho*cv/dt * T^n + sum_g sigma_g * (phi_g - B_g + (dB_g/dT)*T_lag)
// References: Brunner (2023) Eq. (5), (29).

std::vector<double> update_temperatures(const TrtState2D& state,
                                        const std::vector<double>& Tlag,
                                        const std::vector<double>& scalar_flux) {
    const Problem2D& p = state.transport.problem;
    std::vector<double> next(Tlag.size(), state.config.temperature_floor);

    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const MaterialModelTRT& mat = state.materials[state.trt_cells[cell].material_id];
        const double T    = std::max(Tlag[cell], state.config.temperature_floor);
        const double Told = state.trt_cells[cell].previous_temperature;

        double lhs = mat.density * mat.cv / state.config.dt;
        double rhs = lhs * Told;

        for (int g = 0; g < p.groups; ++g) {
            const double sigma = planck_opacity_group(
                state.group_edges[g], state.group_edges[g + 1], T, mat.density, mat.opacity);
            const double B   = planck_B_group  (state.group_edges[g], state.group_edges[g + 1], T);
            const double dB  = planck_dB_dT_group(state.group_edges[g], state.group_edges[g + 1], T);
            // Note: scalar_flux stores angle-averaged psi; physics requires
            // phi = 4*pi * <psi>.  The factor 4*pi is explicit here.
            lhs += 4.0 * kPi * sigma * dB;
            rhs += 4.0 * kPi * sigma * (scalar_flux[cell * p.groups + g] - B + dB * T);
        }
        next[cell] = std::max(state.config.temperature_floor, rhs / lhs);
    }
    return next;
}

/// Max relative change between two temperature iterates.
double max_rel_change(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double denom = std::max({1.0e-14, std::abs(a[i]), std::abs(b[i])});
        err = std::max(err, std::abs(a[i] - b[i]) / denom);
    }
    return err;
}

} // namespace

// ---------------------------------------------------------------------------
// Public: group-integrated Planck functions and opacities
// ---------------------------------------------------------------------------

/// Build integration cuts that also include temperature-scale breakpoints so
/// the Gauss quadrature resolves the Planck peak (at ~2.82*T) regardless of
/// how wide the group is relative to T.  Without these, a single 8-point
/// Gauss interval over [1e-4, 45] keV places its smallest point at ~0.9 keV,
/// completely missing the peak at 2.82e-3 keV when T = 1e-3 keV.
static std::vector<double> thermal_cuts(double elo, double ehi,
                                        double temperature,
                                        const AnalyticOpacityParams& params) {
    auto cuts = cuts_for_group(elo, ehi, params);
    const double T = std::max(temperature, 1.0e-12);
    // Add breakpoints bracketing the Planck peak (epsilon_peak = 2.82*T).
    // Three decades of refinement around T ensures the peak is captured for
    // any ratio of group width to temperature.
    for (double scale : {0.1, 0.5, 1.0, 2.82, 5.0, 10.0, 30.0}) {
        const double x = scale * T;
        if (x > elo && x < ehi) cuts.push_back(x);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](double a, double b){ return std::abs(a-b) < 1.0e-13 * b; }),
               cuts.end());
    return cuts;
}

double planck_B_group(double elo, double ehi, double temperature) {
    const auto cuts = thermal_cuts(elo, ehi, temperature, AnalyticOpacityParams{});
    return integrate_pieces(cuts, [temperature](double e) {
        return B_epsilon(e, temperature);
    });
}

double planck_dB_dT_group(double elo, double ehi, double temperature) {
    const auto cuts = thermal_cuts(elo, ehi, temperature, AnalyticOpacityParams{});
    return integrate_pieces(cuts, [temperature](double e) {
        return dB_epsilon_dT(e, temperature);
    });
}

double planck_opacity_group(double elo, double ehi,
                            double temperature, double density,
                            const AnalyticOpacityParams& params) {
    const auto cuts = thermal_cuts(elo, ehi, temperature, params);
    const double B = integrate_pieces(cuts, [temperature](double e) {
        return B_epsilon(e, temperature);
    });
    if (B <= 0.0) return 0.0;
    const double weighted = integrate_pieces(cuts,
        [temperature, density, &params](double e) {
            return sigma_epsilon(e, temperature, density, params) * B_epsilon(e, temperature);
        });
    return weighted / B;
}

// ---------------------------------------------------------------------------
// Public: fill_transport_coefficients
// ---------------------------------------------------------------------------
// Implements the linearised multigroup TRT transport coefficients
// (Brunner 2023, Eqs. 28-30) for the current Picard iterate temperature Tlag.
//
// sigma_t[g]        = sigma_P,g(T_lag)
// sigma_s[g_to, g_from] = alpha[g_to] * sigma_P,g_from(T_lag)
// source[g,d,dof]   = sigma_P,g * B_g(T_lag)
//                       - alpha[g] * sum_{g'} sigma_P,g' * B_g'(T_lag)
//
// where alpha[g] = 4*pi * sigma_P,g * (dB_g/dT)
//                 / (rho*cv/dt  +  4*pi * sum_{g''} sigma_P,g'' * dB_g''/dT)

void fill_transport_coefficients(TrtState2D& state,
                                 const std::vector<double>& Tlag,
                                 bool use_openmp) {
    SolverState2D& s = state.transport;
    Problem2D&     p = s.problem;

    const std::vector<double> left_B =
        planck_groups_vec(state.group_edges, state.config.hot_boundary_temperature);
    const std::vector<double> cold_B =
        planck_groups_vec(state.group_edges, state.config.cold_boundary_temperature);
    set_trt_boundaries(s, left_B, cold_B);

#ifndef _OPENMP
    (void)use_openmp;
#endif

#ifdef _OPENMP
    #pragma omp parallel for if(use_openmp)
#endif
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const MaterialModelTRT& mat = state.materials[state.trt_cells[cell].material_id];
        const double T = std::max(Tlag[cell], state.config.temperature_floor);
        Cell2D& c      = s.cells[cell];

        c.dt = state.config.dt;
        c.velocity.assign(p.groups, kTrtSpeedOfLight);
        c.sigma_t.assign(p.groups, 0.0);
        c.sigma_s.assign(p.groups * p.groups, 0.0);
        c.source.assign(p.cell_block_size(), 0.0);

        std::vector<double> sigma(p.groups, 0.0);
        std::vector<double> B    (p.groups, 0.0);
        std::vector<double> dB   (p.groups, 0.0);

        // Denominator for alpha: rho*cv/dt + 4*pi * sum_g sigma_g * dB_g/dT
        double denom = mat.density * mat.cv / state.config.dt;
        for (int g = 0; g < p.groups; ++g) {
            sigma[g] = planck_opacity_group(state.group_edges[g], state.group_edges[g + 1],
                                            T, mat.density, mat.opacity);
            B    [g] = planck_B_group      (state.group_edges[g], state.group_edges[g + 1], T);
            dB   [g] = planck_dB_dT_group  (state.group_edges[g], state.group_edges[g + 1], T);
            denom += 4.0 * kPi * sigma[g] * dB[g];
        }

        // Weighted emission sum: sum_g sigma_g * B_g
        double sigma_B_sum = 0.0;
        for (int g = 0; g < p.groups; ++g) sigma_B_sum += sigma[g] * B[g];

        // alpha[g] = 4*pi * sigma[g] * dB[g] / denom
        std::vector<double> alpha(p.groups, 0.0);
        for (int g = 0; g < p.groups; ++g) {
            alpha[g]    = (denom > 0.0) ? (4.0 * kPi * sigma[g] * dB[g] / denom) : 0.0;
            c.sigma_t[g] = sigma[g];
        }

        // Effective group-to-group scattering (Brunner Eq. 28)
        for (int g_to = 0; g_to < p.groups; ++g_to)
            for (int g_from = 0; g_from < p.groups; ++g_from)
                c.sigma_s[g_to * p.groups + g_from] = alpha[g_to] * sigma[g_from];

        // Net emission source per direction (isotropic → same for all d)
        for (int d = 0; d < p.num_dirs(); ++d) {
            for (int g = 0; g < p.groups; ++g) {
                // Brunner Eq. (30), tau*psi^{n-1} term handled by build_constant_rhs
                const double q = sigma[g] * B[g] - alpha[g] * sigma_B_sum;
                const int off  = local_angle_group_offset(p, g, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof)
                    c.source[off + dof] = q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public: post-processing fields
// ---------------------------------------------------------------------------

std::vector<double> radiation_temperature_field(const TrtState2D& state) {
    const SolverState2D& s = state.transport;
    const Problem2D&     p = s.problem;
    std::vector<double> values(p.num_cells(), state.config.temperature_floor);

    for (int cell = 0; cell < p.num_cells(); ++cell) {
        double Jsum = 0.0;
        for (int g = 0; g < p.groups; ++g)
            Jsum += cell_centered_scalar_flux(s, s.flux_previous, cell, g);
        // phi = 4*pi * Jsum;  ur = phi / c = 4*pi * Jsum / c
        const double ur = 4.0 * kPi * Jsum / kTrtSpeedOfLight;
        values[cell] = std::max(state.config.temperature_floor,
            std::pow(std::max(0.0, ur / kTrtRadiationConstant), 0.25));
    }
    return values;
}

std::vector<double> material_temperature_field(const TrtState2D& state) {
    const Problem2D& p = state.transport.problem;
    std::vector<double> values(p.num_cells(), state.config.temperature_floor);
    for (int cell = 0; cell < p.num_cells(); ++cell)
        values[cell] = state.trt_cells[cell].temperature;
    return values;
}

// ---------------------------------------------------------------------------
// Public: initialiser and time-stepping
// ---------------------------------------------------------------------------

void initialize_trt_state(TrtState2D& state) {
    const Problem2D& p = state.transport.problem;

    // Set up transport coefficients with the initial temperatures.
    std::vector<double> initial_T(state.trt_cells.size());
    for (std::size_t i = 0; i < state.trt_cells.size(); ++i)
        initial_T[i] = state.trt_cells[i].temperature;
    fill_transport_coefficients(state, initial_T);

    // Initial angular flux = B_g(T) at each cell (equilibrium).
    std::vector<double> initial_flux(p.total_unknowns(), 0.0);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const double T = state.trt_cells[cell].temperature;
        for (int g = 0; g < p.groups; ++g) {
            const double B = planck_B_group(state.group_edges[g],
                                            state.group_edges[g + 1], T);
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int off = global_offset(p, cell, g, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof)
                    initial_flux[off + dof] = B;
            }
        }
    }
    initialize_state(state.transport, initial_flux);
}

TrtTimestepStats2D run_one_timestep_trt_cpu(TrtState2D& state,
                                            CpuLUCache& cache,
                                            bool use_openmp) {
    // Save previous-step temperatures.
    std::vector<double> Tlag(state.trt_cells.size());
    for (std::size_t cell = 0; cell < state.trt_cells.size(); ++cell) {
        state.trt_cells[cell].previous_temperature = state.trt_cells[cell].temperature;
        Tlag[cell] = state.trt_cells[cell].temperature;
    }

    TrtTimestepStats2D rec;
    double max_change_prev = 1.0;

    // Anderson mixing accelerator — created fresh each timestep so history
    // from one Picard loop does not pollute the next.  Memory for the history
    // vectors is allocated lazily on the first apply() call.
    FixedPointAccelerator acc(state.config.anderson_m,
                              state.config.anderson_damping,
                              state.config.anderson_regularization);

    for (int nl = 0; nl < state.config.max_nonlinear_iters; ++nl) {
        fill_transport_coefficients(state, Tlag, use_openmp);
        assemble_cell_matrices(state.transport);
        cache.valid = false;  // coefficients changed; must re-factor
        build_constant_rhs(state.transport);
        rec.transport_stats = run_one_timestep_cpu(state.transport, cache, use_openmp);

        const std::vector<double> J = scalar_flux_by_group(state);
        std::vector<double> Tnext   = update_temperatures(state, Tlag, J);

        // Convergence check on the RAW (pre-acceleration) residual so the
        // criterion reflects true energy-equation error, not accelerated step size.
        rec.max_temperature_change = max_rel_change(Tlag, Tnext);
        rec.nonlinear_iterations   = nl + 1;

        const double rho_nl = (max_change_prev > 0.0)
                             ? (rec.max_temperature_change / max_change_prev)
                             : 0.0;
        std::cout << "  NL " << nl
                  << "  dT=" << rec.max_temperature_change
                  << "  rho=" << rho_nl;
        if (state.config.use_anderson_acceleration)
            std::cout << "  [Anderson k=" << acc.history_size() + 1 << "]";
        std::cout << '\n';
        max_change_prev = rec.max_temperature_change;

        // Apply Anderson mixing to get the next iterate fed into the transport.
        // Tnext is modified in place; the raw convergence residual was already
        // recorded above so the tolerance check is unaffected.
        if (state.config.use_anderson_acceleration)
            acc.apply(Tlag, Tnext);

        Tlag = std::move(Tnext);
        if (rec.max_temperature_change < state.config.nonlinear_tol) break;
    }

    for (std::size_t cell = 0; cell < state.trt_cells.size(); ++cell)
        state.trt_cells[cell].temperature = Tlag[cell];
    return rec;
}

std::vector<TrtTimestepStats2D> run_time_trt_cpu(TrtState2D& state,
                                                 CpuLUCache& cache,
                                                 bool use_openmp,
                                                 const TrtOutputFiles2D& outputs) {
    std::filesystem::create_directories(outputs.output_dir);

    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state.transport),
        ParaviewSeriesConfig2D{outputs.output_dir, outputs.series_name,
                               outputs.write_pvd_every_step});

    state.history.clear();
    state.history.reserve(state.config.num_time_steps);

    double time = 0.0;
    for (int step = 0; step < state.config.num_time_steps; ++step) {
        std::cout << "TRT step " << step << '\n';
        TrtTimestepStats2D stats = run_one_timestep_trt_cpu(state, cache, use_openmp);
        time += state.config.dt;
        stats.step = step;
        stats.time = time;
        state.history.push_back(stats);

        std::vector<CellScalarField2D> fields;
        // Always write temperature fields.
        fields.push_back(make_cell_scalar_field("radiation_temperature",
                                                radiation_temperature_field(state)));
        fields.push_back(make_cell_scalar_field("material_temperature",
                                                material_temperature_field(state)));
        // Optionally write flux fields (large: one field per group per direction).
        if (outputs.save_flux) {
            append_fields(fields,
                make_angular_flux_group_dir_fields(state.transport,
                                                   state.transport.flux_previous,
                                                   "angular_intensity"));
            append_fields(fields,
                make_scalar_flux_group_fields(state.transport,
                                              state.transport.flux_previous,
                                              "scalar_flux_g"));
        }
        writer.write_step(step, time, fields);

        std::cout << "  time=" << time
                  << "  nl_iters=" << stats.nonlinear_iterations
                  << "  max_dT=" << stats.max_temperature_change
                  << "  transport_iters=" << stats.transport_stats.iterations
                  << "  transport_err=" << stats.transport_stats.final_error << '\n';
    }

    write_trt_summary_json(outputs.summary_json, state, writer.pvd_path());

    std::cout << "Wrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << outputs.summary_json << '\n';
    return state.history;
}

} // namespace therefore2d
