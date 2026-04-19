#include "trt2d.hpp"
#include "output.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <stdexcept>

#include <iostream>

namespace therefore2d {
namespace {

constexpr double kPi = 3.1415926535897932384626433832795;
constexpr double kSpeedOfLight = 2.99792458e10;
constexpr double kPlanckH = 6.62607015e-27;
constexpr double kBoltzmann = 1.602176634e-9;
constexpr double kPlanckPrefactor = 2.0 * kBoltzmann * kBoltzmann * kBoltzmann * kBoltzmann
                                 / (kSpeedOfLight * kSpeedOfLight * kPlanckH * kPlanckH * kPlanckH);
constexpr double kRadiationConstant = 8.0 * kPi * kPi * kPi * kPi * kPi
                                   * kBoltzmann * kBoltzmann * kBoltzmann * kBoltzmann
                                   / (15.0 * kSpeedOfLight * kSpeedOfLight * kSpeedOfLight
                                      * kPlanckH * kPlanckH * kPlanckH);

std::array<double, 8> nodes8() {
    return {-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363};
}

std::array<double, 8> weights8() {
    return {0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
            0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763};
}

double integrate_pieces(const std::vector<double>& cuts, const std::function<double(double)>& f) {
    static const auto n = nodes8();
    static const auto w = weights8();
    double total = 0.0;
    for (std::size_t k = 1; k < cuts.size(); ++k) {
        const double a = cuts[k - 1];
        const double b = cuts[k];
        if (!(b > a)) {
            continue;
        }
        const double half = 0.5 * (b - a);
        const double mid = 0.5 * (a + b);
        for (int q = 0; q < 8; ++q) {
            total += half * w[q] * f(mid + half * n[q]);
        }
    }
    return total;
}

std::vector<double> cuts_for_group(double elo, double ehi, const AnalyticOpacityParams& p) {
    std::vector<double> cuts{elo, ehi};
    auto push = [&](double x) {
        if (x > elo && x < ehi) {
            cuts.push_back(x);
        }
    };
    push(p.epsilon_min);
    push(p.epsilon_edge);
    if (p.num_lines > 0 && p.delta_w > 0.0) {
        for (int l = 0; l < p.num_lines; ++l) {
            const double center = p.epsilon_edge - (static_cast<double>(l) + 1.0) * p.delta_s;
            push(center - 5.0 * p.delta_w);
            push(center);
            push(center + 5.0 * p.delta_w);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double a, double b) {
        return std::abs(a - b) < 1.0e-13;
    }), cuts.end());
    return cuts;
}

double B_epsilon(double epsilon, double temperature) {
    const double T = std::max(temperature, 1.0e-12);
    const double x = epsilon / T;
    const double denom = std::exp(x) - 1.0;
    if (denom <= 0.0) {
        return 0.0;
    }
    return kPlanckPrefactor * epsilon * epsilon * epsilon / denom;
}

double dB_epsilon_dT(double epsilon, double temperature) {
    const double T = std::max(temperature, 1.0e-12);
    const double x = epsilon / T;
    const double ex = std::exp(x);
    const double denom = ex - 1.0;
    if (denom <= 0.0) {
        return 0.0;
    }
    return kPlanckPrefactor * epsilon * epsilon * epsilon * epsilon * ex / (T * T * denom * denom);
}

double sigma_epsilon(double epsilon,
                     double temperature,
                     double density,
                     const AnalyticOpacityParams& p) {
    const double T = std::max(temperature, 1.0e-12);
    const double ehat = std::max(p.epsilon_min, epsilon);
    const double base = p.C0 * density * density / (std::sqrt(T) * ehat * ehat * ehat)
                      * (1.0 - std::exp(-ehat / T));

    double feature = 1.0;
    if (ehat >= p.epsilon_edge) {
        feature += p.C1;
    }
    if (p.num_lines > 0 && p.delta_w > 0.0) {
        for (int l = 0; l < p.num_lines; ++l) {
            const double center = p.epsilon_edge - (static_cast<double>(l) + 1.0) * p.delta_s;
            const double z = (ehat - center) / p.delta_w;
            feature += p.C2 / static_cast<double>(p.num_lines - l) * std::exp(-0.5 * z * z);
        }
    }
    return base * feature;
}

double group_B(double elo, double ehi, double temperature) {
    const auto cuts = cuts_for_group(elo, ehi, AnalyticOpacityParams{});
    return integrate_pieces(cuts, [temperature](double e) { return B_epsilon(e, temperature); });
}

double group_dB_dT(double elo, double ehi, double temperature) {
    const auto cuts = cuts_for_group(elo, ehi, AnalyticOpacityParams{});
    return integrate_pieces(cuts, [temperature](double e) { return dB_epsilon_dT(e, temperature); });
}

double group_planck_opacity(double elo,
                            double ehi,
                            double temperature,
                            double density,
                            const AnalyticOpacityParams& p) {
    const auto cuts = cuts_for_group(elo, ehi, p);
    const double B = integrate_pieces(cuts, [temperature](double e) { return B_epsilon(e, temperature); });
    if (B <= 0.0) {
        return 0.0;
    }
    const double weighted = integrate_pieces(cuts, [temperature, density, &p](double e) {
        return sigma_epsilon(e, temperature, density, p) * B_epsilon(e, temperature);
    });
    return weighted / B;
}

int material_id_figure24a(double x, double y) {
    auto in = [](double v, double lo, double hi) { return v >= lo && v < hi; };
    if (in(x, 0.0, 1.0) && in(y, 2.0, 3.0)) return 1;
    if (in(x, 2.0, 3.0) && in(y, 2.0, 3.0)) return 1;
    if (in(x, 0.0, 1.0) && in(y, 0.0, 1.0)) return 1;
    if (in(x, 0.0, 1.0) && in(y, 1.0, 2.0)) return 3;
    if (in(x, 1.0, 2.0) && in(y, 1.0, 2.0)) return 2;
    return 0;
}

void assign_geometry(TrtState2D& state) {
    Problem2D& p = state.transport.problem;
    state.transport.cells.assign(p.num_cells(), Cell2D{});
    state.trt_cells.assign(p.num_cells(), TrtCellState2D{});

    const double dx = p.Lx / static_cast<double>(p.nx);
    const double dy = p.Ly / static_cast<double>(p.ny);
    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            const double xc = (static_cast<double>(i) + 0.5) * dx;
            const double yc = (static_cast<double>(j) + 0.5) * dy;
            const int mat = material_id_figure24a(xc, yc);

            Cell2D& c = state.transport.cells[cell];
            c.x_left = static_cast<double>(i) * dx;
            c.y_bottom = static_cast<double>(j) * dy;
            c.dx = dx;
            c.dy = dy;
            c.dt = state.config.dt;

            state.trt_cells[cell].material_id = mat;
            state.trt_cells[cell].temperature = state.materials[mat].initial_temperature;
            state.trt_cells[cell].previous_temperature = state.materials[mat].initial_temperature;
        }
    }
}

std::vector<double> planck_groups(const std::vector<double>& edges, double temperature) {
    std::vector<double> out(edges.size() - 1, 0.0);
    for (std::size_t g = 0; g + 1 < edges.size(); ++g) {
        out[g] = group_B(edges[g], edges[g + 1], temperature);
    }
    return out;
}

void set_boundaries(SolverState2D& state,
                    const std::vector<double>& left_B,
                    const std::vector<double>& cold_B) {
    Problem2D& p = state.problem;
    p.boundary.west.assign(p.ny * p.groups * p.num_dirs() * 4, 0.0);
    p.boundary.east.assign(p.ny * p.groups * p.num_dirs() * 4, 0.0);
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

void fill_transport_coefficients(TrtState2D& state, const std::vector<double>& Tlag) {
    SolverState2D& s = state.transport;
    Problem2D& p = s.problem;

    const std::vector<double> left_B = planck_groups(state.group_edges, state.config.hot_boundary_temperature);
    const std::vector<double> cold_B = planck_groups(state.group_edges, state.config.cold_boundary_temperature);
    set_boundaries(s, left_B, cold_B);

    #pragma omp parallel for
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const MaterialModelTRT& mat = state.materials[state.trt_cells[cell].material_id];
        const double T = std::max(Tlag[cell], state.config.temperature_floor);
        Cell2D& c = s.cells[cell];

        c.dt = state.config.dt;
        c.velocity.assign(p.groups, kSpeedOfLight);
        c.sigma_t.assign(p.groups, 0.0);
        c.sigma_s.assign(p.groups * p.groups, 0.0);
        c.source.assign(p.cell_block_size(), 0.0);

        std::vector<double> sigma(p.groups, 0.0);
        std::vector<double> B(p.groups, 0.0);
        std::vector<double> dB(p.groups, 0.0);
        double denom = mat.density * mat.cv / state.config.dt;

        for (int g = 0; g < p.groups; ++g) {
            sigma[g] = group_planck_opacity(state.group_edges[g], state.group_edges[g + 1], T, mat.density, mat.opacity);
            B[g] = group_B(state.group_edges[g], state.group_edges[g + 1], T);
            dB[g] = group_dB_dT(state.group_edges[g], state.group_edges[g + 1], T);
            denom += 4.0 * kPi * sigma[g] * dB[g];
        }

        double sigma_B_sum = 0.0;
        for (int g = 0; g < p.groups; ++g) sigma_B_sum += sigma[g] * B[g];

        std::vector<double> alpha(p.groups, 0.0);
        for (int g = 0; g < p.groups; ++g) {
            alpha[g] = (denom > 0.0) ? (4.0 * kPi * sigma[g] * dB[g] / denom) : 0.0;
            c.sigma_t[g] = sigma[g];
        }

        for (int g_to = 0; g_to < p.groups; ++g_to) {
            for (int g_from = 0; g_from < p.groups; ++g_from) {
                c.sigma_s[g_to * p.groups + g_from] = alpha[g_to] * sigma[g_from];
            }
        }

        for (int d = 0; d < p.num_dirs(); ++d) {
            for (int g = 0; g < p.groups; ++g) {
                const double q = sigma[g] * B[g] - alpha[g] * sigma_B_sum;
                const int off = local_angle_group_offset(p, g, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) c.source[off + dof] = q;
            }
        }
    }
}

std::vector<double> scalar_flux_groups(const TrtState2D& state) {
    const SolverState2D& s = state.transport;
    const Problem2D& p = s.problem;
    std::vector<double> flux(p.num_cells() * p.groups, 0.0);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        for (int g = 0; g < p.groups; ++g) {
            flux[cell * p.groups + g] = cell_centered_scalar_flux(s, s.flux_previous, cell, g);
        }
    }
    return flux;
}
std::vector<double> radiation_temperature_field(const TrtState2D& state) {
    const SolverState2D& s = state.transport;
    const Problem2D& p = s.problem;
    std::vector<double> values(p.num_cells(), state.config.temperature_floor);
    #pragma omp parallel for
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        double Jsum = 0.0;
        for (int g = 0; g < p.groups; ++g) {
            Jsum += cell_centered_scalar_flux(s, s.flux_previous, cell, g);
        }
        const double ur = 4.0 * kPi * Jsum / kSpeedOfLight;
        values[cell] = std::max(state.config.temperature_floor, std::pow(std::max(0.0, ur / kRadiationConstant), 0.25));
    }
    return values;
}

std::vector<double> material_temperature_field(const TrtState2D& state) {
    const Problem2D& p = state.transport.problem;
    std::vector<double> values(p.num_cells(), state.config.temperature_floor);

    for (int cell = 0; cell < p.num_cells(); ++cell) {
        values[cell] = state.trt_cells[cell].temperature;
    }
    return values;
}

std::vector<std::string> material_name_field(const TrtState2D& state) {
    const Problem2D& p = state.transport.problem;
    std::vector<std::string> names(p.num_cells());
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        names[cell] = state.materials[state.trt_cells[cell].material_id].name;
    }
    return names;
}

std::vector<double> update_temperatures(const TrtState2D& state,
                                        const std::vector<double>& Tlag,
                                        const std::vector<double>& scalar_flux) {
    const Problem2D& p = state.transport.problem;
    std::vector<double> next(Tlag.size(), state.config.temperature_floor);

    #pragma omp parallel for
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const MaterialModelTRT& mat = state.materials[state.trt_cells[cell].material_id];
        const double T = std::max(Tlag[cell], state.config.temperature_floor);
        const double Told = state.trt_cells[cell].previous_temperature;

        double lhs = mat.density * mat.cv / state.config.dt;
        double rhs = lhs * Told;
        for (int g = 0; g < p.groups; ++g) {
            const double sigma = group_planck_opacity(state.group_edges[g], state.group_edges[g + 1], T, mat.density, mat.opacity);
            const double B = group_B(state.group_edges[g], state.group_edges[g + 1], T);
            const double dB = group_dB_dT(state.group_edges[g], state.group_edges[g + 1], T);
            lhs += 4.0 * kPi * sigma * dB;
            rhs += 4.0 * kPi * sigma * (scalar_flux[cell * p.groups + g] - B + dB * T);
        }
        next[cell] = std::max(state.config.temperature_floor, rhs / lhs);
    }

    return next;
}

double max_rel_change(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double denom = std::max({1.0e-14, std::abs(a[i]), std::abs(b[i])});
        err = std::max(err, std::abs(a[i] - b[i]) / denom);
    }
    return err;
}

} // namespace

std::vector<double> make_table5_group_edges() {
    return {
        1.0e-4, 3.0e-3, 1.095445115010333e-2, 4.0e-2, 5.0e-2,
        7.825422900366437e-2, 1.224744871391589e-1, 1.916829312738817e-1,
        3.0e-1, 6.708203932499368e-1, 1.5, 3.240370349203930, 7.0,
        1.114619555925213e1, 1.774823934929885e1, 2.826076380281411e1, 4.5e1
    };
}

std::vector<MaterialModelTRT> make_trt_materials() {
    return {
        MaterialModelTRT{"foam", 0.2, 2.41213e14, 1.0e-3, {0.04, 0.3, 2.0, 4.0e2, 0.0, 0, 0.0, 0.0}},
        MaterialModelTRT{"carbon", 2.0, 2.41213e14, 1.0e-3, {0.04, 1.5, 0.77, 1.2e3, 30.0, 1, 0.01, 1.2}},
        MaterialModelTRT{"cold_iron", 6.0, 5.4273e14, 1.0e-3, {0.05, 7.0, 20.1, 1.2e3, 1.2e3, 5, 0.01, 0.2}},
        MaterialModelTRT{"hot_iron", 8.0, 5.4273e14, 5.0e-1, {0.05, 7.0, 20.1, 1.2e3, 1.2e3, 5, 0.01, 0.2}}
    };
}

TrtState2D make_figure24a_lattice_problem(const TrtConfig2D& config) {
    TrtState2D state;
    state.config = config;
    state.group_edges = make_table5_group_edges();
    state.materials = make_trt_materials();

    Problem2D& p = state.transport.problem;
    p.nx = config.nx;
    p.ny = config.ny;
    p.Lx = config.Lx;
    p.Ly = config.Ly;
    p.groups = static_cast<int>(state.group_edges.size()) - 1;
    p.max_iters = config.max_transport_iters;
    p.num_time_steps = config.num_time_steps;
    p.time_step = config.dt;
    p.convergence_tol = config.transport_tol;
    p.initialize_from_previous = true;
    p.reuse_factorization = false;
    p.directions = make_level_symmetric_quadrature_2d(config.sn_order);

    assign_geometry(state);
    return state;
}

void initialize_trt_state(TrtState2D& state) {
    const Problem2D& p = state.transport.problem;
    std::vector<double> initial_T(state.trt_cells.size(), state.config.temperature_floor);
    for (std::size_t i = 0; i < state.trt_cells.size(); ++i) initial_T[i] = state.trt_cells[i].temperature;
    fill_transport_coefficients(state, initial_T);

    std::vector<double> initial_flux(p.total_unknowns(), 0.0);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const double T = state.trt_cells[cell].temperature;
        for (int g = 0; g < p.groups; ++g) {
            const double B = group_B(state.group_edges[g], state.group_edges[g + 1], T);
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int off = global_offset(p, cell, g, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) initial_flux[off + dof] = B;
            }
        }
    }
    initialize_state(state.transport, initial_flux);
}

TrtTimestepStats2D run_one_timestep_trt_cpu(TrtState2D& state, CpuLUCache& cache, bool use_openmp) {
    std::vector<double> Tlag(state.trt_cells.size(), 0.0);
    for (std::size_t cell = 0; cell < state.trt_cells.size(); ++cell) {
        state.trt_cells[cell].previous_temperature = state.trt_cells[cell].temperature;
        Tlag[cell] = state.trt_cells[cell].temperature;
    }
    double max_change_last = 1.0;

    TrtTimestepStats2D rec;
    for (int nl = 0; nl < state.config.max_nonlinear_iters; ++nl) {
        fill_transport_coefficients(state, Tlag);
        assemble_cell_matrices(state.transport);
        cache.valid = false;
        build_constant_rhs(state.transport);
        rec.transport_stats = run_one_timestep_cpu(state.transport, cache, use_openmp);
        const std::vector<double> J = scalar_flux_groups(state);
        const std::vector<double> Tnext = update_temperatures(state, Tlag, J);
        rec.max_temperature_change = max_rel_change(Tlag, Tnext);
        rec.nonlinear_iterations = nl + 1;
        Tlag = Tnext;
        std::cout << "NL Iteration " << nl << " e " << rec.max_temperature_change << " rho " << rec.max_temperature_change/max_change_last << std::endl;
        max_change_last = rec.max_temperature_change;
        if (rec.max_temperature_change < state.config.nonlinear_tol) break;
    }

    for (std::size_t cell = 0; cell < state.trt_cells.size(); ++cell) state.trt_cells[cell].temperature = Tlag[cell];
    return rec;
}

std::vector<TrtTimestepStats2D> run_time_trt_cpu(TrtState2D& state,
                                                 CpuLUCache& cache,
                                                 bool use_openmp,
                                                 const TrtOutputFiles2D& outputs) {
    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state.transport),
        ParaviewSeriesConfig2D{outputs.output_dir, outputs.series_name, outputs.write_pvd_every_step});

    state.history.clear();
    state.history.reserve(state.config.num_time_steps);

    double time = 0.0;
    for (int step = 0; step < state.config.num_time_steps; ++step) {
        std::cout << "TRT STEP " << step << '\n';
        TrtTimestepStats2D stats = run_one_timestep_trt_cpu(state, cache, use_openmp);
        time += state.config.dt;
        stats.step = step;
        stats.time = time;
        state.history.push_back(stats);

        std::vector<CellScalarField2D> fields;
        append_fields(fields, make_angular_flux_group_dir_fields(state.transport, state.transport.flux_previous, "angular_intensity"));
        append_fields(fields, make_scalar_flux_group_fields(state.transport, state.transport.flux_previous, "scalar_flux_g"));
        fields.push_back(make_cell_scalar_field("radiation_temperature", radiation_temperature_field(state)));
        fields.push_back(make_cell_scalar_field("material_temperature", material_temperature_field(state)));
        writer.write_step(step, time, fields);

        std::cout << "  time=" << time
                  << "  nonlinear_iters=" << stats.nonlinear_iterations
                  << "  max_dT_rel=" << stats.max_temperature_change
                  << "  transport_iters=" << stats.transport_stats.iterations
                  << "  transport_error=" << stats.transport_stats.final_error
                  << '\n';
    }

    const std::filesystem::path out_path(outputs.summary_json);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    const Problem2D& p = state.transport.problem;
    std::ofstream summary(outputs.summary_json);
    if (!summary) {
        throw std::runtime_error("Could not open TRT summary JSON for writing: " + outputs.summary_json);
    }
    summary << std::setprecision(16);
    summary << "{\n";
    summary << "  \"nx\": " << p.nx << ",\n";
    summary << "  \"ny\": " << p.ny << ",\n";
    summary << "  \"groups\": " << p.groups << ",\n";
    summary << "  \"num_dirs\": " << p.num_dirs() << ",\n";
    summary << "  \"dt\": " << state.config.dt << ",\n";
    summary << "  \"num_time_steps\": " << state.config.num_time_steps << ",\n";
    summary << "  \"paraview_pvd\": \"" << writer.pvd_path() << "\",\n";
    summary << "  \"history\": [\n";
    for (std::size_t k = 0; k < state.history.size(); ++k) {
        const auto& rec = state.history[k];
        summary << "    {\"step\": " << rec.step
                << ", \"time\": " << rec.time
                << ", \"nonlinear_iterations\": " << rec.nonlinear_iterations
                << ", \"max_temperature_change\": " << rec.max_temperature_change
                << ", \"transport_iterations\": " << rec.transport_stats.iterations
                << ", \"transport_final_error\": " << rec.transport_stats.final_error << "}";
        if (k + 1 != state.history.size()) {
            summary << ',';
        }
        summary << '\n';
    }
    summary << "  ]\n";
    summary << "}\n";

    std::cout << "Wrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << outputs.summary_json << '\n';
    return state.history;
}

} // namespace therefore2d
