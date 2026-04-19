#include "output.hpp"
#include "transport2d.hpp"
#include "trt2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace therefore2d;

constexpr int kNx = 40;
constexpr int kNy = 40;
constexpr int kSN = 8;
constexpr int kNumTimeSteps = 40;
constexpr int kMaxNonlinearIters = 50;
constexpr int kMaxTransportIters = 3000;
constexpr double kDt = 1.0e-10;
constexpr double kTransportTol = 1.0e-10;
constexpr double kNonlinearTol = 1.0e-6;
constexpr double kTemperatureFloor = 1.0e-3;
constexpr double kInitialTemperature = 1.0e-3;
constexpr double kHotBoundaryTemperature = 1.0;
constexpr double kLx = 1.0;
constexpr double kLy = 1.0;

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

struct StepStats {
    int step = 0;
    double time = 0.0;
    int nonlinear_iterations = 0;
    double max_temperature_change = 0.0;
    IterationStats transport_stats;
};

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

std::vector<double> scalar_flux_groups(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    std::vector<double> flux(p.num_cells() * p.groups, 0.0);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        for (int g = 0; g < p.groups; ++g) {
            flux[cell * p.groups + g] = cell_centered_scalar_flux(state, state.flux_previous, cell, g);
        }
    }
    return flux;
}

std::vector<double> radiation_temperature_field(const SolverState2D& state) {
    const Problem2D& p = state.problem;
    std::vector<double> values(p.num_cells(), kTemperatureFloor);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        double Jsum = 0.0;
        for (int g = 0; g < p.groups; ++g) {
            Jsum += cell_centered_scalar_flux(state, state.flux_previous, cell, g);
        }
        const double ur = 4.0 * kPi * Jsum / kSpeedOfLight;
        values[cell] = std::max(kTemperatureFloor, std::pow(std::max(0.0, ur / kRadiationConstant), 0.25));
    }
    return values;
}

std::vector<double> material_temperature_field(const std::vector<double>& temperature) {
    return temperature;
}

void set_hotwall_vacuum_boundaries(Problem2D& problem, double psi_left) {
    problem.boundary.west.assign(problem.ny * problem.groups * problem.num_dirs() * 4, 0.0);
    problem.boundary.east.clear();
    problem.boundary.south.clear();
    problem.boundary.north.clear();

    for (int j = 0; j < problem.ny; ++j) {
        for (int g = 0; g < problem.groups; ++g) {
            for (int d = 0; d < problem.num_dirs(); ++d) {
                if (problem.directions[d].mu <= 0.0) {
                    continue;
                }
                const int off = face_offset_west_east(problem, j, g, d, 0);
                for (int k = 0; k < 4; ++k) {
                    problem.boundary.west[off + k] = psi_left;
                }
            }
        }
    }
}

void fill_cells_and_coefficients(SolverState2D& state,
                                 const std::vector<double>& group_edges,
                                 const MaterialModelTRT& material,
                                 const std::vector<double>& temperature_lag) {
    Problem2D& p = state.problem;
    const double dx = p.Lx / static_cast<double>(p.nx);
    const double dy = p.Ly / static_cast<double>(p.ny);
    const double left_B = group_B(group_edges[0], group_edges[1], kHotBoundaryTemperature);
    set_hotwall_vacuum_boundaries(p, left_B);

    for (int j = 0; j < p.ny; ++j) {
        for (int i = 0; i < p.nx; ++i) {
            const int cell = cell_id(i, j, p.nx);
            Cell2D& c = state.cells[cell];
            c.x_left = static_cast<double>(i) * dx;
            c.y_bottom = static_cast<double>(j) * dy;
            c.dx = dx;
            c.dy = dy;
            c.dt = p.time_step;
            c.velocity.assign(p.groups, kSpeedOfLight);
            c.sigma_t.assign(p.groups, 0.0);
            c.sigma_s.assign(p.groups * p.groups, 0.0);
            c.source.assign(p.cell_block_size(), 0.0);

            const double T = std::max(temperature_lag[cell], kTemperatureFloor);
            const double sigma = group_planck_opacity(group_edges[0], group_edges[1], T, material.density, material.opacity);
            const double B = group_B(group_edges[0], group_edges[1], T);
            const double dB = group_dB_dT(group_edges[0], group_edges[1], T);
            const double denom = material.density * material.cv / p.time_step + 4.0 * kPi * sigma * dB;
            const double alpha = (denom > 0.0) ? (4.0 * kPi * sigma * dB / denom) : 0.0;

            c.sigma_t[0] = sigma;
            c.sigma_s[0] = alpha * sigma;

            const double q = sigma * B - alpha * sigma * B;
            for (int d = 0; d < p.num_dirs(); ++d) {
                const int off = local_angle_group_offset(p, 0, d, 0);
                for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                    c.source[off + dof] = q;
                }
            }
        }
    }
}

std::vector<double> update_temperatures(const SolverState2D& state,
                                        const std::vector<double>& old_temperature,
                                        const std::vector<double>& temperature_lag,
                                        const std::vector<double>& group_edges,
                                        const MaterialModelTRT& material,
                                        const std::vector<double>& scalar_flux) {
    const Problem2D& p = state.problem;
    std::vector<double> next(p.num_cells(), kTemperatureFloor);

    for (int cell = 0; cell < p.num_cells(); ++cell) {
        const double T = std::max(temperature_lag[cell], kTemperatureFloor);
        double lhs = material.density * material.cv / p.time_step;
        double rhs = lhs * old_temperature[cell];

        const double sigma = group_planck_opacity(group_edges[0], group_edges[1], T, material.density, material.opacity);
        const double B = group_B(group_edges[0], group_edges[1], T);
        const double dB = group_dB_dT(group_edges[0], group_edges[1], T);
        lhs += 4.0 * kPi * sigma * dB;
        rhs += 4.0 * kPi * sigma * (scalar_flux[cell] - B + dB * T);

        next[cell] = std::max(kTemperatureFloor, rhs / lhs);
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

void write_summary_json(const std::string& path,
                        const SolverState2D& state,
                        const std::vector<StepStats>& history,
                        const std::string& pvd_path) {
    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Could not open summary JSON for writing: " + path);
    }

    out << std::setprecision(16);
    out << "{\n";
    out << "  \"problem\": \"simple_trt_hotwall_vacuum\",\n";
    out << "  \"nx\": " << state.problem.nx << ",\n";
    out << "  \"ny\": " << state.problem.ny << ",\n";
    out << "  \"groups\": " << state.problem.groups << ",\n";
    out << "  \"num_dirs\": " << state.problem.num_dirs() << ",\n";
    out << "  \"dt\": " << state.problem.time_step << ",\n";
    out << "  \"num_time_steps\": " << state.problem.num_time_steps << ",\n";
    out << "  \"paraview_pvd\": \"" << pvd_path << "\",\n";
    out << "  \"time_history\": [\n";
    for (std::size_t k = 0; k < history.size(); ++k) {
        const auto& rec = history[k];
        out << "    {\"step\": " << rec.step
            << ", \"time\": " << rec.time
            << ", \"nonlinear_iterations\": " << rec.nonlinear_iterations
            << ", \"max_temperature_change\": " << rec.max_temperature_change
            << ", \"transport_iterations\": " << rec.transport_stats.iterations
            << ", \"transport_final_error\": " << rec.transport_stats.final_error
            << ", \"transport_spectral_radius\": " << rec.transport_stats.spectral_radius << "}";
        if (k + 1 != history.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    bool use_openmp = false;
#ifdef THEREFORE2D_EXAMPLE_USE_OPENMP
    use_openmp = true;
#endif

    int num_time_steps = kNumTimeSteps;
    if (argc > 1) {
        num_time_steps = std::stoi(argv[1]);
    }

    const std::vector<MaterialModelTRT> materials = make_trt_materials();
    const MaterialModelTRT material = materials.front();
    const std::vector<double> group_edges{1.0e-4, 4.5e1};

    SolverState2D state;
    Problem2D& p = state.problem;
    p.nx = kNx;
    p.ny = kNy;
    p.Lx = kLx;
    p.Ly = kLy;
    p.groups = 1;
    p.max_iters = kMaxTransportIters;
    p.num_time_steps = num_time_steps;
    p.time_step = kDt;
    p.convergence_tol = kTransportTol;
    p.initialize_from_previous = true;
    p.reuse_factorization = false;
    p.directions = make_level_symmetric_quadrature_2d(kSN);

    state.cells.assign(p.num_cells(), Cell2D{});
    std::vector<double> temperature(p.num_cells(), kInitialTemperature);

    fill_cells_and_coefficients(state, group_edges, material, temperature);

    std::vector<double> initial_flux(p.total_unknowns(), 0.0);
    const double initial_B = group_B(group_edges[0], group_edges[1], kInitialTemperature);
    for (int cell = 0; cell < p.num_cells(); ++cell) {
        for (int dir = 0; dir < p.num_dirs(); ++dir) {
            const int off = global_offset(p, cell, 0, dir, 0);
            for (int dof = 0; dof < kDofsPerAngleGroup2D; ++dof) {
                initial_flux[off + dof] = initial_B;
            }
        }
    }
    initialize_state(state, initial_flux);

    const std::string output_dir = "results/simple_trt_hotwall_vacuum";
    const std::string series_name = "simple_trt_hotwall_vacuum";
    const std::string summary_json = output_dir + "/summary.json";

    ParaviewSeriesWriter2D writer(
        make_rectilinear_grid(state),
        ParaviewSeriesConfig2D{output_dir, series_name, true});

    CpuLUCache cache;
    std::vector<StepStats> history;
    history.reserve(static_cast<std::size_t>(num_time_steps));

    double time = 0.0;
    for (int step = 0; step < num_time_steps; ++step) {
        std::vector<double> old_temperature = temperature;
        std::vector<double> temperature_lag = temperature;

        StepStats rec;
        for (int nl = 0; nl < kMaxNonlinearIters; ++nl) {
            fill_cells_and_coefficients(state, group_edges, material, temperature_lag);
            assemble_cell_matrices(state);
            cache.valid = false;
            build_constant_rhs(state);
            rec.transport_stats = run_one_timestep_cpu(state, cache, use_openmp);

            const std::vector<double> scalar_flux = scalar_flux_groups(state);
            const std::vector<double> next_temperature = update_temperatures(
                state, old_temperature, temperature_lag, group_edges, material, scalar_flux);
            rec.max_temperature_change = max_rel_change(temperature_lag, next_temperature);
            rec.nonlinear_iterations = nl + 1;
            temperature_lag = next_temperature;

            if (rec.max_temperature_change < kNonlinearTol) {
                break;
            }
        }

        temperature = temperature_lag;
        time += p.time_step;
        rec.step = step;
        rec.time = time;
        history.push_back(rec);

        std::vector<CellScalarField2D> fields;
        append_fields(fields, make_scalar_flux_group_fields(state, state.flux_previous, "scalar_flux_g"));
        fields.push_back(make_cell_scalar_field("radiation_temperature", radiation_temperature_field(state)));
        fields.push_back(make_cell_scalar_field("material_temperature", material_temperature_field(temperature)));
        writer.write_step(step, time, fields);

        std::cout << "step " << step
                  << " time=" << time
                  << " nonlinear_iters=" << rec.nonlinear_iterations
                  << " max_dT_rel=" << rec.max_temperature_change
                  << " transport_iters=" << rec.transport_stats.iterations
                  << " transport_error=" << rec.transport_stats.final_error
                  << '\n';
    }

    write_summary_json(summary_json, state, history, writer.pvd_path());

    std::cout << "Wrote:\n"
              << "  " << writer.pvd_path() << '\n'
              << "  " << summary_json << '\n';
    return 0;
}
