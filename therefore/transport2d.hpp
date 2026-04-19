#ifndef THEREFORE_TRANSPORT2D_HPP
#define THEREFORE_TRANSPORT2D_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace therefore2d {

constexpr int kSpatialCorners2D = 4;
constexpr int kTemporalMoments = 2;
constexpr int kDofsPerAngleGroup2D = kSpatialCorners2D * kTemporalMoments;

enum class SolveBackend {
    CpuReferenceLU,
    CpuOpenMPLU,
    RocmStridedBatchedLU
};

struct Direction2D {
    double mu = 0.0;
    double eta = 0.0;
    double weight = 0.0;
};

struct BoundaryData2D {
    // Vacuum is obtained by leaving these vectors empty.
    // West/East face storage for one boundary segment uses 4 values:
    // [low_bottom, low_top, high_bottom, high_top].
    // South/North face storage for one boundary segment uses 4 values:
    // [low_left, low_right, high_left, high_right].
    std::vector<double> west;
    std::vector<double> east;
    std::vector<double> south;
    std::vector<double> north;
};

struct Cell2D {
    double x_left = 0.0;
    double y_bottom = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dt = 0.0;

    // Size = groups
    std::vector<double> velocity;
    std::vector<double> sigma_t;

    // Size = groups * groups, flattened as sigma_s[g_to * G + g_from].
    std::vector<double> sigma_s;

    // Size = 8 * groups * directions. Layout matches local_angle_group_offset().
    std::vector<double> source;
};

struct Problem2D {
    int nx = 0;
    int ny = 0;
    double Lx = 0;
    double Ly = 0;
    int groups = 0;
    int max_iters = 200;
    int num_time_steps = 1;
    double time_step = 0.1;
    double convergence_tol = 1.0e-10;
    bool initialize_from_previous = true;
    bool reuse_factorization = true;
    BoundaryData2D boundary;
    std::vector<Direction2D> directions;

    int num_cells() const { return nx * ny; }
    int num_dirs() const { return static_cast<int>(directions.size()); }
    int cell_block_size() const { return groups * num_dirs() * kDofsPerAngleGroup2D; }
    int cell_block_elems() const { return cell_block_size() * cell_block_size(); }
    int total_unknowns() const { return num_cells() * cell_block_size(); }
    double total_time() const { return num_time_steps * time_step;}
};

struct IterationStats {
    int iterations = 0;
    double final_error = 0.0;
    double error_previous = 0.0;
    double spectral_radius = 0.0;

    void iterate(){
        error_previous = final_error;
    };
};

struct CpuLUCache {
    std::vector<double> lu;       // Col-major LU factors, all cells packed consecutively.
    std::vector<int> pivots;      // Size = num_cells * cell_block_size.
    bool valid = false;
};

struct SolverState2D {
    Problem2D problem;
    std::vector<Cell2D> cells;

    // Col-major cell matrices, one dense matrix per cell.
    std::vector<double> cell_matrices;

    // Constant part of the RHS for the current time step.
    std::vector<double> rhs_const;

    // Iteration vectors.
    std::vector<double> flux_previous;
    std::vector<double> flux_last;
    std::vector<double> flux_current;
};

inline int cell_id(int i, int j, int nx) {
    return j * nx + i;
}

inline int local_angle_group_offset(const Problem2D& problem, int group, int dir, int dof) {
    return ((group * problem.num_dirs() + dir) * kDofsPerAngleGroup2D) + dof;
}

inline int global_offset(const Problem2D& problem, int cell, int group, int dir, int dof) {
    return cell * problem.cell_block_size() + local_angle_group_offset(problem, group, dir, dof);
}

inline int face_offset_west_east(const Problem2D& problem, int boundary_j, int group, int dir, int face_dof) {
    return (((boundary_j * problem.groups + group) * problem.num_dirs() + dir) * 4) + face_dof;
}

inline int face_offset_south_north(const Problem2D& problem, int boundary_i, int group, int dir, int face_dof) {
    return (((boundary_i * problem.groups + group) * problem.num_dirs() + dir) * 4) + face_dof;
}

inline void require(bool cond, const std::string& message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void validate_problem(const SolverState2D& state);
void initialize_state(SolverState2D& state, const std::vector<double>& initial_condition);
void assemble_cell_matrices(SolverState2D& state);
void build_constant_rhs(SolverState2D& state);
void add_upwind_inflow_rhs(std::vector<double>& rhs, const std::vector<double>& iterate_flux, const SolverState2D& state);

double relative_l2_error(const std::vector<double>& previous, const std::vector<double>& current);

struct TimestepRecord2D {
    int step = 0;
    double time = 0.0;
    IterationStats stats;
};

struct TransportOutputFiles2D {
    std::string output_dir = "results/transport";
    std::string series_name = "transport";
    std::string summary_json = "results/transport_run_summary.json";
    bool write_pvd_every_step = true;
};

double cell_average_angular_flux(const SolverState2D& state,
                                 const std::vector<double>& flux,
                                 int cell,
                                 int group,
                                 int dir);
double cell_centered_scalar_flux(const SolverState2D& state,
                                 const std::vector<double>& flux,
                                 int cell,
                                 int group);

void factor_cells_cpu(const SolverState2D& state, CpuLUCache& cache, bool use_openmp);
void solve_cells_cpu(const SolverState2D& state, const CpuLUCache& cache, std::vector<double>& rhs, bool use_openmp);

IterationStats run_one_timestep_cpu(
    SolverState2D& state,
    CpuLUCache& cache,
    bool use_openmp);

std::vector<TimestepRecord2D> run_time_cpu(
    SolverState2D& state,
    CpuLUCache& cache,
    bool use_openmp,
    const TransportOutputFiles2D& outputs = TransportOutputFiles2D{});

std::vector<Direction2D> make_tensor_product_quadrature_2d(const std::vector<double>& mu, const std::vector<double>& w);
std::vector<Direction2D> make_level_symmetric_quadrature_2d(int sn_order);

#ifdef THEREFORE2D_ENABLE_ROCM
struct RocmLUCache {
    int n = 0;
    int batch_count = 0;
    std::size_t stride_a = 0;
    std::size_t stride_b = 0;
    std::size_t stride_p = 0;
    void* rocblas_handle = nullptr;
    double* d_lu = nullptr;
    double* d_rhs = nullptr;
    double* d_flux_last = nullptr;
    double* d_rhs_const = nullptr;
    double* d_work = nullptr;
    double* d_cell_dx = nullptr;
    double* d_cell_dy = nullptr;
    double* d_cell_dt = nullptr;
    double* d_cell_velocity = nullptr;
    double* d_cell_source = nullptr;
    double* d_dir_mu = nullptr;
    double* d_dir_eta = nullptr;
    double* d_boundary_west = nullptr;
    double* d_boundary_east = nullptr;
    double* d_boundary_south = nullptr;
    double* d_boundary_north = nullptr;
    int* d_pivots = nullptr;
    int* d_info = nullptr;
    bool sweep_data_valid = false;
    bool valid = false;
};

void factor_cells_rocm(const SolverState2D& state, RocmLUCache& cache);
void solve_cells_rocm(const SolverState2D& state, RocmLUCache& cache, std::vector<double>& rhs);
void destroy_rocm_cache(RocmLUCache& cache);
IterationStats run_one_timestep_rocm(SolverState2D& state, RocmLUCache& cache);
std::vector<TimestepRecord2D> run_time_rocm(
    SolverState2D& state,
    RocmLUCache& cache,
    const TransportOutputFiles2D& outputs = TransportOutputFiles2D{});
#endif

} // namespace therefore2d

#endif
