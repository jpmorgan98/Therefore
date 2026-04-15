#include "transport2d.hpp"

#ifdef THEREFORE2D_ENABLE_ROCM

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <stdexcept>

namespace therefore2d {
namespace {

rocblas_handle as_handle(void* handle) {
    return reinterpret_cast<rocblas_handle>(handle);
}

void hip_check(hipError_t status, const char* msg) {
    if (status != hipSuccess) {
        throw std::runtime_error(msg);
    }
}

void rocblas_check(rocblas_status status, const char* msg) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(msg);
    }
}

} // namespace

void factor_cells_rocm(const SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;
    cache.n = p.cell_block_size();
    cache.batch_count = p.num_cells();
    cache.stride_a = static_cast<std::size_t>(p.cell_block_elems());
    cache.stride_b = static_cast<std::size_t>(p.cell_block_size());
    cache.stride_p = static_cast<std::size_t>(p.cell_block_size());

    if (!cache.rocblas_handle) {
        rocblas_handle handle = nullptr;
        rocblas_check(rocblas_create_handle(&handle), "rocBLAS handle creation failed.");
        cache.rocblas_handle = reinterpret_cast<void*>(handle);
    }

    const std::size_t a_bytes = sizeof(double) * cache.stride_a * cache.batch_count;
    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    const std::size_t p_bytes = sizeof(int) * cache.stride_p * cache.batch_count;
    const std::size_t i_bytes = sizeof(int) * cache.batch_count;

    if (!cache.d_lu) {
        hip_check(hipMalloc(&cache.d_lu, a_bytes), "hipMalloc failed for d_lu.");
        hip_check(hipMalloc(&cache.d_rhs, b_bytes), "hipMalloc failed for d_rhs.");
        hip_check(hipMalloc(&cache.d_pivots, p_bytes), "hipMalloc failed for d_pivots.");
        hip_check(hipMalloc(&cache.d_info, i_bytes), "hipMalloc failed for d_info.");
    }

    hip_check(hipMemcpy(cache.d_lu, state.cell_matrices.data(), a_bytes, hipMemcpyHostToDevice), "hipMemcpy failed for LU upload.");

    // Factor once, then reuse with dgetrs on every iteration. This is the 2D analog
    // of the optimization already present in the uploaded GPU path.
    rocblas_check(
        rocsolver_dgetrf_strided_batched(
            as_handle(cache.rocblas_handle),
            cache.n,
            cache.n,
            cache.d_lu,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_a),
            cache.d_pivots,
            static_cast<rocblas_stride>(cache.stride_p),
            cache.d_info,
            cache.batch_count),
        "rocsolver_dgetrf_strided_batched failed.");

    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize failed after LU factorization.");
    cache.valid = true;
}

void solve_cells_rocm(const SolverState2D& state, RocmLUCache& cache, std::vector<double>& rhs) {
    const Problem2D& p = state.problem;
    if (!cache.valid) {
        factor_cells_rocm(state, cache);
    }

    const std::size_t b_bytes = sizeof(double) * cache.stride_b * cache.batch_count;
    hip_check(hipMemcpy(cache.d_rhs, rhs.data(), b_bytes, hipMemcpyHostToDevice), "hipMemcpy failed for RHS upload.");

    rocblas_check(
        rocsolver_dgetrs_strided_batched(
            as_handle(cache.rocblas_handle),
            rocblas_operation_none,
            cache.n,
            1,
            cache.d_lu,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_a),
            cache.d_pivots,
            static_cast<rocblas_stride>(cache.stride_p),
            cache.d_rhs,
            cache.n,
            static_cast<rocblas_stride>(cache.stride_b),
            cache.batch_count),
        "rocsolver_dgetrs_strided_batched failed.");

    hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize failed after dgetrs.");
    hip_check(hipMemcpy(rhs.data(), cache.d_rhs, b_bytes, hipMemcpyDeviceToHost), "hipMemcpy failed for RHS download.");
}

IterationStats run_one_timestep_rocm(SolverState2D& state, RocmLUCache& cache) {
    const Problem2D& p = state.problem;
    if (!cache.valid || !p.reuse_factorization) {
        factor_cells_rocm(state, cache);
    }

    state.flux_last = p.initialize_from_previous ? state.flux_previous : std::vector<double>(p.total_unknowns(), 0.0);

    IterationStats stats{};
    for (int it = 0; it < p.max_iters; ++it) {
        state.flux_current = state.rhs_const;
        add_upwind_inflow_rhs(state.flux_current, state.flux_last, state);
        solve_cells_rocm(state, cache, state.flux_current);

        stats.final_error = relative_l2_error(state.flux_last, state.flux_current);
        stats.iterations = it + 1;
        state.flux_last.swap(state.flux_current);

        if (stats.final_error < p.convergence_tol) {
            break;
        }
    }

    state.flux_previous = state.flux_last;
    return stats;
}

void destroy_rocm_cache(RocmLUCache& cache) {
    if (cache.d_lu) { hipFree(cache.d_lu); cache.d_lu = nullptr; }
    if (cache.d_rhs) { hipFree(cache.d_rhs); cache.d_rhs = nullptr; }
    if (cache.d_pivots) { hipFree(cache.d_pivots); cache.d_pivots = nullptr; }
    if (cache.d_info) { hipFree(cache.d_info); cache.d_info = nullptr; }
    if (cache.rocblas_handle) {
        rocblas_destroy_handle(as_handle(cache.rocblas_handle));
        cache.rocblas_handle = nullptr;
    }
    cache.valid = false;
}

} // namespace therefore2d

#endif
