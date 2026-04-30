// ============================================================
// CHANGES TO transport2d.hpp — replace the existing RocmLUCache
// struct with the one below.  Everything else in the header is
// unchanged.
// ============================================================

#ifdef THEREFORE2D_ENABLE_ROCM
struct RocmLUCache {
    // ---- problem dimensions (set by factor_cells_rocm) ----
    int n            = 0;    // cell_block_size
    int batch_count  = 0;    // total number of cells (ncells)
    int chunk_size   = 0;    // cells processed per chunk (<= ncells)
    std::size_t stride_a = 0;  // n * n  (elements per cell matrix)
    std::size_t stride_b = 0;  // n      (elements per cell RHS)
    std::size_t stride_p = 0;  // n      (pivot entries per cell)

    // ---- rocBLAS handle ----
    void* rocblas_handle = nullptr;

    // ---- chunk-sized temporary LU workspace ----------------
    // Replaces the old d_lu / d_pivots / d_info fields which
    // held ncells matrices simultaneously (O(ncells × n²) bytes).
    // These buffers hold at most chunk_size matrices at a time;
    // they are reused each iteration and never persist between
    // source-iteration steps.
    double* d_lu_chunk     = nullptr;   // [chunk_size * n * n]
    int*    d_pivots_chunk = nullptr;   // [chunk_size * n]
    int*    d_info_chunk   = nullptr;   // [chunk_size]

    // ---- flux / RHS state (total_unknowns = ncells * n) ----
    double* d_rhs       = nullptr;   // working RHS (rhs_const + upwind)
    double* d_flux_last = nullptr;   // previous-iterate flux
    double* d_rhs_const = nullptr;   // time-step constant RHS
    double* d_work      = nullptr;   // scratch for convergence check

    // ---- cell geometry (uploaded once, reused every timestep) ----
    double* d_cell_dx       = nullptr;   // [ncells]
    double* d_cell_dy       = nullptr;   // [ncells]
    double* d_cell_dt       = nullptr;   // [ncells]
    double* d_cell_velocity = nullptr;   // [ncells * groups]
    double* d_cell_source   = nullptr;   // [total_unknowns]

    // ---- cross-sections (NEW: needed for GPU-side assembly) ----
    double* d_cell_sigma_t  = nullptr;   // [ncells * groups]
    double* d_cell_sigma_s  = nullptr;   // [ncells * groups * groups]

    // ---- quadrature (direction cosines + weights) ----
    double* d_dir_mu     = nullptr;   // [num_dirs]
    double* d_dir_eta    = nullptr;   // [num_dirs]
    double* d_dir_weight = nullptr;   // [num_dirs]  (NEW: needed for scatter)

    // ---- boundary conditions ----
    double* d_boundary_west  = nullptr;
    double* d_boundary_east  = nullptr;
    double* d_boundary_south = nullptr;
    double* d_boundary_north = nullptr;

    // ---- state flags ----
    bool sweep_data_valid = false;  // cell + cross-section data uploaded
    bool valid            = false;  // buffers allocated, ready to iterate
};
#endif // THEREFORE2D_ENABLE_ROCM

// ============================================================
// Add these function declarations to the #ifdef THEREFORE2D_ENABLE_ROCM
// block in transport2d.hpp (alongside the existing factor_cells_rocm etc.):
//
//   // One-time GPU initialisation: allocates buffers, uploads geometry,
//   // cross-sections, quadrature, and boundary conditions.
//   void factor_cells_rocm(const SolverState2D& state, RocmLUCache& cache);
//
//   // Lightweight per-nonlinear-iteration refresh for TRT outer loops.
//   // Re-uploads sigma_t, sigma_s, source without reallocating.
//   void refresh_cell_opacities_rocm(const SolverState2D& state, RocmLUCache& cache);
//
//   // Run a full source-iteration timestep entirely on the GPU.
//   IterationStats run_one_timestep_rocm(SolverState2D& state, RocmLUCache& cache);
//
//   void destroy_rocm_cache(RocmLUCache& cache);
// ============================================================

// ============================================================
// REMOVED fields (were in the old RocmLUCache):
//
//   double* d_lu;       // ncells * n * n  ← replaced by d_lu_chunk
//   int*    d_pivots;   // ncells * n      ← replaced by d_pivots_chunk
//   int*    d_info;     // ncells          ← replaced by d_info_chunk
//
// All three are now chunk-sized, so peak LU allocation drops
// from O(ncells × n²) to O(chunk_size × n²).
// ============================================================
