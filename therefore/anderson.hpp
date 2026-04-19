#ifndef THEREFORE_ANDERSON_HPP
#define THEREFORE_ANDERSON_HPP

/// fixed_point_accelerator.hpp  (file named gmres.hpp per project convention)
///
/// Anderson mixing / DIIS acceleration for fixed-point iterations.
///
/// Despite the filename the algorithm is NOT standard Krylov GMRES.  It is
/// the DIIS (Direct Inversion in the Iterative Subspace) method from quantum
/// chemistry, which is equivalent to Type-I Anderson mixing.  The name
/// "fixed-point GMRES" is used in some physics codes for this class of
/// method; this implementation ports gmres.py from a prior Python codebase.
///
/// Algorithm (matches gmres.py exactly):
///   Maintain a window of the m most-recent raw iterates x_i = G(x_{i-1})
///   and their fixed-point residuals r_i = x_i - x_{i-1}.
///
///   At each step, form the k×k Gram matrix  G_ij = <r_i, r_j> + reg*δ_ij
///   and solve the (k+1)×(k+1) constrained least-squares system:
///
///     [G   1 ] [c]   [0]
///     [1^T 0 ] [λ] = [1]
///
///   to obtain DIIS coefficients c with sum(c) = 1.
///
///   The accelerated iterate is:
///     x_acc  = sum_i c_i * x_i
///     x_out  = (1 - damping) * x_raw + damping * x_acc
///
/// Typical usage:
///   FixedPointAccelerator acc(/*m=*/5);
///   while (!converged) {
///       xn = G(xo);            // raw fixed-point map output
///       acc.apply(xo, xn);     // xn overwritten with accelerated iterate
///       xo = xn;
///   }
///
/// The accelerator is stateful; call reset() before reusing for a new solve.
///
/// BLAS/LAPACK: uses the Fortran interface (linked via -lblas -llapack).
/// No cblas.h or lapacke.h header required.

#include <cstddef>
#include <vector>

class FixedPointAccelerator {
public:
    /// @param m              History window (number of past (x,r) pairs kept).
    ///                       Must be ≥ 2; clamped upward if smaller.
    /// @param damping        Blend factor: x_out = (1-d)*x_raw + d*x_acc.
    ///                       1.0 = pure Anderson; 0.5–0.8 if iteration is noisy.
    /// @param regularization Small diagonal shift on the Gram matrix for
    ///                       numerical conditioning when history is nearly
    ///                       linearly dependent.  Matches gmres.py default.
    /// @param max_weight_norm L₁ safety check: if ||c||₁ > this the accelerator
    ///                        returns the raw iterate unchanged.  ≤ 0 disables.
    explicit FixedPointAccelerator(int    m               = 5,
                                   double damping         = 1.0,
                                   double regularization  = 1.0e-12,
                                   double max_weight_norm = 1.0e6);

    /// Clear all stored history.  Call before reusing for a new solve or when
    /// the problem size changes.
    void reset();

    /// Accelerate xn **in place**, given the previous iterate xo.
    ///
    /// @param xo   The previous iterate (= input that was fed to G to produce xn).
    /// @param xn   [in]  Raw output of the fixed-point map G(xo).
    ///             [out] Accelerated iterate (overwritten).  If acceleration is
    ///                   skipped (k < 2, bad weights, etc.) xn is left unchanged.
    /// @param res  Optional explicit residual vector (length == xn.size()).
    ///             When null, the residual is computed as xn − xo internally.
    ///             Pass an explicit residual if you have a better-quality one
    ///             (e.g. a transport sweep residual rather than just the iterate
    ///             difference).
    void apply(const std::vector<double>& xo,
               std::vector<double>&       xn,
               const std::vector<double>* res = nullptr);

    /// Non-destructive variant: returns the accelerated vector; xn is unchanged.
    std::vector<double> accelerated(const std::vector<double>& xo,
                                    const std::vector<double>& xn,
                                    const std::vector<double>* res = nullptr);

    // --------------- configuration setters --------------------------------
    /// Resize the history window and reset all stored history.
    void   set_m(int m);
    void   set_damping(double d)         { damping_ = d; }
    void   set_regularization(double r)  { reg_     = r; }
    void   set_max_weight_norm(double w) { max_w_   = w; }

    // --------------- query ------------------------------------------------
    int    m()              const { return m_; }
    double damping()        const { return damping_; }
    double regularization() const { return reg_; }
    /// Number of (x, r) pairs currently in the history (0 ≤ k ≤ m).
    int    history_size()   const { return ring_size_; }

private:
    int    m_;
    double damping_;
    double reg_;
    double max_w_;

    // ---- circular ring buffer of capacity m_ ----------------------------
    // x_pool_[i] / r_pool_[i] are pre-allocated vectors of length n_.
    // Entries ring_start_ .. ring_start_+ring_size_-1  (modulo m_) are valid.
    // New entry → oldest slot overwritten when full.
    std::vector<std::vector<double>> x_pool_;
    std::vector<std::vector<double>> r_pool_;
    int         ring_start_ = 0;
    int         ring_size_  = 0;
    std::size_t n_          = 0;  // vector length; fixed after the first call

    // Initialise pool vectors to length n (called once, on the first apply()).
    void init_pool(std::size_t n);

    // Push a new (x, r) pair into the ring buffer.
    void push(const std::vector<double>& x, const std::vector<double>& r);

    // Read-only access to the i-th history entry (0 = oldest, k-1 = newest).
    const std::vector<double>& x_at(int i) const;
    const std::vector<double>& r_at(int i) const;

    // Solve the (k+1)×(k+1) DIIS constrained least-squares system and write
    // coefficients into coeffs[0 .. k-1].
    // Returns false (and leaves coeffs unchanged) on numerical failure.
    bool solve_diis(std::vector<double>& coeffs) const;
};

#endif // THEREFORE_GMRES_HPP
