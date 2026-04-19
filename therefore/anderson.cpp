#include "anderson.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Fortran BLAS / LAPACK declarations (no cblas.h or lapacke.h required)
// Link with: -lblas -llapack
// ---------------------------------------------------------------------------
extern "C" {
    // BLAS level 1 ---------------------------------------------------------
    /// Dot product: result = x^T y
    double ddot_(const int* n,
                 const double* x, const int* incx,
                 const double* y, const int* incy);

    /// Scaled addition: y += alpha * x
    void daxpy_(const int* n, const double* alpha,
                const double* x, const int* incx,
                double* y,       const int* incy);

    /// Scale: x *= alpha
    void dscal_(const int* n, const double* alpha,
                double* x, const int* incx);

    // LAPACK ---------------------------------------------------------------
    /// Solve A*X = B via LU factorisation (A and B overwritten in place).
    /// Uses column-major (Fortran) storage throughout.
    void dgesv_(const int* n, const int* nrhs,
                double* A,    const int* lda,
                int*    ipiv,
                double* B,    const int* ldb,
                int*    info);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

FixedPointAccelerator::FixedPointAccelerator(int    m,
                                             double damping,
                                             double regularization,
                                             double max_weight_norm)
    : m_(std::max(m, 2))
    , damping_(damping)
    , reg_(regularization)
    , max_w_(max_weight_norm)
    , x_pool_(m_)
    , r_pool_(m_)
{}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void FixedPointAccelerator::reset() {
    ring_start_ = 0;
    ring_size_  = 0;
    n_          = 0;
    // Leave pool vectors allocated — they will be reused on the next call.
}

void FixedPointAccelerator::set_m(int m) {
    m_ = std::max(m, 2);
    x_pool_.assign(m_, {});
    r_pool_.assign(m_, {});
    reset();
}

void FixedPointAccelerator::apply(const std::vector<double>& xo,
                                  std::vector<double>&       xn,
                                  const std::vector<double>* explicit_res) {
    if (xo.size() != xn.size())
        throw std::invalid_argument(
            "FixedPointAccelerator::apply: xo and xn must have the same length");

    const std::size_t n = xn.size();

    // Initialise or validate the vector length.
    if (n_ == 0) {
        init_pool(n);
    } else if (n_ != n) {
        throw std::invalid_argument(
            "FixedPointAccelerator::apply: vector length changed — "
            "call reset() before reusing the accelerator on a different problem");
    }

    // Build the residual r = xn - xo  (or use the caller-supplied one).
    std::vector<double> r(n);
    if (explicit_res) {
        if (explicit_res->size() != n)
            throw std::invalid_argument(
                "FixedPointAccelerator::apply: explicit residual length mismatch");
        r = *explicit_res;
    } else {
        for (std::size_t i = 0; i < n; ++i) r[i] = xn[i] - xo[i];
    }

    // Store the RAW new iterate and its residual in the ring buffer.
    // Note: we store xn BEFORE any damping/blending so the history reflects
    //       actual outputs of the fixed-point map G.
    push(xn, r);
    const int k = ring_size_;

    // Need at least 2 history entries before acceleration helps.
    if (k < 2) return;

    // Solve the DIIS constrained least-squares system.
    std::vector<double> coeffs(k);
    if (!solve_diis(coeffs)) return;   // numerical failure → keep raw xn

    // Optional safety check: very large weights indicate near-linear-dependence
    // in the residual history.  Fall back to the raw iterate in that case.
    if (max_w_ > 0.0) {
        double w_l1 = 0.0;
        for (double c : coeffs) w_l1 += std::abs(c);
        if (w_l1 > max_w_) return;
    }

    // Form the accelerated iterate:  x_acc = sum_i  c_i * x_history_i
    // Using BLAS daxpy for each term:  x_acc += c_i * x_i
    std::vector<double> x_acc(n, 0.0);
    const int nn  = static_cast<int>(n);
    const int one = 1;
    for (int i = 0; i < k; ++i) {
        const double ci = coeffs[i];
        daxpy_(&nn, &ci, x_at(i).data(), &one, x_acc.data(), &one);
    }

    // Blend:  xn = (1 - damping) * xn_raw + damping * x_acc
    if (damping_ == 1.0) {
        // Avoid pointless arithmetic: pure Anderson.
        xn = std::move(x_acc);
    } else {
        // xn *= (1 - damping)
        const double w = 1.0 - damping_;
        dscal_(&nn, &w, xn.data(), &one);
        // xn += damping * x_acc
        daxpy_(&nn, &damping_, x_acc.data(), &one, xn.data(), &one);
    }
}

std::vector<double> FixedPointAccelerator::accelerated(
    const std::vector<double>& xo,
    const std::vector<double>& xn,
    const std::vector<double>* res) {
    std::vector<double> out = xn;      // copy
    apply(xo, out, res);
    return out;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void FixedPointAccelerator::init_pool(std::size_t n) {
    n_ = n;
    for (auto& v : x_pool_) { v.assign(n, 0.0); }
    for (auto& v : r_pool_) { v.assign(n, 0.0); }
}

void FixedPointAccelerator::push(const std::vector<double>& x,
                                 const std::vector<double>& r) {
    // Slot to write: next available position in the ring.
    // When the buffer is full, this overwrites the oldest entry.
    const int slot = (ring_start_ + ring_size_) % m_;
    x_pool_[slot] = x;
    r_pool_[slot] = r;

    if (ring_size_ < m_) {
        // Buffer not yet full: grow it.
        ++ring_size_;
    } else {
        // Buffer full: the oldest entry has been overwritten;
        // advance the ring start so x_at(0) still refers to the oldest valid entry.
        ring_start_ = (ring_start_ + 1) % m_;
    }
}

const std::vector<double>& FixedPointAccelerator::x_at(int i) const {
    return x_pool_[(ring_start_ + i) % m_];
}

const std::vector<double>& FixedPointAccelerator::r_at(int i) const {
    return r_pool_[(ring_start_ + i) % m_];
}

bool FixedPointAccelerator::solve_diis(std::vector<double>& coeffs) const {
    // Build the (k+1) × (k+1) DIIS system in COLUMN-MAJOR (Fortran) order.
    //
    //   K = [ G + reg*I ,  1   ]     dimensions: (k) × (k)  top-left
    //       [ 1^T       ,  0   ]                 (1) × (1)  bottom-right
    //
    //   rhs = [ 0, ..., 0, 1 ]^T
    //
    // Solving K * [c; λ] = rhs gives the DIIS weights c with sum(c_i) = 1
    // that minimise ‖ R c ‖² (R = column matrix of residuals).

    const int k   = ring_size_;
    const int km1 = k + 1;                      // system size

    const int    nn  = static_cast<int>(n_);
    const int    one = 1;

    // Allocate the system matrix (column-major, km1 × km1).
    std::vector<double> K(km1 * km1, 0.0);

    // Fill the k × k Gram matrix block.
    // G[i,j] = r_i · r_j + reg * δ_ij
    // Column-major index: row i, col j  →  i + j * km1
    for (int j = 0; j < k; ++j) {
        for (int i = j; i < k; ++i) {
            double g = ddot_(&nn,
                             r_at(i).data(), &one,
                             r_at(j).data(), &one);
            if (i == j) g += reg_;
            K[i + j * km1] = g;
            K[j + i * km1] = g;  // symmetric
        }
        // Constraint row/column: ones in the (k+1)-th row and column.
        K[k + j * km1] = 1.0;   // row k, col j
        K[j + k * km1] = 1.0;   // row j, col k
    }
    // K[k, k] = 0 (already zero-initialised).

    // Right-hand side: [0, ..., 0, 1].
    std::vector<double> rhs(km1, 0.0);
    rhs[k] = 1.0;

    // Solve with LAPACK dgesv (LU + partial pivoting).
    // K and rhs are overwritten: K → LU factors, rhs → solution [c; λ].
    std::vector<int> ipiv(km1);
    int nrhs = 1, info = 0;
    dgesv_(&km1, &nrhs,
           K.data(), &km1,
           ipiv.data(),
           rhs.data(), &km1,
           &info);

    if (info != 0) return false;   // singular or near-singular system

    // Verify all coefficients are finite (guard against NaN propagation).
    for (int i = 0; i < k; ++i) {
        if (!std::isfinite(rhs[i])) return false;
    }

    coeffs.assign(rhs.begin(), rhs.begin() + k);
    return true;
}
