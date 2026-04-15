# 2D multigroup time-dependent SN corner-balance extension

This refactor keeps the same core design as the uploaded 1D code:

- each **cell** assembles one dense matrix over **all groups and all directions**,
- all cell systems are independent inside a fixed-point iteration,
- the cell solves can run on either **CPU** or **ROCm strided-batched GPU solvers**,
- storage remains centered on `std::vector<double>` and flat double arrays.

The uploaded 1D code already uses exactly that pattern: a cell-local dense system with one batched solve per iteration on GPU and one cell solve per cell on CPU. The 2D refactor here preserves that architecture while replacing the 1D `(left,right) x (low,high)` unknown structure by a 2D `(LL,LR,UL,UR) x (low,high)` structure. This directly generalizes the 4 unknowns per angle-group in `A_pos_rm` / `A_neg_rm` to 8 unknowns per angle-group in 2D. fileciteturn0file0 fileciteturn0file3 fileciteturn0file2

## Files

- `transport2d.hpp`: public data structures and API.
- `transport2d.cpp`: 2D assembly, CPU LU factorization, CPU solve loop.
- `transport2d_rocm.cpp`: ROCm LU factorization and strided-batched solve path.
- `example_run_2d.cpp`: small CPU example.

## Unknown layout

For one cell `K`, one energy group `g`, and one direction `m = (mu, eta)`, define the local 8-vector

**(1)**

```text
psi(K,g,m) = [
  psi_LL^L, psi_LR^L, psi_UL^L, psi_UR^L,
  psi_LL^H, psi_LR^H, psi_UL^H, psi_UR^H
]^T
```

where `L` and `H` are the two temporal multiple-balance unknowns on the slab, and the four spatial entries are the corner values

```text
LL = lower-left, LR = lower-right, UL = upper-left, UR = upper-right.
```

With `G` groups and `M` directions, each cell matrix therefore has size

**(2)**

```text
N_cell = 8 * G * M .
```

## 2D local balance equation

For each cell, direction, and outgoing group, the local dense problem is

**(3)**

```text
A_cell * Psi_cell = b_const + b_upwind
```

where `Psi_cell` stacks all `psi(K,g,m)` blocks for that cell.

### 1. Diagonal angle-group block

For one `(g,m)` block, define

**(4)**

```text
ax = |mu_m| * dy / 2
ay = |eta_m| * dx / 2
gamma_g = sigma_t,g * dx * dy / 4
tau_g = dx * dy / (v_g * dt)
```

and let `S(g,m)` be the 4x4 spatial-reaction block

**(5)**

```text
S(g,m) = gamma_g * I4 + Kx(g,m) + Ky(g,m)
```

with sign-dependent x and y corner-balance streaming operators.

For `mu >= 0`,

**(6a)**

```text
Kx+ = ax *
[  1   1   0   0
  -1   1   0   0
   0   0   1   1
   0   0  -1   1 ]
```

For `mu < 0`,

**(6b)**

```text
Kx- = ax *
[ -1  -1   0   0
   1  -1   0   0
   0   0  -1  -1
   0   0   1  -1 ]
```

For `eta >= 0`,

**(7a)**

```text
Ky+ = ay *
[  1   0   1   0
   0   1   0   1
  -1   0   1   0
   0  -1   0   1 ]
```

For `eta < 0`,

**(7b)**

```text
Ky- = ay *
[ -1   0  -1   0
   0  -1   0  -1
   1   0  -1   0
   0   1   0  -1 ]
```

The full 8x8 multiple-balance block is then

**(8)**

```text
A(g,m) = [  S(g,m)         (tau_g/2) I4 ]
         [ -tau_g I4   S(g,m) + tau_g I4 ]
```

This is the direct 2D analog of the 1D `A_pos_rm` / `A_neg_rm` pattern in the uploaded code. The 1D code uses 4 unknowns per angle-group and coefficients proportional to `dx/2`, `dx/4`, and `dx/(v dt)`; the 2D extension keeps the same algebraic structure with `dx` replaced by the correct cell measure or face measure in 2D. fileciteturn0file0

### 2. Scattering coupling across all groups and directions

To preserve the existing “dense-per-cell over all angles and groups” design, scattering is assembled as off-diagonal coupling between all incoming and outgoing direction/group blocks.

For isotropic multigroup scattering,

**(9)**

```text
A_scat[(g,m,a),(g',m',a')] =
    -delta(a,a') * (dx*dy/8) * sigma_s(g <- g') * w(m')
```

where `a` and `a'` are the local dof indices in `{0,...,7}`.

This is the 2D version of the same-dof dense coupling used by the uploaded `scatter(...)` builder, which couples all angles through the angular weights while keeping the local corner/time index aligned. fileciteturn0file0

## RHS terms

### Constant part

The constant part contains source plus the previous-time contribution:

**(10)**

```text
b_const = [ (dx*dy/8) * Q^L + (tau_g/2) * psi_prev^L ]
          [ (dx*dy/8) * Q^H                       ]
```

### Upwind part

No sweep is required. Instead, neighbor-cell inflow is taken from the last outer iteration and moved to the RHS.

For example, for `mu > 0`, west-neighbor values enter the left corners `(LL, UL)` of both temporal blocks; for `mu < 0`, east-neighbor values enter the right corners `(LR, UR)`. Likewise, `eta > 0` uses south-neighbor inflow and `eta < 0` uses north-neighbor inflow.

That gives the point-Jacobi / OCI style iteration

**(11)**

```text
A_cell * Psi_cell^(ell+1) = b_const + b_upwind(Psi^(ell))
```

which preserves the “each cell solve is independent inside the iteration” property needed for both OpenMP and ROCm batched solves. This matches the structure already present in the uploaded convergence loop, where the constant RHS is built once and the neighbor-dependent inflow term is refreshed each iteration before batched cell solves. fileciteturn0file3 fileciteturn0file2

## Boundary storage

The new boundary arrays are also flat `std::vector<double>` containers.

- `west` / `east` size: `ny * groups * dirs * 4`
- `south` / `north` size: `nx * groups * dirs * 4`

Face dof ordering is:

- west/east: `[low_bottom, low_top, high_bottom, high_top]`
- south/north: `[low_left, low_right, high_left, high_right]`

If a boundary vector is left empty, vacuum inflow is assumed.

## CPU and GPU backends

### CPU

`transport2d.cpp` contains:

- a reference dense LU factorization with partial pivoting,
- LU reuse across iterations,
- optional OpenMP over cells.

### ROCm

`transport2d_rocm.cpp` uses the ROCm path that best matches the uploaded implementation:

1. factor all cell matrices once with `rocsolver_dgetrf_strided_batched`,
2. solve each new RHS with `rocsolver_dgetrs_strided_batched`.

That is the same optimization idea already present in the uploaded GPU code, where the first iteration performs a full factor-and-solve and later iterations reuse the LU factors through `dgetrs`. fileciteturn0file2 fileciteturn0file3

## Build notes

### CPU example

```bash
g++ -std=c++20 -O2 /mnt/data/example_run_2d.cpp /mnt/data/transport2d.cpp -o /mnt/data/example_run_2d
```

### CPU + OpenMP

```bash
g++ -std=c++20 -O2 -fopenmp /mnt/data/example_run_2d.cpp /mnt/data/transport2d.cpp -o /mnt/data/example_run_2d_omp
```

### ROCm

```bash
hipcc -std=c++20 -O2 -DTHEREFORE2D_ENABLE_ROCM \
  /mnt/data/example_run_2d.cpp \
  /mnt/data/transport2d.cpp \
  /mnt/data/transport2d_rocm.cpp \
  -lrocsolver -lrocblas -o /mnt/data/example_run_2d_rocm
```

## Mapping from the uploaded 1D code to this refactor

- `A_pos_rm` / `A_neg_rm` become the 2D angle-group block in Eq. (8). fileciteturn0file0
- `scatter(...)` becomes Eq. (9), still dense across all groups/directions inside each cell. fileciteturn0file0
- `PBJlinear_solver(...)` becomes `solve_cells_cpu(...)`, still one dense solve per cell. fileciteturn0file3
- `amdGPU_dgesv_strided_batched(...)` evolves into LU-once plus `dgetrs` batched solves in `transport2d_rocm.cpp`. fileciteturn0file2
- the convergence loop remains an OCI / point-Jacobi iteration with no sweep. fileciteturn0file3

## Important assumptions

This extension is intentionally conservative and keeps the algebraic structure close to the uploaded code. A few assumptions are explicit:

1. scattering is isotropic in angle,
2. scattering is multigroup through a full `groups x groups` matrix,
3. the included quadrature helper is a **tensor-product test quadrature**, not a production level-symmetric 2D SN set,
4. boundaries are explicit face vectors or vacuum if omitted.

If you want, the next natural step is to swap the example quadrature for your preferred 2D SN set and wire this API into your existing driver/input format.
