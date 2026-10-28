# fast-wigner-d

Wigner *d*- and *D*-matrices for **any spin** — integer *and* half-integer — and any rotation
angle, accurate to about 10⁻¹⁴ even for *j* in the thousands. A single small Cython extension,
parallelised with OpenMP, whose only runtime dependency is NumPy.

* **Half-integer spin is first-class.** *j* = ½, 3/2, 5/2, … work exactly like integer *j*.
  The fast recursion-based libraries (e.g. [`spherical`](https://github.com/moble/spherical),
  [`wigner`](https://github.com/ntessore/wigner)) handle integer ℓ only.
* **Robust.** The matrix elements are evaluated from Jacobi polynomials in log space with exact
  rescaling, so nothing overflows or turns into NaN, and every real angle (negative, > π, > 2π)
  is valid. Checked against mpmath references with hundreds of digits up to *j* = 2000.
* **Fast.** About a microsecond for a small matrix; large matrices are split over all cores
  (the full 2001 × 2001 matrix for *j* = 1000 takes ≈ 0.2 s on a 10-core laptop).
* **Varshalovich conventions** (see [Conventions](#conventions)).

## Installation

```bash
pip install fast-wigner-d
```

Building from source (what pip does when no wheel matches your platform) needs a C compiler;
pip fetches Cython and setuptools by itself:

```bash
git clone <repository-url> fast-wigner-d
cd fast-wigner-d
pip install .
```

OpenMP is optional: without it the package builds and gives identical results, only serially.
It is detected automatically at build time.

| Platform | OpenMP |
|---|---|
| Linux (gcc) | works out of the box |
| Windows (MSVC) | works out of the box |
| macOS (Apple clang) | needs the libomp runtime: `brew install libomp`, or `conda install -c conda-forge llvm-openmp` inside a conda environment |

Build-time environment variables: `FAST_WIGNER_D_OPENMP=0` (build without OpenMP) or `=1` (fail
if it is unavailable), `LIBOMP_PREFIX=/path/to/libomp` (non-standard libomp location on macOS),
`FAST_WIGNER_D_NATIVE=1` (add `-march=native`, for local builds only). To check what you got:

```bash
python -c "import fast_wigner_d as f; print(f.__version__, 'OpenMP:', f.OPENMP_ENABLED)"
```

## Quick start

```python
import numpy as np
from fast_wigner_d import wigner_d, wigner_D

d = wigner_d(3/2, 0.7)        # (4, 4) float64 array with d[m' + j, m + j] = d^j_{m'm}(0.7)
d[3, 2]                       # d^{3/2}_{3/2, 1/2}(0.7) = -sqrt(3) cos²(0.35) sin(0.35)

wigner_d(1000.5, 2.0)         # (2002, 2002): no overflow, |d dᵀ - 1| ~ 1e-13
wigner_d(1, np.linspace(0, np.pi, 50)).shape   # (50, 3, 3): one matrix per angle

D = wigner_D(1/2, 0.3, 1.2, -0.4)              # complex (2, 2), z-y-z Euler angles
wigner_d(500, 1.0, num_threads=1)              # cap the OpenMP threads of one call
```

`j` may be an `int`, a `float` such as `2.5`, or a `fractions.Fraction`. Rows and columns run
over *m* = −*j*, −*j*+1, …, *j*, so the element d<sup>j</sup><sub>m′m</sub> sits at
`d[int(m' + j), int(m + j)]`. A worked tour is in
[`examples/example_usage.ipynb`](examples/example_usage.ipynb).

## Conventions

Everything follows D. A. Varshalovich, A. N. Moskalev and V. K. Khersonskii, *Quantum Theory
of Angular Momentum* (World Scientific, 1988), Chapter 4. Rotations are active and
right-handed, the Euler angles are z-y-z, and the phases are those of Condon and Shortley:

```text
D^j_{m'm}(α, β, γ) = <j m'| exp(-iα J_z) exp(-iβ J_y) exp(-iγ J_z) |j m>
                   = exp(-i m' α) d^j_{m'm}(β) exp(-i m γ),
d^j_{m'm}(β)       = <j m'| exp(-iβ J_y) |j m>.
```

For example, with rows and columns ordered *m* = −½, +½:

```text
d^{1/2}(β) = [[ cos(β/2),  sin(β/2)],        d^1_{10}(β) = -sin(β)/√2
              [-sin(β/2),  cos(β/2)]]        d^1_{01}(β) = +sin(β)/√2
```

These are also the conventions of the Wikipedia article on the Wigner D-matrix, of `spherical`, of
ntessore's `wigner` and of `sympy.physics.quantum.spin.Rotation.d` (all checked numerically).
`sympy.physics.wigner.wigner_d_small` (after Edmonds) instead returns the transpose, i.e. d(−β),
with rows ordered from *m* = +*j*. Useful identities:
d<sup>j</sup><sub>m′m</sub>(β) = (−1)<sup>m′−m</sup> d<sup>j</sup><sub>mm′</sub>(β) = d<sup>j</sup><sub>−m,−m′</sub>(β),
d(−β) = d(β)ᵀ, and d(β + 2π) = (−1)<sup>2j</sup> d(β).

## Benchmarks

<!-- PyPI cannot resolve relative image paths: switch these to absolute
     https://raw.githubusercontent.com/... URLs before publishing. -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/results/benchmark-dark.png">
  <img alt="Time per matrix and largest error versus spin j for fast-wigner-d and six other methods" src="benchmarks/results/benchmark.png">
</picture>

Every method computes the complete (2*j*+1) × (2*j*+1) matrix, converted to the conventions above.
Times are medians over β = 0.3, 1.1 and 2.6, excluding one-off setup (object construction and numba
compilation for `spherical`, the generator matrix for `expm`, prefactor tables for `eval_jacobi`). Errors
are the largest absolute deviation from mpmath evaluated with hundreds of digits. Each cell reads
*time · error*; **bold** marks an error above 10⁻⁸ or NaN.

Apple M5 laptop (4 performance + 6 efficiency cores), macOS 26.6, Python 3.14.7, NumPy 2.5.4
(Accelerate), SciPy 1.18.1, spherical 1.1.4, wigner 2025.1, sympy 1.14.0. fast-wigner-d uses all 10
cores unless marked "1 thread" (the other methods are single-threaded, except `expm`, which uses
multithreaded BLAS).

**Integer spin**

| j | fast-wigner-d | fast-wigner-d<br>1 thread | spherical | wigner | scipy expm | scipy<br>eval_jacobi | textbook<br>sum | sympy |
|---:|---|---|---|---|---|---|---|---|
| 1 | 1.1 µs · 1e-16 | 1.1 µs · 1e-16 | 1.4 µs · 3e-16 | 1.6 µs · 1e-16 | 1.6 µs · 2e-15 | 1.3 µs · 1e-16 | 3.2 µs · 1e-16 | 301 µs · 1e-16 |
| 5 | 1.4 µs · 4e-16 | 1.4 µs · 4e-16 | 2.1 µs · 5e-16 | 18 µs · 1e-15 | 4.3 µs · 3e-15 | 2.5 µs · 9e-16 | 62 µs · 9e-16 | 5.42 ms · 8e-16 |
| 10 | 2.3 µs · 8e-16 | 2.3 µs · 8e-16 | 4.1 µs · 6e-16 | 68 µs · 4e-15 | 16 µs · 7e-15 | 7.8 µs · 2e-15 | 336 µs · 7e-15 | 29.4 ms · 8e-15 |
| 50 | 114 µs · 1e-15 | 121 µs · 1e-15 | 100 µs · 1e-15 | 2 ms · **NaN** | 204 µs · 3e-13 | 502 µs · 1e-14 | 27.4 ms · **4e-04** | 5.67 s · **3e-04** |
| 100 | 279 µs · 3e-15 | 1.11 ms · 3e-15 | 574 µs · 3e-15 | 10.7 ms · **NaN** | 1.02 ms · 4e-13 | 4.39 ms · 2e-14 | overflow | 48 s · **5e-07** |
| 200 | 1.71 ms · 4e-15 | 10.6 ms · 4e-15 | 3.76 ms · 2e-15 | 66.7 ms · **NaN** | 5.83 ms · 5e-13 | 34.9 ms · 4e-14 | overflow | (too slow) |
| 500 | 27.7 ms · 8e-15 | 191 ms · 8e-15 | 51 ms · 4e-15 | 880 ms · **NaN** | 70.8 ms · 6e-14 | 580 ms · 1e-13 | overflow | (too slow) |
| 1000 | 225 ms · 1e-14 | 1.58 s · 1e-14 | 383 ms · 5e-15 | 6.54 s · **NaN** | 626 ms · 8e-14 | 4.7 s · **NaN** | overflow | (too slow) |
| 2000 | 1.76 s · 2e-14 | 12.9 s · 2e-14 | 76.5 s · **NaN** | 50.5 s · **NaN** | 4.79 s · 1e-13 | 38.1 s · **NaN** | overflow | (too slow) |

**Half-integer spin** (`spherical` and `wigner` do not support it)

| j | fast-wigner-d | fast-wigner-d<br>1 thread | scipy expm | scipy<br>eval_jacobi | textbook<br>sum | sympy |
|---:|---|---|---|---|---|---|
| 1/2 | 1.1 µs · 0 | 1.1 µs · 0 | 1.4 µs · 1e-16 | 1.3 µs · 0 | 1.3 µs · 0 | 140 µs · 0 |
| 3/2 | 1.1 µs · 1e-16 | 1.1 µs · 1e-16 | 1.6 µs · 2e-14 | 1.4 µs · 1e-16 | 5.6 µs · 1e-16 | 576 µs · 1e-16 |
| 9/2 | 1.3 µs · 4e-16 | 1.3 µs · 4e-16 | 3.3 µs · 4e-15 | 2.3 µs · 8e-16 | 49 µs · 4e-16 | 4.56 ms · 6e-16 |
| 19/2 | 2.2 µs · 7e-16 | 2.1 µs · 7e-16 | 15 µs · 6e-15 | 6.9 µs · 2e-15 | 294 µs · 9e-15 | 27.5 ms · 4e-15 |
| 99/2 | 114 µs · 2e-15 | 117 µs · 2e-15 | 199 µs · 3e-13 | 486 µs · 1e-14 | 26.7 ms · **2e-04** | 5.66 s · **3e-04** |
| 199/2 | 277 µs · 4e-15 | 1.1 ms · 4e-15 | 928 µs · 4e-13 | 4.32 ms · 3e-14 | overflow | 48.7 s · **6e-07** |
| 999/2 | 27.6 ms · 7e-15 | 191 ms · 7e-15 | 73.7 ms · 6e-14 | 579 ms · 7e-14 | overflow | (too slow) |
| 1999/2 | 224 ms · 2e-14 | 1.6 s · 2e-14 | 610 ms · 8e-14 | 4.72 s · **NaN** | overflow | (too slow) |
| 3999/2 | 1.82 s · 1e-14 | 12.9 s · 1e-14 | 4.7 s · 1e-13 | 38 s · **NaN** | overflow | (too slow) |

What this shows:

* **Small matrices** (*j* ≲ 10): fast-wigner-d is the fastest method tested, at about 1–2 µs per matrix.
* **Large integer ℓ**: on all cores fast-wigner-d is 1.7–2.2× faster than `spherical`, but on a single
  core `spherical`'s recursion is 2–4× faster (O(1) work per element instead of an O(*j*) recurrence) and
  up to 2× more accurate. At ℓ = 2000 `spherical` returned NaN and took over a minute, while
  fast-wigner-d took 1.8 s with errors of 2 × 10⁻¹⁴.
* **Half-integer spin**: fast-wigner-d is the fastest method at every *j* and at least as accurate as all
  the others. The generic matrix exponential is 1.3–3.3× slower and up to 100× less accurate; the other
  methods lose accuracy or fail by *j* ≈ 50 (textbook sum, sympy) or *j* ≈ 1000 (`eval_jacobi`).

Notes on the other methods:

* [`spherical`](https://github.com/moble/spherical): `Wigner(ℓ, ell_min=ℓ).d(exp(iβ))`, the
  Gumerov–Duraiswami recursion compiled with numba. Integer ℓ only.
* [`wigner`](https://github.com/ntessore/wigner): `wigner_dl` gives d<sup>ℓ</sup><sub>m₁m₂</sub> for a range of ℓ at
  fixed (m₁, m₂), which is its real use case, so the full matrix takes (2ℓ+1)² calls. Its starting value uses a
  32-bit integer binomial coefficient that overflows once max(|m₁|, |m₂|) ≥ 17, hence the NaNs from ℓ = 17 on;
  elements with smaller |m| are accurate.
* `scipy.linalg.expm(-1j * beta * J_y)`, written as a real matrix: the generic approach, also what QuTiP-style
  code does. Dense O(*n*³) linear algebra.
* `scipy.special.eval_jacobi`: the same Jacobi formula as fast-wigner-d, evaluated directly in float64.
  Intermediate factors overflow or underflow, giving inf · 0 = NaN from *j* ≈ 1000.
* Textbook sum: Wigner's explicit alternating sum in float64, as in many hand-written codes. Cancellation leaves
  4 × 10⁻⁴ errors at *j* = 50, and since 171! does not fit in a double it fails for 2*j* > 170.
* `sympy.physics.wigner.wigner_d_small` with a float angle: built from sympy expressions on every call and
  evaluated in 15-digit floating point, so it is slow and suffers the same cancellation.

Full tables, including every size measured, are in
[`benchmarks/results/results.md`](benchmarks/results/results.md). To reproduce:

```bash
pip install ".[benchmark]"
python benchmarks/run_benchmarks.py     # about 15 minutes; --quick for 2j <= 200
python benchmarks/plot_benchmarks.py
```

## How it works

For *m′* ≥ |*m*| (the other three quarters of the matrix follow by symmetry),

```text
d^j_{m'm}(β) = (-1)^(m'-m) sqrt[(j+m')! (j-m')! / ((j+m)! (j-m)!)]
               × sin(β/2)^(m'-m) cos(β/2)^(m'+m) P_{j-m'}^{(m'-m, m'+m)}(cos β),
```

with *P* a Jacobi polynomial. Each element is evaluated independently, so rounding errors do not
propagate across the matrix:

1. The factorial ratio and the powers of sin(β/2), cos(β/2) are combined in log space. The log-factorials
   are tabulated once per matrix in double-double arithmetic, so their difference keeps full precision
   even when ln((2j)!) ≈ 10⁴; with plain `lgamma` it would lose about four digits at *j* = 1000.
2. The polynomial comes from the standard three-term recurrence (DLMF §18.9), rescaled by exact powers
   of two whenever it exceeds 2¹⁰⁰.
3. Near β = 0 and β = π the polynomials depend on 1 ∓ cos β, which `cos(β)` carries with a large relative
   error; there the recurrence is fed 2 sin²(β/2) or 2 cos²(β/2) instead.
4. The signs of sin(β/2) and cos(β/2) are restored separately, so any real β is valid.
5. Rows are distributed over OpenMP threads when 2*j* ≥ 96; smaller matrices are computed serially,
   because starting threads would cost more than the work.

The cost is O(*j*³) per matrix: about *j*² independent elements, each with an O(*j*) recurrence.

Absolute errors grow slowly with *j*, from about 10⁻¹⁶ for small spins to 2 × 10⁻¹⁴ at *j* = 2000 for
generic angles. The weakest case is very small angles at large spin: for β = 10⁻³ and *j* = 1000 errors
reach about 10⁻¹², where `spherical` stays near 10⁻¹⁵.

## API

| Function | Returns |
|---|---|
| `wigner_d(j, beta, *, num_threads=None)` | real (2j+1, 2j+1) array; `beta.shape + (2j+1, 2j+1)` for array `beta` |
| `wigner_D(j, alpha, beta, gamma, *, num_threads=None)` | complex D-matrix; the angles broadcast |
| `get_wigner_d_matrix_optimized(j_times_2, beta, num_threads=0)` | the compiled kernel, taking 2j as an integer |
| `OPENMP_ENABLED` | whether this build uses OpenMP |

`num_threads=None` (or ≤ 0 for the kernel) uses the OpenMP default: `OMP_NUM_THREADS`, or all cores.
If you already parallelise over many angles yourself, pass `num_threads=1`.

## Development

```bash
pip install -e ".[test]"          # rebuild after editing the .pyx: pip install -e . again
pytest                            # ~180 tests against mpmath references, ~20 s
pip install -e ".[benchmark]"
python benchmarks/run_benchmarks.py && python benchmarks/plot_benchmarks.py
```

`.github/workflows/wheels.yml` builds wheels for Linux, macOS and Windows with
[cibuildwheel](https://cibuildwheel.pypa.io) (configuration in `pyproject.toml`), runs the tests on
each, and uploads them to PyPI when a GitHub release is published.

## License

MIT — see [LICENSE](LICENSE).
