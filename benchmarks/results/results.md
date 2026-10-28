# Benchmark results

Machine: Apple M5, 10 cores, macOS-26.6-arm64-arm-64bit-Mach-O. Python 3.14.7, NumPy BLAS accelerate, fast-wigner-d OpenMP True.

Versions: fast-wigner-d 0.1.0, numpy 2.5.4, scipy 1.18.1, sympy 1.14.0, spherical 1.1.4, wigner 2025.1, mpmath 1.3.0

## Time per full (2j+1)×(2j+1) matrix (median)

| j | fast-wigner-d | fast-wigner-d, 1 thread | spherical | wigner (ntessore) | scipy expm(-i beta J_y) | scipy eval_jacobi formula | textbook sum (float64) | sympy wigner_d_small |
|---|---|---|---|---|---|---|---|---|
| 1/2 | 1.08 µs | 1.08 µs | — | — | 1.37 µs | 1.25 µs | 1.33 µs | 140 µs |
| 1 | 1.12 µs | 1.08 µs | 1.42 µs | 1.58 µs | 1.62 µs | 1.29 µs | 3.17 µs | 301 µs |
| 3/2 | 1.13 µs | 1.12 µs | — | — | 1.58 µs | 1.35 µs | 5.58 µs | 576 µs |
| 2 | 1.17 µs | 1.17 µs | 1.58 µs | 4.04 µs | 2.04 µs | 1.38 µs | 9.37 µs | 829 µs |
| 9/2 | 1.33 µs | 1.29 µs | — | — | 3.29 µs | 2.29 µs | 49 µs | 4.56 ms |
| 5 | 1.37 µs | 1.37 µs | 2.12 µs | 18.4 µs | 4.25 µs | 2.54 µs | 62.3 µs | 5.42 ms |
| 19/2 | 2.17 µs | 2.12 µs | — | — | 14.7 µs | 6.87 µs | 294 µs | 27.5 ms |
| 10 | 2.29 µs | 2.33 µs | 4.13 µs | 68 µs | 16.3 µs | 7.79 µs | 336 µs | 29.4 ms |
| 39/2 | 7.87 µs | 7.67 µs | — | — | 37.5 µs | 35 µs | 1.96 ms | 260 ms |
| 20 | 8.42 µs | 8.33 µs | 13.4 µs | 272 µs | 45.5 µs | 37.2 µs | 2.09 ms | 256 ms |
| 99/2 | 114 µs | 117 µs | — | — | 199 µs | 486 µs | 26.7 ms | 5.66 s |
| 50 | 114 µs | 121 µs | 100 µs | 2 ms | 204 µs | 502 µs | 27.4 ms | 5.67 s |
| 199/2 | 277 µs | 1.1 ms | — | — | 928 µs | 4.32 ms | overflow | 48.7 s |
| 100 | 279 µs | 1.11 ms | 574 µs | 10.7 ms | 1.02 ms | 4.39 ms | — | 48 s |
| 399/2 | 1.68 ms | 10.6 ms | — | — | 5.27 ms | 34.6 ms | — | (too slow) |
| 200 | 1.71 ms | 10.6 ms | 3.76 ms | 66.7 ms | 5.83 ms | 34.9 ms | — | (too slow) |
| 999/2 | 27.6 ms | 191 ms | — | — | 73.7 ms | 579 ms | — | (too slow) |
| 500 | 27.7 ms | 191 ms | 51 ms | 880 ms | 70.8 ms | 580 ms | — | (too slow) |
| 1999/2 | 224 ms | 1.6 s | — | — | 610 ms | 4.72 s | — | (too slow) |
| 1000 | 225 ms | 1.58 s | 383 ms | 6.54 s | 626 ms | 4.7 s | — | (too slow) |
| 3999/2 | 1.82 s | 12.9 s | — | — | 4.7 s | 38 s | — | (too slow) |
| 2000 | 1.76 s | 12.9 s | 76.5 s | 50.5 s | 4.79 s | 38.1 s | — | (too slow) |

## Largest absolute error vs. mpmath (β = 0.3, 1.1, 2.6)

| j | fast-wigner-d | fast-wigner-d, 1 thread | spherical | wigner (ntessore) | scipy expm(-i beta J_y) | scipy eval_jacobi formula | textbook sum (float64) | sympy wigner_d_small |
|---|---|---|---|---|---|---|---|---|
| 1/2 | 0e+00 | 0e+00 | — | — | 1e-16 | 0e+00 | 0e+00 | 0e+00 |
| 1 | 1e-16 | 1e-16 | 3e-16 | 1e-16 | 2e-15 | 1e-16 | 1e-16 | 1e-16 |
| 3/2 | 1e-16 | 1e-16 | — | — | 2e-14 | 1e-16 | 1e-16 | 1e-16 |
| 2 | 2e-16 | 2e-16 | 3e-16 | 2e-16 | 3e-15 | 2e-16 | 3e-16 | 2e-16 |
| 9/2 | 4e-16 | 4e-16 | — | — | 4e-15 | 8e-16 | 4e-16 | 6e-16 |
| 5 | 4e-16 | 4e-16 | 5e-16 | 1e-15 | 3e-15 | 9e-16 | 9e-16 | 8e-16 |
| 19/2 | 7e-16 | 7e-16 | — | — | 6e-15 | 2e-15 | 9e-15 | 4e-15 |
| 10 | 8e-16 | 8e-16 | 6e-16 | 4e-15 | 7e-15 | 2e-15 | 7e-15 | 8e-15 |
| 39/2 | 1e-15 | 1e-15 | — | — | 1e-14 | 6e-15 | 3e-12 | 1e-12 |
| 20 | 1e-15 | 1e-15 | 9e-16 | NaN | 1e-14 | 6e-15 | 5e-12 | 3e-12 |
| 99/2 | 2e-15 | 2e-15 | — | — | 3e-13 | 1e-14 | 2e-04 | 3e-04 |
| 50 | 1e-15 | 1e-15 | 1e-15 | NaN | 3e-13 | 1e-14 | 4e-04 | 3e-04 |
| 199/2 | 4e-15 | 4e-15 | — | — | 4e-13 | 3e-14 | overflow | 6e-07 |
| 100 | 3e-15 | 3e-15 | 3e-15 | NaN | 4e-13 | 2e-14 | — | 5e-07 |
| 399/2 | 6e-15 | 6e-15 | — | — | 4e-13 | 3e-14 | — | (too slow) |
| 200 | 4e-15 | 4e-15 | 2e-15 | NaN | 5e-13 | 4e-14 | — | (too slow) |
| 999/2 | 7e-15 | 7e-15 | — | — | 6e-14 | 7e-14 | — | (too slow) |
| 500 | 8e-15 | 8e-15 | 4e-15 | NaN | 6e-14 | 1e-13 | — | (too slow) |
| 1999/2 | 2e-14 | 2e-14 | — | — | 8e-14 | NaN | — | (too slow) |
| 1000 | 1e-14 | 1e-14 | 5e-15 | NaN | 8e-14 | NaN | — | (too slow) |
| 3999/2 | 1e-14 | 1e-14 | — | — | 1e-13 | NaN | — | (too slow) |
| 2000 | 2e-14 | 2e-14 | NaN | NaN | 1e-13 | NaN | — | (too slow) |

— = not supported (half-integer j). "(too slow)": skipped, extrapolated time above 60 s per matrix.
