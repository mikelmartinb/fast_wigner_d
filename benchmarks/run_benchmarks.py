"""Benchmark fast-wigner-d against other ways of computing full Wigner d-matrices.

Each method returns d^j(beta) in Varshalovich conventions. Recorded per method and
j: the median time per matrix (one-off setup excluded), the largest error against
mpmath (tests/reference.py) over beta = 0.3, 1.1, 2.6, and max |d d^T - 1|. The
error uses all elements for 2j <= 40, else 150 random ones.

    python benchmarks/run_benchmarks.py [--quick]   # ~15 min (--quick: 2j <= 200)
    python benchmarks/plot_benchmarks.py
"""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))
from reference import d_element_mp_fast, d_matrix_mp  # noqa: E402

BETAS = (0.3, 1.1, 2.6)
TWO_J = [1, 2, 3, 4, 9, 10, 19, 20, 39, 40, 99, 100, 199, 200,
         399, 400, 999, 1000, 1999, 2000, 3999, 4000]
QUICK_MAX_TWO_J = 200
FULL_REFERENCE_MAX_TWO_J = 40
N_SAMPLE = 150
CALL_BUDGET = 30.0     # s: after a call this slow, skip the remaining angles
PREDICT_BUDGET = 60.0  # s: skip sizes whose extrapolated time per call exceeds this
MIN_TIMING = 0.2       # s: repeat fast calls until at least this much time is spent


# --- methods: setup(two_j) -> f(beta) returning d[m' + j, m + j] ------------------

def fast_wigner_d(num_threads):
    from fast_wigner_d import get_wigner_d_matrix_optimized

    def setup(two_j):
        return lambda beta: get_wigner_d_matrix_optimized(two_j, beta, num_threads)
    return setup


def spherical_setup(two_j):
    import spherical

    ell, n = two_j // 2, two_j + 1
    wig = spherical.Wigner(ell, ell_min=ell)

    def f(beta):
        return wig.d(np.exp(1j * beta)).reshape(n, n)
    f(0.5)  # numba compilation
    return f


def wigner_ntessore_setup(two_j):
    import wigner

    ell = two_j // 2
    ms = range(-ell, ell + 1)

    def f(beta):  # the API gives d^l_{m1 m2} for a range of l at fixed (m1, m2)
        return np.array([[wigner.wigner_dl(ell, ell, m1, m2, beta)[0] for m2 in ms]
                         for m1 in ms])
    return f


def expm_setup(two_j):
    from scipy.linalg import expm

    # -i J_y is real: <m+1|-i J_y|m> = -sqrt((j-m)(j+m+1))/2, index i = m + j
    i = np.arange(two_j)
    off = 0.5 * np.sqrt((two_j - i) * (i + 1.0))
    gen = np.zeros((two_j + 1, two_j + 1))
    gen[i + 1, i] = -off
    gen[i, i + 1] = off
    return lambda beta: expm(beta * gen)


def eval_jacobi_setup(two_j):
    from scipy.special import eval_jacobi, gammaln

    # Jacobi form with k = min(j+m, j-m, j+m', j-m') (four cases), in float64
    i, k = np.indices((two_j + 1, two_j + 1))  # i = m' + j, k = m + j
    kmin = np.minimum.reduce([k, two_j - k, i, two_j - i])
    cases = [kmin == k, kmin == two_j - k, kmin == i]
    a = np.select(cases, [i - k, k - i, k - i], i - k)
    lam = np.select(cases, [i - k, 0, 0], i - k)
    b = two_j - 2 * kmin - a
    pref = (-1.0) ** lam * np.exp(0.5 * (gammaln(kmin + a + b + 1) + gammaln(kmin + 1)
                                          - gammaln(kmin + a + 1) - gammaln(kmin + b + 1)))
    a, b = a.astype(float), b.astype(float)

    def f(beta):
        c, s = math.cos(beta / 2), math.sin(beta / 2)
        return pref * s ** a * c ** b * eval_jacobi(kmin, a, b, math.cos(beta))
    return f


def textbook_setup(two_j):
    # Wigner's explicit sum with float64 factorials, as in many hand-written codes
    fact = [float(math.factorial(n)) for n in range(two_j + 1)]  # OverflowError beyond 170!
    n = two_j + 1

    def f(beta):
        c, s = math.cos(beta / 2), math.sin(beta / 2)
        d = np.empty((n, n))
        for i in range(n):
            for k in range(n):
                total = 0.0
                for t in range(max(0, k - i), min(k, two_j - i) + 1):
                    term = (c ** (two_j + k - i - 2 * t) * s ** (i - k + 2 * t)
                            / (fact[k - t] * fact[t] * fact[i - k + t] * fact[two_j - i - t]))
                    total += -term if (i - k + t) % 2 else term
                d[i, k] = math.sqrt(fact[i] * fact[two_j - i] * fact[k] * fact[two_j - k]) * total
        return d
    return f


def sympy_setup(two_j):
    import sympy
    from sympy.physics.wigner import wigner_d_small

    spin = sympy.Rational(two_j, 2)

    def f(beta):  # rows/columns ordered m = j..-j, and transposed (Edmonds) convention
        mat = wigner_d_small(spin, sympy.Float(beta)).evalf()
        return np.array(mat.tolist(), dtype=complex).real[::-1, ::-1].T
    return f


METHODS = {
    "fast_wigner_d": dict(
        label="fast-wigner-d", package="fast-wigner-d", half_integer=True,
        threads="all cores (OpenMP)", setup=fast_wigner_d(0)),
    "fast_wigner_d_1thread": dict(
        label="fast-wigner-d, 1 thread", package="fast-wigner-d", half_integer=True,
        threads="1", setup=fast_wigner_d(1)),
    "spherical": dict(
        label="spherical", package="spherical", half_integer=False,
        threads="1", setup=spherical_setup),
    "wigner": dict(
        label="wigner (ntessore)", package="wigner", half_integer=False,
        threads="1", setup=wigner_ntessore_setup),
    "expm": dict(
        label="scipy expm(-i beta J_y)", package="scipy", half_integer=True,
        threads="BLAS threads", setup=expm_setup),
    "eval_jacobi": dict(
        label="scipy eval_jacobi formula", package="scipy", half_integer=True,
        threads="1", setup=eval_jacobi_setup),
    "textbook": dict(
        label="textbook sum (float64)", package=None, half_integer=True,
        threads="1", setup=textbook_setup),
    "sympy": dict(
        label="sympy wigner_d_small", package="sympy", half_integer=True,
        threads="1", setup=sympy_setup),
}


# --- reference values ---------------------------------------------------------

_reference_cache = {}


def reference(two_j, beta):
    """(indices or None, reference values) for this size and angle."""
    key = (two_j, beta)
    if key not in _reference_cache:
        if two_j <= FULL_REFERENCE_MAX_TWO_J:
            _reference_cache[key] = (None, d_matrix_mp(two_j, beta))
        else:
            rng = np.random.default_rng(two_j)
            idx = rng.integers(0, two_j + 1, size=(N_SAMPLE, 2))
            near = N_SAMPLE // 3  # the large elements cluster around the diagonals
            idx[:near, 1] = np.clip(idx[:near, 0] + rng.integers(-8, 9, near), 0, two_j)
            ref = np.array([d_element_mp_fast(two_j, i, k, beta) for i, k in idx])
            _reference_cache[key] = (idx, ref)
    return _reference_cache[key]


def max_error(d, two_j, beta):
    idx, ref = reference(two_j, beta)
    got = d if idx is None else d[idx[:, 0], idx[:, 1]]
    err = np.abs(got - ref)
    return float("nan") if np.isnan(err).any() else float(err.max())


# --- driver -------------------------------------------------------------------

def run_one(f, beta):
    t0 = time.perf_counter()
    d = f(beta)
    times = [time.perf_counter() - t0]
    while sum(times) < MIN_TIMING and len(times) < 200:
        t0 = time.perf_counter()
        f(beta)
        times.append(time.perf_counter() - t0)
    return d, times


def benchmark(method_names, sizes):
    results, last = [], {}
    for two_j in sizes:
        for name in method_names:
            spec = METHODS[name]
            row = dict(method=name, two_j=two_j)
            if two_j % 2 and not spec["half_integer"]:
                continue
            if name in last and last[name][1] is None:
                continue  # failed at a smaller size
            if name in last:
                prev_two_j, prev_time = last[name]
                predicted = prev_time * (two_j / prev_two_j) ** 3
                if predicted > PREDICT_BUDGET:
                    results.append(dict(row, status="skipped",
                                        note=f"predicted {predicted:.0f} s per matrix"))
                    print(f"  2j={two_j:5d} {name:22s} skipped (predicted {predicted:.0f} s)")
                    continue
            try:
                f = spec["setup"](two_j)
                times, errors, orth = [], [], []
                for beta in BETAS:
                    d, t = run_one(f, beta)
                    times += t
                    errors.append(max_error(d, two_j, beta))
                    with np.errstate(all="ignore"):
                        orth.append(float(np.abs(d @ d.T - np.eye(two_j + 1)).max()))
                    if t[0] > CALL_BUDGET:
                        break
            except (OverflowError, MemoryError, ValueError) as exc:
                results.append(dict(row, status="failed", note=f"{type(exc).__name__}: {exc}"))
                last[name] = (two_j, None)
                print(f"  2j={two_j:5d} {name:22s} FAILED: {type(exc).__name__}: {exc}")
                continue
            nan = lambda xs: float("nan") if any(map(math.isnan, xs)) else max(xs)  # noqa: E731
            row.update(status="ok", time=statistics.median(times), n_calls=len(times),
                       n_angles=len(errors), error=nan(errors), orthogonality=nan(orth))
            results.append(row)
            last[name] = (two_j, row["time"])
            print(f"  2j={two_j:5d} {name:22s} {row['time']:10.3e} s   "
                  f"error {row['error']:.1e}   |dd^T-1| {row['orthogonality']:.1e}", flush=True)
    return results


def machine_info():
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True).strip()
        elif sys.platform.startswith("linux"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        blas = "unknown"
    import fast_wigner_d

    return dict(cpu=cpu, cores=os.cpu_count(), os=platform.platform(),
                python=platform.python_version(), numpy_blas=blas,
                openmp=fast_wigner_d.OPENMP_ENABLED,
                omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"))


def versions():
    out = {}
    for pkg in ["fast-wigner-d", "numpy", "scipy", "sympy", "spherical", "wigner", "mpmath"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def fmt_time(t):
    if t < 1e-3:
        return f"{t * 1e6:.3g} µs"
    if t < 1:
        return f"{t * 1e3:.3g} ms"
    return f"{t:.3g} s"


def markdown(data):
    names = [n for n in METHODS if any(r["method"] == n for r in data["results"])]
    sizes = sorted({r["two_j"] for r in data["results"]})
    cell = {(r["method"], r["two_j"]): r for r in data["results"]}

    def table(value):
        head = "| j | " + " | ".join(METHODS[n]["label"] for n in names) + " |"
        lines = [head, "|" + "---|" * (len(names) + 1)]
        for two_j in sizes:
            j = f"{two_j // 2}" if two_j % 2 == 0 else f"{two_j}/2"
            row = [value(cell.get((n, two_j))) for n in names]
            lines.append(f"| {j} | " + " | ".join(row) + " |")
        return "\n".join(lines)

    def time_cell(r):
        if r is None:
            return "—"
        if r["status"] != "ok":
            return "overflow" if "Overflow" in r.get("note", "") else "(too slow)"
        return fmt_time(r["time"])

    def error_cell(r):
        if r is None or r["status"] != "ok":
            return time_cell(r)
        return "NaN" if math.isnan(r["error"]) else f"{r['error']:.0e}"

    m = data["machine"]
    return "\n\n".join([
        "# Benchmark results",
        f"Machine: {m['cpu']}, {m['cores']} cores, {m['os']}. Python {m['python']}, "
        f"NumPy BLAS {m['numpy_blas']}, fast-wigner-d OpenMP {m['openmp']}.",
        "Versions: " + ", ".join(f"{k} {v}" for k, v in data["versions"].items() if v),
        "## Time per full (2j+1)×(2j+1) matrix (median)",
        table(time_cell),
        "## Largest absolute error vs. mpmath (β = 0.3, 1.1, 2.6)",
        table(error_cell),
        "— = not supported (half-integer j). \"(too slow)\": skipped, extrapolated time "
        f"above {PREDICT_BUDGET:.0f} s per matrix.",
    ]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help=f"only 2j <= {QUICK_MAX_TWO_J}")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    sizes = [n for n in TWO_J if not args.quick or n <= QUICK_MAX_TWO_J]
    data = dict(machine=machine_info(), versions=versions(),
                methods={n: {k: v for k, v in METHODS[n].items() if k != "setup"}
                         for n in args.methods})
    print(json.dumps(data["machine"], indent=1))
    data["results"] = benchmark(args.methods, sizes)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(data, indent=1))
    (args.out / "results.md").write_text(markdown(data))
    print(f"wrote {args.out / 'results.json'} and {args.out / 'results.md'}")


if __name__ == "__main__":
    main()
