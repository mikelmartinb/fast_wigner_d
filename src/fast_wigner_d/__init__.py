"""Wigner d- and D-matrices for integer and half-integer spin.

Conventions of Varshalovich, Moskalev & Khersonskii, *Quantum Theory of Angular
Momentum* (1988), Ch. 4 (active rotations, z-y-z Euler angles):

    D^j_{m'm}(alpha, beta, gamma) = exp(-i m' alpha) d^j_{m'm}(beta) exp(-i m gamma),
    d^j_{m'm}(beta) = <j m'| exp(-i beta J_y) |j m>,

with matrices indexed as d[m' + j, m + j].
"""
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .wigner_d_jacobi import OPENMP_ENABLED, get_wigner_d_matrix_optimized

try:
    __version__ = version("fast-wigner-d")
except PackageNotFoundError:  # source tree, not installed
    __version__ = "unknown"

__all__ = [
    "OPENMP_ENABLED",
    "get_wigner_d_matrix_optimized",
    "wigner_D",
    "wigner_d",
]


def _two_j(j):
    """2j as an int, for j a non-negative integer or half-integer."""
    try:
        if isinstance(j, (str, bytes)):
            raise TypeError
        two_j = float(2 * j)
    except (TypeError, ValueError):
        raise TypeError(f"j must be a real number, got {j!r}") from None
    n = int(round(two_j))
    if n < 0 or abs(two_j - n) > 1e-9:
        raise ValueError(f"j must be a non-negative integer or half-integer, got {j!r}")
    return n


def wigner_d(j, beta, *, num_threads=None):
    """Wigner small-d matrix d^j(beta).

    Parameters
    ----------
    j : int, float or fractions.Fraction
        Non-negative integer or half-integer.
    beta : float or array_like
        Angle(s) in radians, any real value.
    num_threads : int, optional
        OpenMP threads per matrix (default: the OpenMP default). Matrices with
        2j < 96 always run serially.

    Returns
    -------
    ndarray, shape (2j+1, 2j+1) or beta.shape + (2j+1, 2j+1)
        d[..., m' + j, m + j] = d^j_{m'm}(beta).

    Examples
    --------
    >>> wigner_d(0.5, np.pi / 2).round(3)
    array([[ 0.707,  0.707],
           [-0.707,  0.707]])
    """
    two_j = _two_j(j)
    threads = 0 if num_threads is None else int(num_threads)
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim == 0:
        return get_wigner_d_matrix_optimized(two_j, float(beta), threads)
    out = np.empty(beta.shape + (two_j + 1, two_j + 1))
    for idx, b in np.ndenumerate(beta):
        out[idx] = get_wigner_d_matrix_optimized(two_j, float(b), threads)
    return out


def wigner_D(j, alpha, beta, gamma, *, num_threads=None):
    """Wigner D-matrix for z-y-z Euler angles in radians (the angles broadcast).

    Returns complex128 D[..., m' + j, m + j] = exp(-i m' alpha) d^j_{m'm}(beta)
    exp(-i m gamma). Other arguments are as in wigner_d. For half-integer j, D
    changes sign under a 2 pi rotation.
    """
    two_j = _two_j(j)
    alpha, beta, gamma = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (alpha, beta, gamma))
    )
    m = np.arange(-two_j, two_j + 1, 2) / 2.0
    d = wigner_d(two_j / 2, beta, num_threads=num_threads)
    left = np.exp(-1j * m * alpha[..., None])
    right = np.exp(-1j * m * gamma[..., None])
    return left[..., :, None] * d * right[..., None, :]
