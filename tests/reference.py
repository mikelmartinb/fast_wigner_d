"""mpmath reference for d^j_{m'm}(beta), independent of the Jacobi method.

Wigner's explicit sum (Varshalovich et al., Ch. 4), with ~0.3 * 2j extra digits
to absorb its cancellation. Indices are i = m' + j, k = m + j:

    d^j_{m'm}(beta) = sqrt((j+m')! (j-m')! (j+m)! (j-m)!)
        * sum_s (-1)^(m'-m+s) cos(beta/2)^(2j+m-m'-2s) sin(beta/2)^(m'-m+2s)
                / ((j+m-s)! s! (m'-m+s)! (j-m'-s)!)
"""
import mpmath as mp
import numpy as np


def d_element_mp(two_j, i, k, beta, extra_digits=25):
    """d^j_{m'm}(beta) for j = two_j / 2, m' = i - j, m = k - j, as a float."""
    with mp.workdps(int(0.31 * two_j) + extra_digits):
        half = mp.mpf(beta) / 2
        c, s = mp.cos(half), mp.sin(half)
        f = mp.factorial
        t0, t1 = max(0, k - i), min(k, two_j - i)
        total = mp.mpf(0)
        for t in range(t0, t1 + 1):
            term = c ** (two_j + k - i - 2 * t) * s ** (i - k + 2 * t)
            term /= f(k - t) * f(t) * f(i - k + t) * f(two_j - i - t)
            total += -term if (i - k + t) % 2 else term
        return float(mp.sqrt(f(i) * f(two_j - i) * f(k) * f(two_j - k)) * total)


def d_element_mp_fast(two_j, i, k, beta, extra_digits=25):
    """As d_element_mp, building each term from the previous one (fast for large j)."""
    with mp.workdps(int(0.31 * two_j) + extra_digits):
        half = mp.mpf(beta) / 2
        c, s = mp.cos(half), mp.sin(half)
        if c == 0:
            return d_element_mp(two_j, i, k, beta, extra_digits)
        f = mp.factorial
        t0, t1 = max(0, k - i), min(k, two_j - i)
        term = c ** (two_j + k - i - 2 * t0) * s ** (i - k + 2 * t0)
        term /= f(k - t0) * f(t0) * f(i - k + t0) * f(two_j - i - t0)
        if (i - k + t0) % 2:
            term = -term
        total = term
        ratio = -(s / c) ** 2
        for t in range(t0, t1):
            term *= ratio * (k - t) * (two_j - i - t) / ((t + 1) * (i - k + t + 1))
            total += term
        return float(mp.sqrt(f(i) * f(two_j - i) * f(k) * f(two_j - k)) * total)


def d_matrix_mp(two_j, beta):
    """Full (2j+1, 2j+1) reference matrix, d[m' + j, m + j]."""
    n = two_j + 1
    return np.array([[d_element_mp(two_j, i, k, beta) for k in range(n)] for i in range(n)])
