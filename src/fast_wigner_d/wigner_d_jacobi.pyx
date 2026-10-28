# wigner_d_jacobi.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True

cimport cython
from cython.parallel cimport prange
import numpy as np
from libc.math cimport sin, cos, log, exp, fabs, ceil

# Without OpenMP, prange runs serially and this stub is used
cdef extern from *:
    """
    #ifdef _OPENMP
    #include <omp.h>
    #define FWD_HAVE_OPENMP 1
    #else
    #define FWD_HAVE_OPENMP 0
    static int omp_get_max_threads(void) { return 1; }
    #endif
    """
    int FWD_HAVE_OPENMP
    int omp_get_max_threads() noexcept nogil

OPENMP_ENABLED = bool(FWD_HAVE_OPENMP)

# Below this 2j, starting threads costs more than the work
cdef int PARALLEL_MIN_2J = 96


cdef inline void _two_sum(double x, double y, double* s, double* e) noexcept nogil:
    # s + e == x + y exactly, with s = fl(x + y)
    cdef double v
    s[0] = x + y
    v = s[0] - x
    e[0] = (x - (s[0] - v)) + (y - v)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_wigner_d_matrix_optimized(int j_times_2, double beta, int num_threads=0):
    """
    Computes the Wigner d-matrix for j = j_times_2 / 2 using log-scaled
    Jacobi polynomials

    d[m' + j, m + j] = d^j_{m'm}(beta) = <j m'| exp(-i beta J_y) |j m>
    (Varshalovich conventions), for any real beta. num_threads <= 0: OpenMP default.
    """
    if j_times_2 < 0:
        raise ValueError("j_times_2 must be non-negative")

    cdef int size = j_times_2 + 1
    cdef double j = j_times_2 / 2.0

    cdef double[:, ::1] d_mat = np.zeros((size, size), dtype=np.float64)

    # ln(n!) + ln((2j - n)!) for n = 0..2j in double-double (hi + lo): the
    # prefactor is a difference of two of these large numbers
    cdef double[::1] lf_hi = np.empty(size, dtype=np.float64)
    cdef double[::1] lf_lo = np.empty(size, dtype=np.float64)
    cdef double[::1] lfs_hi = np.empty(size, dtype=np.float64)
    cdef double[::1] lfs_lo = np.empty(size, dtype=np.float64)
    cdef double dd_s, dd_e
    cdef int i

    lf_hi[0] = 0.0
    lf_lo[0] = 0.0
    for i in range(1, size):
        _two_sum(lf_hi[i - 1], log(<double>i), &dd_s, &dd_e)
        _two_sum(dd_s, dd_e + lf_lo[i - 1], &lf_hi[i], &lf_lo[i])
    for i in range(size):
        _two_sum(lf_hi[i], lf_hi[size - 1 - i], &dd_s, &dd_e)
        _two_sum(dd_s, dd_e + (lf_lo[i] + lf_lo[size - 1 - i]), &lfs_hi[i], &lfs_lo[i])

    # Precompute trigonometry and logs
    cdef double half_beta = 0.5 * beta
    cdef double sin_hb = sin(half_beta)
    cdef double cos_hb = cos(half_beta)

    # Powers of |sin| and |cos| in log space, with their signs restored below
    cdef bint sin_neg = sin_hb < 0.0
    cdef bint cos_neg = cos_hb < 0.0
    cdef double log_sin = -1.0e30
    cdef double log_cos = -1.0e30
    if sin_hb != 0.0:
        log_sin = log(fabs(sin_hb))
    if cos_hb != 0.0:
        log_cos = log(fabs(cos_hb))

    cdef double cos_b = cos(beta)

    # x = cos(beta) = x_sigma + x_w. Near beta = 0 (pi) the polynomials depend on
    # 1 - x (1 + x), which cos(beta) resolves poorly: use x_sigma = 1 (-1) and
    # x_w = -2 sin^2(beta/2) (2 cos^2(beta/2))
    cdef double x_sigma = 0.0
    cdef double x_w = cos_b
    if cos_b > 0.75:
        x_sigma = 1.0
        x_w = -2.0 * sin_hb * sin_hb
    elif cos_b < -0.75:
        x_sigma = -1.0
        x_w = 2.0 * cos_hb * cos_hb

    # Rescaling constants (powers of two, so rescaling is exact)
    cdef double HUGE_VAL = 1.2676506002282294e30   # 2**100
    cdef double TINY_VAL = 7.888609052210118e-31   # 2**-100
    cdef double LOG_HUGE_VAL = 69.31471805599453   # log(2**100)
    cdef double LOG_UNDERFLOW = -745.0             # exp() underflows below this

    cdef int start_r = <int>ceil(j)

    cdef int r, c, c_start, c_end, n, n_rescale
    cdef double mp, m, k, a, b
    cdef double log_A, log_ang, total_log
    cdef double poly, p_nm1, p_nm2, n_f, ab_n, ab_2n, c1, c2, c3
    cdef double val, sign

    if num_threads <= 0:
        num_threads = omp_get_max_threads()

    for r in prange(start_r, size, nogil=True, schedule='dynamic',
                    num_threads=num_threads,
                    use_threads_if=j_times_2 >= PARALLEL_MIN_2J):
        mp = r - j
        k = j - mp

        if k < 0:
            continue

        c_start = <int>(2.0 * j - r)
        c_end = r + 1

        for c in range(c_start, c_end):
            m = c - j
            a = mp - m
            b = mp + m

            # Prefactor in log space: k!(k+a+b)!/((k+a)!(k+b)!) = (2j-r)! r!/((2j-c)! c!)
            log_A = 0.5 * ((lfs_hi[r] - lfs_hi[c]) + (lfs_lo[r] - lfs_lo[c]))

            log_ang = (a * log_sin) + (b * log_cos)
            n_rescale = 0

            # 3-term recurrence for Jacobi polynomials
            if k == 0:
                poly = 1.0
            elif k == 1:
                poly = 0.5 * (((a - b) + x_sigma * (a + b + 2.0)) + x_w * (a + b + 2.0))
            else:
                p_nm2 = 1.0
                p_nm1 = 0.5 * (((a - b) + x_sigma * (a + b + 2.0)) + x_w * (a + b + 2.0))
                poly = 0.0

                for n in range(2, <int>k + 1):
                    n_f = <double>n
                    ab_n = a + b + n_f
                    ab_2n = a + b + 2.0 * n_f

                    c1 = 2.0 * n_f * (ab_n) * (ab_2n - 2.0)
                    c2 = ab_2n * (ab_2n - 2.0)
                    c2 = (ab_2n - 1.0) * (((a * a - b * b) + x_sigma * c2) + x_w * c2)
                    c3 = 2.0 * (n_f + a - 1.0) * (n_f + b - 1.0) * ab_2n

                    poly = (c2 * p_nm1 - c3 * p_nm2) / c1
                    p_nm2 = p_nm1
                    p_nm1 = poly

                    if fabs(poly) > HUGE_VAL:
                        poly = poly * TINY_VAL
                        p_nm1 = p_nm1 * TINY_VAL
                        p_nm2 = p_nm2 * TINY_VAL
                        n_rescale = n_rescale + 1

            total_log = log_A + log_ang + n_rescale * LOG_HUGE_VAL

            if total_log < LOG_UNDERFLOW:
                val = 0.0
            else:
                if m >= mp:
                    sign = 1.0
                else:
                    sign = -1.0 if (<int>a % 2 != 0) else 1.0
                # signs of sin(beta/2)^a and cos(beta/2)^b
                if sin_neg and (<int>a % 2 != 0):
                    sign = -sign
                if cos_neg and (<int>b % 2 != 0):
                    sign = -sign
                val = sign * exp(total_log) * poly

            d_mat[r, c] = val

            # Populate symmetry quadrants
            sign = 1.0 if ((r - c) % 2 == 0) else -1.0
            d_mat[c, r] = sign * val
            d_mat[size - 1 - r, size - 1 - c] = sign * val
            d_mat[size - 1 - c, size - 1 - r] = val

    return np.asarray(d_mat)
