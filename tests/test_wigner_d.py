import math
from fractions import Fraction

import numpy as np
import pytest

from fast_wigner_d import get_wigner_d_matrix_optimized, wigner_D, wigner_d
from reference import d_element_mp, d_element_mp_fast, d_matrix_mp

# includes beta = 0 and pi exactly, tiny angles, negative angles and angles > 2 pi
BETAS = [0.0, 1e-300, 1e-12, 0.3, 1.0, np.pi / 2, 2.0, np.pi - 1e-9, np.pi,
         3.5, -0.7, -3.0, 2 * np.pi, 7.5, 40.0]


@pytest.mark.parametrize("two_j", [0, 1, 2, 3, 4, 7, 10, 21, 32])
@pytest.mark.parametrize("beta", BETAS)
def test_matches_mpmath_reference(two_j, beta):
    d = get_wigner_d_matrix_optimized(two_j, beta)
    np.testing.assert_allclose(d, d_matrix_mp(two_j, beta), rtol=0, atol=1e-14)


def test_reference_implementations_agree():
    for two_j, i, k, beta in [(41, 3, 30, 0.7), (40, 20, 20, 2.9), (17, 0, 17, -1.2), (9, 9, 0, 0.0)]:
        assert abs(d_element_mp_fast(two_j, i, k, beta) - d_element_mp(two_j, i, k, beta)) < 1e-16


@pytest.mark.parametrize("two_j", [301, 1000, 2001])
@pytest.mark.parametrize("beta", [1e-3, 0.4, 1.7, 2.9, -2.2])
def test_large_j_against_reference(two_j, beta):
    rng = np.random.default_rng(two_j)
    idx = rng.integers(0, two_j + 1, size=(40, 2))
    d = get_wigner_d_matrix_optimized(two_j, beta)
    ref = [d_element_mp_fast(two_j, i, k, beta) for i, k in idx]
    np.testing.assert_allclose(d[idx[:, 0], idx[:, 1]], ref, rtol=0, atol=5e-12)


def test_closed_forms():
    # closed forms for j = 1/2 and 1 (Varshalovich et al., Ch. 4) and d^{3/2}_{3/2, m}
    b = 0.83
    c, s = math.cos(b / 2), math.sin(b / 2)
    np.testing.assert_allclose(wigner_d(0.5, b), [[c, s], [-s, c]], rtol=0, atol=1e-15)

    cb, sb = math.cos(b), math.sin(b) / math.sqrt(2)
    expected = [[(1 + cb) / 2, sb, (1 - cb) / 2],
                [-sb, cb, sb],
                [(1 - cb) / 2, -sb, (1 + cb) / 2]]
    np.testing.assert_allclose(wigner_d(1, b), expected, rtol=0, atol=1e-15)

    r3 = math.sqrt(3)
    top_row = [-s ** 3, r3 * c * s ** 2, -r3 * c ** 2 * s, c ** 3]  # m = -3/2 .. 3/2
    np.testing.assert_allclose(wigner_d(1.5, b)[3], top_row, rtol=0, atol=1e-15)


@pytest.mark.parametrize("two_j", [0, 1, 6, 99, 600])
def test_group_property(two_j):
    b1, b2 = 0.7, -2.1
    np.testing.assert_allclose(
        wigner_d(two_j / 2, b1) @ wigner_d(two_j / 2, b2),
        wigner_d(two_j / 2, b1 + b2), rtol=0, atol=1e-12)


@pytest.mark.parametrize("two_j", [100, 1001, 4001])
@pytest.mark.parametrize("beta", [1e-4, 0.9, 2.5])
def test_orthogonality_large_j(two_j, beta):
    d = get_wigner_d_matrix_optimized(two_j, beta)
    assert np.isfinite(d).all()
    assert np.abs(d @ d.T - np.eye(two_j + 1)).max() < 1e-10


@pytest.mark.parametrize("two_j", [1, 4, 13, 50])
def test_symmetries(two_j):
    j = two_j / 2
    b = 1.234
    d = wigner_d(j, b)
    np.testing.assert_allclose(wigner_d(j, -b), d.T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(wigner_d(j, b + 2 * np.pi), (-1) ** two_j * d, rtol=0, atol=1e-13)
    # d^j_{m'm}(pi) = (-1)^(j-m) delta_{m',-m}
    k = np.arange(two_j + 1)
    expected = np.zeros((two_j + 1, two_j + 1))
    expected[two_j - k, k] = (-1.0) ** (two_j - k)
    np.testing.assert_allclose(wigner_d(j, np.pi), expected, rtol=0, atol=1e-13)
    assert np.array_equal(wigner_d(j, 0.0), np.eye(two_j + 1))


def test_wigner_D_spin_one_is_the_rotation_matrix():
    # D^1_{m'm} = e_{m'}^* . R e_m, R = Rz(alpha) Ry(beta) Rz(gamma), e_{+-1} = -+(x +- iy)/sqrt2, e_0 = z
    def rz(t):
        return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])

    def ry(t):
        return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    alpha, beta, gamma = 0.4, 1.3, -2.2
    rot = rz(alpha) @ ry(beta) @ rz(gamma)
    basis = np.array([[1, -1j, 0], [0, 0, np.sqrt(2)], [-1, -1j, 0]]) / np.sqrt(2)  # m = -1, 0, 1
    np.testing.assert_allclose(wigner_D(1, alpha, beta, gamma), basis.conj() @ rot @ basis.T,
                               rtol=0, atol=1e-15)


def test_wigner_D_half_integer():
    D = wigner_D(Fraction(5, 2), 0.3, 1.1, -0.8)
    assert D.shape == (6, 6) and D.dtype == np.complex128
    np.testing.assert_allclose(D @ D.conj().T, np.eye(6), rtol=0, atol=1e-14)
    np.testing.assert_allclose(wigner_D(2.5, 0.3 + 2 * np.pi, 1.1, -0.8), -D, rtol=0, atol=1e-14)


def test_broadcasting_and_shapes():
    betas = np.linspace(-1, 4, 6).reshape(2, 3)
    d = wigner_d(1.5, betas)
    assert d.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(d[1, 2], wigner_d(1.5, betas[1, 2]))
    D = wigner_D(1, np.array([0.0, 0.5]), 1.0, 0.2)
    assert D.shape == (2, 3, 3)
    np.testing.assert_array_equal(D[1], wigner_D(1, 0.5, 1.0, 0.2))


def test_num_threads_gives_identical_results():
    d1 = wigner_d(300, 1.1, num_threads=1)
    assert np.array_equal(d1, wigner_d(300, 1.1, num_threads=3))
    assert np.array_equal(d1, wigner_d(300, 1.1))


@pytest.mark.parametrize("j", [Fraction(7, 2), 3.5, np.float64(3.5), 7 / 2])
def test_accepts_half_integers(j):
    assert np.array_equal(wigner_d(j, 0.5), get_wigner_d_matrix_optimized(7, 0.5))


@pytest.mark.parametrize("j, error", [(1 / 3, ValueError), (-1, ValueError), (0.25, ValueError),
                                      ("1", TypeError), (None, TypeError)])
def test_rejects_invalid_j(j, error):
    with pytest.raises(error):
        wigner_d(j, 0.1)


def test_negative_two_j_raises():
    with pytest.raises(ValueError):
        get_wigner_d_matrix_optimized(-1, 0.1)
