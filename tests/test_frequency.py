"""Tests for frequency extraction."""

import numpy as np

from numerics.core.frequency import omega_frequency, omega_from_H_eigvals


def test_omega_rank_one():
    """For rank-one R, omega = Re Tr(HR)/Tr(R) equals the active eigenvalue."""
    H = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=complex)
    alpha = np.array([1.0 + 0.5j, 0.3 - 0.2j])
    R = np.outer(alpha, alpha.conj())
    omega = omega_frequency(H, R)
    # For a rank-one state the expectation value is the Rayleigh quotient
    expected = float(np.real(alpha.conj() @ H @ alpha) / (alpha.conj() @ alpha))
    assert np.isclose(omega, expected)


def test_omega_from_H_eigvals():
    """omega_from_H_eigvals returns real parts of eigenvalues."""
    H = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=complex)
    vals = omega_from_H_eigvals(H)
    expected = np.real(np.linalg.eigvals(H))
    assert np.allclose(np.sort(vals), np.sort(expected))
