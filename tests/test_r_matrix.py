"""Tests for R-matrix parameterization utilities."""

import numpy as np
import pytest

from numerics.core.r_matrix import (
    R_matrix_to_vector,
    vector_to_R_matrix,
    get_parameterized_R,
    is_hermitian,
)


def test_round_trip():
    """R_matrix_to_vector and vector_to_R_matrix are inverses."""
    n = 4
    rng = np.random.default_rng(0)
    R = np.zeros((n, n), dtype=complex)
    for i in range(n):
        R[i, i] = rng.random()
    for i in range(n):
        for j in range(i + 1, n):
            re = rng.normal()
            im = rng.normal()
            R[i, j] = re + 1j * im
            R[j, i] = re - 1j * im

    vec = R_matrix_to_vector(R, n)
    R2 = vector_to_R_matrix(vec, n)
    assert np.allclose(R, R2)
    assert is_hermitian(R2)


def test_get_parameterized_R_shape():
    """Symbolic parameterization has the correct dimensions."""
    import sympy as sp
    R, r_vec = get_parameterized_R(3)
    assert R.shape == (3, 3)
    assert len(r_vec) == 9
    assert R.is_hermitian
