"""Tests for deflation-based multi-root discovery."""

from __future__ import annotations

import numpy as np
import pytest

from numerics.models.vdp_2mode import VdP2Mode
from numerics.solvers.deflation import (
    DeflationOperator,
    DeflationOptions,
    find_roots_deflation,
)
from numerics.solvers.seeds import make_random_guesses
from numerics.core.r_matrix import R_matrix_to_vector


@pytest.fixture
def model_and_params():
    params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 1.0,
        "gamma_b": 1.1,
        "Gamma": 1.0,
        "g": 0.1,
        "D": 0.1,
    }
    return VdP2Mode(), params


def test_deflation_finds_all_roots(model_and_params):
    model, params = model_and_params
    guesses = make_random_guesses(dim=2, n_samples=50, scale=50.0, seed=1)
    roots = find_roots_deflation(
        model,
        params,
        guesses,
        opts=DeflationOptions(alpha=1e-2, residual_tol=1e-7, distance_tol=1.0),
    )
    assert len(roots) == 3
    for fp in roots:
        assert fp.residual <= 1e-7
        assert fp.omega is not None
        assert fp.J_eigvals is not None


def test_deflation_avoids_known_root(model_and_params):
    model, params = model_and_params
    # First discover the stable high-R11 root.
    guesses = make_random_guesses(dim=2, n_samples=50, scale=50.0, seed=1)
    all_roots = find_roots_deflation(
        model,
        params,
        guesses,
        opts=DeflationOptions(alpha=1e-2, residual_tol=1e-7, distance_tol=1.0),
    )
    stable = [fp for fp in all_roots if fp.is_stable]
    assert len(stable) == 1
    known = R_matrix_to_vector(stable[0].R)

    # Re-run with the stable root excluded.
    remaining = find_roots_deflation(
        model,
        params,
        guesses,
        known_roots=[known],
        opts=DeflationOptions(alpha=1e-2, residual_tol=1e-7, distance_tol=1.0),
    )
    assert len(remaining) == 2
    for fp in remaining:
        assert np.linalg.norm(fp.to_vector() - known) >= 1.0


def test_deflation_operator_pole_at_root():
    root = np.array([1.0, 2.0, 0.0, -0.5])
    op = DeflationOperator(known_roots=[root], alpha=1e-2)
    x_near = root + np.array([1e-4, 0.0, 0.0, 0.0])
    factor = op.factor(x_near)
    assert factor > 1e2
    # Far from the root the factor is close to one.
    x_far = root + np.array([10.0, 0.0, 0.0, 0.0])
    assert abs(op.factor(x_far) - 1.0) < 1e-3
