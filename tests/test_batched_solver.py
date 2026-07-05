"""Tests for the batched steady-state solver."""

from __future__ import annotations

import numpy as np
import pytest

from numerics.core.backend import set_backend
from numerics.models.vdp_2mode import VdP2Mode
from numerics.solvers.batched import solve_steady_state_batched


def test_batched_vdp_2mode_four_branches():
    """The batched solver should find all four mathematical branches."""
    cupy = pytest.importorskip("cupy")
    set_backend("cupy")
    try:
        model = VdP2Mode()
        B = 1
        params = {
            "omega_a": np.full(B, 0.02),
            "omega_b": np.zeros(B),
            "gamma_a": np.full(B, 2.0),
            "gamma_b": np.full(B, 0.9),
            "Gamma": np.full(B, 0.01),
            "g": np.full(B, 0.5),
            "D": np.full(B, 1.0),
        }
        guesses = np.zeros((B, 4, 2, 2), dtype=complex)
        guesses[0, 0] = [[0.5, 0], [0, 2.5]]
        guesses[0, 1] = [[40, 0], [0, 51]]
        guesses[0, 2] = [[62, 0], [0, 51]]
        guesses[0, 3] = [[54, -75], [0, 59]]

        results = solve_steady_state_batched(
            model, params, guesses, max_iter=30, tol=1e-10
        )
        assert len(results) == B
        assert len(results[0]) == 4
        converged = [r for r in results[0] if r.success]
        assert len(converged) == 4
    finally:
        set_backend("numpy")
