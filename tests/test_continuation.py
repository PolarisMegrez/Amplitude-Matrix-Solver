"""Tests for pseudo-arclength continuation."""

from __future__ import annotations

import numpy as np
import pytest

from numerics.models.vdp_2mode import VdP2Mode
from numerics.solvers.continuation import (
    ContinuationOptions,
    trace_branch_pseudo_arclength,
)
from numerics.solvers.multi_search import find_steady_states
from numerics.core.r_matrix import R_matrix_to_vector


@pytest.fixture
def model_and_params():
    params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 1.0,
        "gamma_b": 1.0,
        "Gamma": 1.0,
        "g": 0.1,
        "D": 0.1,
    }
    return VdP2Mode(), params


def test_trace_branch_along_gamma_b(model_and_params):
    model, params = model_and_params
    roots = find_steady_states(
        model, params, n_samples=300, scale=50.0, seed=2,
        residual_tol=1e-6, distance_tol=1.0,
    )
    big = max(roots, key=lambda r: r.R[0, 0].real)
    x0 = R_matrix_to_vector(big.R)

    branch = trace_branch_pseudo_arclength(
        model, params, "gamma_b", x0=x0, lam0=1.0, p_range=[1.0, 1.2]
    )
    assert len(branch) >= 10
    assert branch.lambdas.min() >= 0.999
    assert branch.lambdas.max() <= 1.201
    assert np.all(branch.residuals <= 1e-8)
    # No large jumps in omega along the branch.
    assert np.max(np.abs(np.diff(branch.omegas))) < 0.1


def test_trace_branch_across_omega_a(model_and_params):
    model, params = model_and_params
    # Start from a parameter where the branch exists on both sides.
    params["omega_a"] = 0.05
    params["gamma_b"] = 1.1
    roots = find_steady_states(
        model, params, n_samples=300, scale=50.0, seed=3,
        residual_tol=1e-6, distance_tol=1.0,
    )
    big = max(roots, key=lambda r: r.R[0, 0].real)
    x0 = R_matrix_to_vector(big.R)

    branch = trace_branch_pseudo_arclength(
        model, params, "omega_a", x0=x0, lam0=0.05, p_range=[-0.1, 0.1],
        opts=ContinuationOptions(ds=0.002, ds_max=0.02),
    )
    assert len(branch) >= 10
    assert branch.lambdas.min() <= -0.05
    assert branch.lambdas.max() >= 0.05
    assert np.all(branch.residuals <= 1e-8)
