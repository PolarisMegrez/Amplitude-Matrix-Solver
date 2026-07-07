"""Tests for the GPU ring-batch scan path."""

from __future__ import annotations

import numpy as np
import pytest

from numerics.models.vdp_2mode import VdP2Mode
from numerics.scans.multistability import (
    MultistabilityScan2D,
    ScanTolerances,
)


def test_gpu_ring_matches_cpu():
    """A small GPU ring scan should agree with the CPU sequential scan."""
    cupy = pytest.importorskip("cupy")

    model = VdP2Mode()
    params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 2.0,
        "gamma_b": 1.0,
        "Gamma": 0.01,
        "g": 0.5,
        "D": 1.0,
    }
    axes = {
        "gamma_b": np.linspace(0.85, 1.35, 15),
        "omega_a": np.linspace(-0.25, 0.25, 15),
    }

    common = {
        "model": model,
        "base_params": params,
        "axes": axes,
        "n_random_guesses": 30,
        "tolerances": ScanTolerances(distance_tol=3.0, branch_match_tol=10.0),
        "max_branches": 5,
        "parallel": None,
        "symmetry_axis": "omega_a",
        "verbose": False,
    }

    cpu = MultistabilityScan2D(backend="numpy", gpu_ring_batch=False, **common)
    gpu = MultistabilityScan2D(backend="cupy", gpu_ring_batch=True, **common)

    r_cpu = cpu.run()
    r_gpu = gpu.run()

    assert r_cpu.grid_shape == r_gpu.grid_shape
    assert np.sum(r_cpu.n_solutions != r_gpu.n_solutions) <= 2
    # Reflection symmetry must still hold.
    assert np.sum(r_gpu.n_solutions != r_gpu.n_solutions[:, ::-1]) == 0
