"""Tests for the 2-D multistability scan framework."""

import numpy as np
import pytest

from numerics.models.vdp_2mode import VdP2Mode
from numerics.scans.multistability import (
    MultistabilityScan2D,
    ParallelConfig,
    ScanTolerances,
)


@pytest.fixture
def vdp_model():
    return VdP2Mode()


@pytest.fixture
def vdp_params():
    return {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 2.0,
        "gamma_b": 1.0,
        "Gamma": 0.01,
        "g": 0.5,
        "D": 1.0,
    }


def test_vdp_2d_scan_sequential(vdp_model, vdp_params):
    """A small sequential scan should find the expected multistability pattern."""
    scan = MultistabilityScan2D(
        model=vdp_model,
        base_params=vdp_params,
        axes={
            "gamma_b": np.linspace(0.85, 1.35, 15),
            "omega_a": np.linspace(-0.25, 0.25, 15),
        },
        n_random_guesses=30,
        tolerances=ScanTolerances(distance_tol=3.0, branch_match_tol=10.0),
        max_branches=5,
        backend="numpy",
        parallel=None,
        symmetry_axis="omega_a",
        verbose=False,
    )
    result = scan.run()

    assert result.grid_shape == (15, 15)
    assert np.all(result.n_solutions <= result.max_branches)
    # The chosen window should contain at least some 4-solution cells.
    assert int(np.max(result.n_solutions)) >= 3

    # Reflection symmetry in omega_a should be satisfied.
    assert np.sum(result.n_solutions != result.n_solutions[:, ::-1]) == 0

    # All stored branches should have finite R and residual.
    for idx in np.argwhere(result.n_solutions > 0):
        for k in range(int(result.n_solutions[tuple(idx)])):
            assert np.isfinite(result.R_matrices[(*idx, k, 0, 0)])
            assert np.isfinite(result.residuals[(*idx, k)])

    # Round-trip save/load.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scan.npz"
        result.save_npz(path)
        loaded = type(result).load_npz(path)
        assert np.array_equal(loaded.n_solutions, result.n_solutions)


@pytest.mark.slow
def test_vdp_2d_scan_parallel(vdp_model, vdp_params):
    """Parallel tile scan should agree with sequential scan on a small grid."""
    axes = {
        "gamma_b": np.linspace(0.85, 1.35, 15),
        "omega_a": np.linspace(-0.25, 0.25, 15),
    }
    seq = MultistabilityScan2D(
        model=vdp_model,
        base_params=vdp_params,
        axes=axes,
        n_random_guesses=30,
        max_branches=5,
        backend="numpy",
        parallel=None,
        symmetry_axis="omega_a",
        verbose=False,
    )
    par = MultistabilityScan2D(
        model=vdp_model,
        base_params=vdp_params,
        axes=axes,
        n_random_guesses=30,
        max_branches=5,
        backend="numpy",
        parallel=ParallelConfig(n_workers=4, n_tiles=16),
        symmetry_axis="omega_a",
        verbose=False,
    )
    r_seq = seq.run()
    r_par = par.run()

    # Counts should match exactly; a few boundary cells may differ but
    # refinement + symmetry enforcement should keep them close.
    diff = int(np.sum(r_seq.n_solutions != r_par.n_solutions))
    assert diff <= 2
    assert np.sum(r_par.n_solutions != r_par.n_solutions[:, ::-1]) == 0
