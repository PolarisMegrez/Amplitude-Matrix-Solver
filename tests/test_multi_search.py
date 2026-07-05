"""Tests for multi-solution steady-state search."""

import numpy as np

from numerics.models.kerr_3pa import Kerr3PA
from numerics.solvers.multi_search import find_steady_states


def test_kerr_3pa_branches():
    """Kerr3PA above threshold has both R=0 and R>0 steady states."""
    model = Kerr3PA()
    params = {"omega_0": 1.0, "chi": 0.01, "kappa_3": 0.001, "mu": 0.5, "g1": 0.0}
    results = find_steady_states(
        model, params, n_samples=10, scale=50.0, solver_method="root"
    )
    assert len(results) >= 2
    rs = sorted([r.R[0, 0].real for r in results])
    assert np.isclose(rs[0], 0.0, atol=1e-6)
    expected = model.deterministic_amplitude(params) ** 2
    assert np.isclose(rs[-1], expected, rtol=1e-4)
