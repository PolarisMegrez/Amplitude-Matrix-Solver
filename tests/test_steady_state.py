"""Tests for steady-state solvers."""

import numpy as np
import pytest

from numerics.models.kerr_2mode import Kerr2Mode
from numerics.models.vdp_2mode import VdP2Mode
from numerics.models.kerr_3mode_hopf import Kerr3ModeHopf
from numerics.models.kerr_3pa import Kerr3PA
from numerics.solvers.steady_state import solve_steady_state
from numerics.core.liouvillian import liouvillian_residual


def test_kerr_2mode_identity_exact():
    """For symmetric parameters, R=I is an exact steady state of Kerr2Mode."""
    model = Kerr2Mode()
    params = {'s': 0.0, 'omega_A': 1.0, 'omega_B': 1.0,
              'kappa_A': 0.1, 'kappa_B': 0.1, 'g': 0.2}
    res = solve_steady_state(model, params, guess=np.eye(2), method='cholesky')
    assert res.success
    assert np.allclose(res.R, np.eye(2), atol=1e-8)
    L = model.liouvillian(res.R, params)
    assert liouvillian_residual(L) < 1e-8


def test_vdp_2mode_cholesky():
    """VdP2Mode has a stable physical steady state via Cholesky."""
    model = VdP2Mode()
    params = {
        'omega_a': 0.0, 'omega_b': 0.0,
        'gamma_a': 2.0, 'gamma_b': 0.5,
        'Gamma': 0.0001, 'g': 0.5, 'D': 1.0
    }
    from numerics.core.r_matrix import vector_to_R_matrix
    guess = vector_to_R_matrix(np.array([20000.0, 20000.0, 20000.0, -20000.0]))
    res = solve_steady_state(model, params, guess=guess, method='cholesky')
    assert res.success
    assert np.all(np.linalg.eigvalsh(res.R) > -1e-8)
    L = model.liouvillian(res.R, params)
    assert liouvillian_residual(L) < 1e-6


def test_kerr_3mode_hopf():
    """Kerr3ModeHopf (star) reaches a physical steady state."""
    model = Kerr3ModeHopf()
    params = {
        'omega_a': 0.0, 'omega_b': 1.0, 'omega_c': 1.05,
        'kappa_a': 0.05, 'kappa_b': 0.06, 'kappa_c': 0.056,
        'g_b': 0.2, 'g_c': 0.2, 'chi': 0.0001
    }
    res = solve_steady_state(model, params, guess=np.eye(3) * 1000.0,
                             method='cholesky')
    assert res.success
    assert np.all(np.linalg.eigvalsh(res.R) > -1e-8)


def test_kerr_3pa_nonzero_branch():
    """Kerr3PA above threshold finds the nonzero limit-cycle branch."""
    model = Kerr3PA()
    params = {'omega_0': 1.0, 'chi': 0.01, 'kappa_3': 0.001, 'mu': 0.5, 'g1': 0.0}
    res = solve_steady_state(model, params, guess=np.array([[100.0]]), method='root')
    assert res.success
    expected_R2 = model.deterministic_amplitude(params) ** 2
    assert np.isclose(res.R[0, 0].real, expected_R2, rtol=1e-4)
