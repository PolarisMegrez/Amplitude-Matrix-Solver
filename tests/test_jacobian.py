"""Tests for symbolic and numerical Jacobian construction."""

import numpy as np

from numerics.models.kerr_2mode import Kerr2Mode
from numerics.models.vdp_2mode import VdP2Mode
from numerics.core.jacobian import numerical_jacobian
from numerics.core.r_matrix import R_matrix_to_vector


def _residual_func(model, params):
    n = model.dim
    def f(r_vec):
        from numerics.core.r_matrix import vector_to_R_matrix
        from numerics.core.liouvillian import liouvillian
        R = vector_to_R_matrix(r_vec, n)
        L = liouvillian(model.H(R, params), R, model.D(R, params))
        return R_matrix_to_vector(L, n)
    return f


def test_jacobian_symbolic_vs_numerical_kerr():
    """Symbolic Jacobian matches central-difference Jacobian for Kerr2Mode."""
    model = Kerr2Mode()
    params = {'s': 0.1, 'omega_A': 1.0, 'omega_B': 1.0,
              'kappa_A': 0.1, 'kappa_B': 0.1, 'g': 0.2}
    builder = model.build_jacobian_builder(params)
    rng = np.random.default_rng(1)
    r_vec = rng.random(4) + 0.1

    J_sym = builder(r_vec, params)
    f = _residual_func(model, params)
    J_num = numerical_jacobian(f, r_vec, eps=1e-6)

    assert np.allclose(J_sym, J_num, atol=1e-5)


def test_jacobian_symbolic_vs_numerical_vdp():
    """Symbolic Jacobian matches central-difference Jacobian for VdP2Mode."""
    model = VdP2Mode()
    params = {
        'omega_a': 0.0, 'omega_b': 0.0,
        'gamma_a': 2.0, 'gamma_b': 0.5,
        'Gamma': 0.0001, 'g': 0.5, 'D': 1.0
    }
    builder = model.build_jacobian_builder(params)
    rng = np.random.default_rng(2)
    r_vec = rng.random(4) * 100.0

    J_sym = builder(r_vec, params)
    f = _residual_func(model, params)
    J_num = numerical_jacobian(f, r_vec, eps=1e-5)

    assert np.allclose(J_sym, J_num, atol=1e-4)
