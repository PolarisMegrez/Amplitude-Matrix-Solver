"""
2-mode Kerr nonlinear exceptional point model.

Hamiltonian (from 2-mode-Kerr/general.nb):
    H = [[s + omega_A - i kappa_A/2, g],
         [g, omega_B - i kappa_B/2]]

Diffusion:
    D = [[kappa_A, 0],
         [0, kappa_B]]

The control parameter s detunes cavity A and drives the EP.
"""

from __future__ import annotations

import sympy as sp
from numerics.models.base import Model
from numerics.core.backend import get_array_module
from numerics.core.jacobian import JacobianBuilder
from numerics.core.r_matrix import get_parameterized_R


class Kerr2Mode(Model):
    """2-mode Kerr model with tunable detuning s."""

    @property
    def dim(self) -> int:
        return 2

    def control_params(self) -> list[str]:
        return ["s", "g"]

    def H(self, R, params: dict):
        xp = get_array_module(R)
        s = params["s"]
        omega_A = params["omega_A"]
        omega_B = params["omega_B"]
        kappa_A = params["kappa_A"]
        kappa_B = params["kappa_B"]
        g = params["g"]
        R = xp.asarray(R)
        out_shape = R.shape[:-2]
        H = xp.zeros(out_shape + (2, 2), dtype=xp.complex128)
        H[..., 0, 0] = s + omega_A - 0.5j * kappa_A
        H[..., 0, 1] = g
        H[..., 1, 0] = g
        H[..., 1, 1] = omega_B - 0.5j * kappa_B
        return H

    def D(self, R, params: dict):
        xp = get_array_module(R)
        kappa_A = params["kappa_A"]
        kappa_B = params["kappa_B"]
        R = xp.asarray(R)
        out_shape = R.shape[:-2]
        D = xp.zeros(out_shape + (2, 2), dtype=xp.complex128)
        D[..., 0, 0] = kappa_A
        D[..., 1, 1] = kappa_B
        return D

    def build_jacobian_builder(self, params: dict | None = None, modules: str | None = None) -> JacobianBuilder:
        """Return a symbolic Jacobian builder for this model."""
        s, oA, oB, kA, kB, g = sp.symbols(
            "s omega_A omega_B kappa_A kappa_B g", real=True
        )
        R_sym, r_vec = get_parameterized_R(2)
        H_sym = sp.Matrix([
            [s + oA - sp.I * kA / 2, g],
            [g, oB - sp.I * kB / 2],
        ])
        D_sym = sp.Matrix([
            [kA, 0],
            [0, kB],
        ])
        return JacobianBuilder(H_sym, D_sym, R_sym, r_vec, [s, oA, oB, kA, kB, g], modules=modules)
