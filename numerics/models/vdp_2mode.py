"""
2-mode van der Pol oscillator in the R-matrix formalism.

Dynamics (from Tensor/2-mode-vdP.ipynb):
    -i H(R) = [[-i omega_a + gamma_a/2 + Gamma(1 - R_11), -i g],
               [-i g, -i omega_b - gamma_b/2]]

Therefore:
    H(R) = [[omega_a + i (gamma_a/2 + Gamma(1 - R_11)), g],
            [g, omega_b - i gamma_b/2]]

Diffusion:
    D(R) = [[D (gamma_a/2 + Gamma(2 R_11 - 1)), 0],
            [0, D gamma_b/2]]

Control parameters: omega_a, omega_b, gamma_a, gamma_b, Gamma, g, D.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from numerics.models.base import Model
from numerics.core.backend import get_array_module
from numerics.core.jacobian import JacobianBuilder
from numerics.core.r_matrix import get_parameterized_R


class VdP2Mode(Model):
    """2-mode van der Pol oscillator."""

    @property
    def dim(self) -> int:
        return 2

    def control_params(self) -> list[str]:
        return ["omega_a", "omega_b", "gamma_a", "gamma_b", "Gamma", "g", "D"]

    def H(self, R, params: dict):
        xp = get_array_module(R)
        R = xp.asarray(R)
        omega_a = params["omega_a"]
        omega_b = params["omega_b"]
        gamma_a = params["gamma_a"]
        gamma_b = params["gamma_b"]
        Gamma = params["Gamma"]
        g = params["g"]
        r11 = xp.real(R[..., 0, 0])

        # Broadcast scalar parameters with the batch shape of R
        out_shape = r11.shape
        H = xp.zeros(out_shape + (2, 2), dtype=xp.complex128)
        H[..., 0, 0] = omega_a + 1j * (0.5 * gamma_a + Gamma * (1.0 - r11))
        H[..., 0, 1] = g
        H[..., 1, 0] = g
        H[..., 1, 1] = omega_b - 0.5j * gamma_b
        return H

    def D(self, R, params: dict):
        xp = get_array_module(R)
        R = xp.asarray(R)
        gamma_a = params["gamma_a"]
        gamma_b = params["gamma_b"]
        Gamma = params["Gamma"]
        D = params["D"]
        r11 = xp.real(R[..., 0, 0])

        out_shape = r11.shape
        Dmat = xp.zeros(out_shape + (2, 2), dtype=xp.complex128)
        Dmat[..., 0, 0] = D * (0.5 * gamma_a + Gamma * (2.0 * r11 - 1.0))
        Dmat[..., 1, 1] = D * 0.5 * gamma_b
        return Dmat

    def build_jacobian_builder(self, params: dict | None = None, modules: str | None = None) -> JacobianBuilder:
        """Return a symbolic Jacobian builder for this model."""
        oa, ob, ga, gb, Gam, g, D = sp.symbols(
            "omega_a omega_b gamma_a gamma_b Gamma g D", real=True
        )
        R_sym, r_vec = get_parameterized_R(2)
        r11_sym = R_sym[0, 0]
        H_sym = sp.Matrix([
            [oa + sp.I * (ga / 2 + Gam * (1 - r11_sym)), g],
            [g, ob - sp.I * gb / 2],
        ])
        D_sym = sp.Matrix([
            [D * (ga / 2 + Gam * (2 * r11_sym - 1)), 0],
            [0, D * gb / 2],
        ])
        return JacobianBuilder(
            H_sym, D_sym, R_sym, r_vec, [oa, ob, ga, gb, Gam, g, D], modules=modules
        )
