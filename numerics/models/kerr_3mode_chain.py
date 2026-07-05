"""
3-mode Kerr model with chain coupling (from Tensor/3-mode-Kerr.ipynb).

Hamiltonian:
    H(R) = [[omega_a - i kappa_a + 2 chi R_11, g_ab, 0],
            [g_ab, omega_b + i kappa_b, g_ac],
            [0, g_ac, omega_c - i kappa_c]]

Diffusion:
    D = diag([kappa_a, kappa_b, kappa_c])

Control parameters: omega_a, omega_b, omega_c, kappa_a, kappa_b, kappa_c,
g_ab, g_ac, chi.
"""

from __future__ import annotations

import sympy as sp
from numerics.models.base import Model
from numerics.core.backend import get_array_module
from numerics.core.jacobian import JacobianBuilder
from numerics.core.r_matrix import get_parameterized_R


class Kerr3ModeChain(Model):
    """3-mode Kerr model with chain coupling (a-b-c)."""

    @property
    def dim(self) -> int:
        return 3

    def control_params(self) -> list[str]:
        return [
            "omega_a", "omega_b", "omega_c",
            "kappa_a", "kappa_b", "kappa_c",
            "g_ab", "g_ac", "chi",
        ]

    def H(self, R, params: dict):
        xp = get_array_module(R)
        R = xp.asarray(R)
        oa = params["omega_a"]
        ob = params["omega_b"]
        oc = params["omega_c"]
        ka = params["kappa_a"]
        kb = params["kappa_b"]
        kc = params["kappa_c"]
        gab = params["g_ab"]
        gac = params["g_ac"]
        chi = params["chi"]
        r11 = xp.real(R[..., 0, 0])

        out_shape = r11.shape
        H = xp.zeros(out_shape + (3, 3), dtype=xp.complex128)
        H[..., 0, 0] = oa - 1j * ka + 2.0 * chi * r11
        H[..., 0, 1] = gab
        H[..., 1, 0] = gab
        H[..., 1, 1] = ob + 1j * kb
        H[..., 1, 2] = gac
        H[..., 2, 1] = gac
        H[..., 2, 2] = oc - 1j * kc
        return H

    def D(self, R, params: dict):
        xp = get_array_module(R)
        R = xp.asarray(R)
        ka = params["kappa_a"]
        kb = params["kappa_b"]
        kc = params["kappa_c"]
        out_shape = R.shape[:-2]
        D = xp.zeros(out_shape + (3, 3), dtype=xp.complex128)
        D[..., 0, 0] = ka
        D[..., 1, 1] = kb
        D[..., 2, 2] = kc
        return D

    def build_jacobian_builder(self, params: dict | None = None, modules: str | None = None) -> JacobianBuilder:
        """Return a symbolic Jacobian builder for this model."""
        oa, ob, oc = sp.symbols("omega_a omega_b omega_c", real=True)
        ka, kb, kc = sp.symbols("kappa_a kappa_b kappa_c", real=True)
        gab, gac, chi = sp.symbols("g_ab g_ac chi", real=True)
        R_sym, r_vec = get_parameterized_R(3)
        r11_sym = R_sym[0, 0]
        H_sym = sp.Matrix([
            [oa - sp.I * ka + 2 * chi * r11_sym, gab, 0],
            [gab, ob + sp.I * kb, gac],
            [0, gac, oc - sp.I * kc],
        ])
        D_sym = sp.Matrix([
            [ka, 0, 0],
            [0, kb, 0],
            [0, 0, kc],
        ])
        return JacobianBuilder(
            H_sym, D_sym, R_sym, r_vec,
            [oa, ob, oc, ka, kb, kc, gab, gac, chi], modules=modules
        )
