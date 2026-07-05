"""
3-mode Kerr/Hopf model (star coupling, from 3-mode-Hopf/3-mode.nb).

Hamiltonian:
    H(R) = [[omega_a - i kappa_a + 2 chi R_11, g_b, g_c],
            [g_b, omega_b - i kappa_b, 0],
            [g_c, 0, omega_c - i kappa_c]]

Diffusion:
    D = diag([kappa_a, kappa_b, kappa_c])

Control parameters: omega_a, omega_b, omega_c, kappa_a, kappa_b, kappa_c,
g_b, g_c, chi.
"""

from __future__ import annotations

import sympy as sp
from numerics.models.base import Model
from numerics.core.backend import get_array_module
from numerics.core.jacobian import JacobianBuilder
from numerics.core.r_matrix import get_parameterized_R


class Kerr3ModeHopf(Model):
    """3-mode Kerr model with star coupling (a-b and a-c)."""

    @property
    def dim(self) -> int:
        return 3

    def control_params(self) -> list[str]:
        return [
            "omega_a", "omega_b", "omega_c",
            "kappa_a", "kappa_b", "kappa_c",
            "g_b", "g_c", "chi",
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
        gb = params["g_b"]
        gc = params["g_c"]
        chi = params["chi"]
        r11 = xp.real(R[..., 0, 0])

        out_shape = r11.shape
        H = xp.zeros(out_shape + (3, 3), dtype=xp.complex128)
        H[..., 0, 0] = oa - 1j * ka + 2.0 * chi * r11
        H[..., 0, 1] = gb
        H[..., 0, 2] = gc
        H[..., 1, 0] = gb
        H[..., 1, 1] = ob - 1j * kb
        H[..., 2, 0] = gc
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
        gb, gc, chi = sp.symbols("g_b g_c chi", real=True)
        R_sym, r_vec = get_parameterized_R(3)
        r11_sym = R_sym[0, 0]
        H_sym = sp.Matrix([
            [oa - sp.I * ka + 2 * chi * r11_sym, gb, gc],
            [gb, ob - sp.I * kb, 0],
            [gc, 0, oc - sp.I * kc],
        ])
        D_sym = sp.Matrix([
            [ka, 0, 0],
            [0, kb, 0],
            [0, 0, kc],
        ])
        return JacobianBuilder(
            H_sym, D_sym, R_sym, r_vec,
            [oa, ob, oc, ka, kb, kc, gb, gc, chi], modules=modules
        )
