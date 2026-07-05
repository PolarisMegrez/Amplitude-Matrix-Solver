"""
Kerr + three-photon absorption (3PA) single-mode Hopf model.

Deterministic drift (from supplementary material):
    dot(alpha) = (mu - i omega_0) alpha - 2 i chi |alpha|^2 alpha
                 - (3/2) kappa_3 |alpha|^4 alpha + xi(t)

In the R-matrix formalism we take R = |alpha|^2 (1 x 1 real) and write the
drift coefficient as -i H(R) R + i R H^dagger(R), which gives:

    H(R) = omega_0 + 2 chi R + i ( (3/2) kappa_3 R^2 - mu )

The diffusion matrix is a scalar noise strength:
    D(R) = [[g1]]

Parameters: omega_0, chi, kappa_3, mu, g1.  Often mu = beta * epsilon.
"""

from __future__ import annotations

import sympy as sp
from numerics.models.base import Model
from numerics.core.backend import get_array_module
from numerics.core.jacobian import JacobianBuilder


class Kerr3PA(Model):
    """Single-mode Kerr-3PA limit-cycle model."""

    @property
    def dim(self) -> int:
        return 1

    def control_params(self) -> list[str]:
        return ["omega_0", "chi", "kappa_3", "mu", "g1"]

    def H(self, R, params: dict):
        xp = get_array_module(R)
        R = xp.asarray(R)
        omega_0 = params["omega_0"]
        chi = params["chi"]
        kappa_3 = params["kappa_3"]
        mu = params["mu"]
        r = xp.real(R[..., 0, 0])
        return (omega_0 + 2.0 * chi * r + 1j * (1.5 * kappa_3 * r**2 - mu))[..., None, None]

    def D(self, R, params: dict):
        xp = get_array_module(R)
        g1 = params["g1"]
        R = xp.asarray(R)
        out_shape = R.shape[:-2]
        D = xp.zeros(out_shape + (1, 1), dtype=xp.complex128)
        D[..., 0, 0] = g1
        return D

    def build_jacobian_builder(self, params: dict | None = None, modules: str | None = None) -> JacobianBuilder:
        """Return a symbolic Jacobian builder for this scalar model."""
        o0, chi, k3, mu = sp.symbols("omega_0 chi kappa_3 mu", real=True)
        r = sp.Symbol("R_11", real=True)
        R_sym = sp.Matrix([[r]])
        r_vec = sp.Matrix([r])
        H_sym = sp.Matrix([
            [o0 + 2 * chi * r + sp.I * (sp.Rational(3, 2) * k3 * r**2 - mu)]
        ])
        D_sym = sp.Matrix([[0]])  # g1 does not enter the Jacobian
        return JacobianBuilder(H_sym, D_sym, R_sym, r_vec, [o0, chi, k3, mu], modules=modules)

    def deterministic_amplitude(self, params: dict) -> float:
        """
        Analytic limit-cycle amplitude R0 = |alpha| for the deterministic model.

        Returns
        -------
        R0 : float
            Steady-state amplitude (0 if mu <= 0).
        """
        mu = params["mu"]
        kappa_3 = params["kappa_3"]
        if mu <= 0 or kappa_3 <= 0:
            return 0.0
        return (2.0 * mu / (3.0 * kappa_3)) ** 0.25

    def deterministic_frequency(self, params: dict) -> float:
        """Analytic oscillation frequency omega_osc = omega_0 + 2 chi R0^2."""
        R0 = self.deterministic_amplitude(params)
        return params["omega_0"] + 2.0 * params["chi"] * R0**2
