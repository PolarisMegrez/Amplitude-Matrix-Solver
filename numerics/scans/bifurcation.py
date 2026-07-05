"""
Bifurcation-point detection from parameter-scan data.

A bifurcation is signaled by a Jacobian eigenvalue crossing the imaginary
axis, i.e. max Re(lambda) -> 0. This module locates the control-parameter
value where the critical eigenvalue is closest to zero.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numerics.models.base import Model, ScanResult


class BifurcationLocator:
    """Locate bifurcation points and estimate critical exponents."""

    def __init__(self, scan: "ScanResult"):
        """
        Parameters
        ----------
        scan : ScanResult
            Completed parameter scan.
        """
        self.scan = scan

    def critical_indices(self) -> np.ndarray:
        """
        Return indices of scan points sorted by the smallest |Re(lambda)|.

        Only valid points are considered.
        """
        valid = np.asarray(self.scan.valid_mask)
        if not np.any(valid):
            return np.array([], dtype=int)

        eigvals = np.asarray(self.scan.J_eigvals)
        max_re = np.array([
            np.max(np.real(eigvals[i])) if valid[i] else np.inf
            for i in range(len(valid))
        ])
        return np.argsort(np.abs(max_re))

    def locate_zero_crossing(self) -> dict | None:
        """
        Find a sign change of max Re(lambda) and refine with Brent's method.

        Returns
        -------
        dict or None
            Dictionary with 'left', 'right', 'critical_value', 'critical_eigval'
            if a bracket is found; otherwise None.
        """
        valid = np.asarray(self.scan.valid_mask)
        if not np.any(valid):
            return None

        values = np.asarray(self.scan.control_values)
        eigvals = np.asarray(self.scan.J_eigvals)
        max_re = np.array([
            np.max(np.real(eigvals[i])) if valid[i] else np.nan
            for i in range(len(valid))
        ])

        # Find sign changes
        for i in range(len(values) - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            if np.sign(max_re[i]) != np.sign(max_re[i + 1]) and max_re[i] != 0:
                return {
                    "left": values[i],
                    "right": values[i + 1],
                    "left_eigval": max_re[i],
                    "right_eigval": max_re[i + 1],
                }
        return None

    def refine_bifurcation(
        self,
        model: "Model",
        base_params: dict,
        bracket: tuple[float, float],
        tol: float = 1e-12,
        maxiter: int = 100,
    ) -> dict | None:
        """
        Refine a bifurcation point by minimizing |max Re(lambda)| over a
        control-parameter bracket.

        Parameters
        ----------
        model : Model
            Model instance.
        base_params : dict
            Base parameters; control_param is varied inside the bracket.
        bracket : tuple[float, float]
            (left, right) control-parameter bracket.
        tol : float
            Brent tolerance.
        maxiter : int
            Maximum iterations.

        Returns
        -------
        dict or None
            Refined bifurcation information.
        """
        control_param = self.scan.control_param

        def max_real_eigval(s: float) -> float:
            params = base_params.copy()
            params[control_param] = s
            res = solve_steady_state(
                model, params, method="cholesky", tol=tol
            )
            if not res.success or res.J_eigvals is None:
                return np.nan
            return float(np.max(np.real(res.J_eigvals)))

        try:
            from numerics.solvers.steady_state import solve_steady_state
            critical_value = brentq(
                max_real_eigval,
                bracket[0],
                bracket[1],
                xtol=tol,
                maxiter=maxiter,
            )
            params = base_params.copy()
            params[control_param] = critical_value
            res = solve_steady_state(model, params, method="cholesky", tol=tol)
            return {
                "control_param": control_param,
                "critical_value": critical_value,
                "R0": res.R,
                "omega": res.omega,
                "critical_eigvals": res.J_eigvals,
            }
        except ValueError:
            return None
