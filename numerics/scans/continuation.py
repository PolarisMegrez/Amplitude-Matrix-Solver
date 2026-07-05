"""
One-dimensional parameter scan with continuation guessing.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numerics.models.base import Model, ScanResult, SteadyStateResult

from numerics.models.base import ScanResult
from numerics.solvers.steady_state import solve_steady_state
from numerics.core.frequency import omega_frequency


class ParameterScan:
    """
    Scan a control parameter and track steady states, frequencies, and
    Jacobian eigenvalues.
    """

    def __init__(
        self,
        model: "Model",
        base_params: dict,
        control_param: str,
        control_values: np.ndarray,
        solver_method: str = "auto",
        fallback_guess: np.ndarray | None = None,
        tol: float = 1e-10,
        **solver_kwargs,
    ):
        """
        Parameters
        ----------
        model : Model
            Model to scan.
        base_params : dict
            Base parameter dictionary; control_param will be overwritten.
        control_param : str
            Parameter key to scan.
        control_values : np.ndarray
            Array of control parameter values.
        solver_method : str
            Method passed to solve_steady_state.
        fallback_guess : np.ndarray, optional
            Guess to use when continuation fails.
        tol : float
            Solver tolerance.
        """
        self.model = model
        self.base_params = base_params.copy()
        self.control_param = control_param
        self.control_values = np.asarray(control_values)
        self.solver_method = solver_method
        self.fallback_guess = fallback_guess
        self.tol = tol
        self.solver_kwargs = solver_kwargs

    def run(self, initial_guess: np.ndarray | None = None) -> ScanResult:
        """
        Run the parameter scan.

        Parameters
        ----------
        initial_guess : np.ndarray, optional
            Initial guess for the first control value.

        Returns
        -------
        ScanResult
        """
        result = ScanResult(
            control_param=self.control_param,
            control_values=self.control_values,
        )

        guess = initial_guess
        if guess is None:
            guess = np.eye(self.model.dim, dtype=complex)

        for value in self.control_values:
            params = self.base_params.copy()
            params[self.control_param] = value

            res = solve_steady_state(
                self.model,
                params,
                guess=guess,
                method=self.solver_method,
                tol=self.tol,
                **self.solver_kwargs,
            )

            # Fallback if continuation failed
            if not res.success and self.fallback_guess is not None:
                res = solve_steady_state(
                    self.model,
                    params,
                    guess=self.fallback_guess,
                    method=self.solver_method,
                    tol=self.tol,
                    **self.solver_kwargs,
                )

            result.steady_states.append(res.R)
            result.residuals.append(res.residual)
            result.valid_mask.append(res.success)
            result.omegas.append(res.omega if res.omega is not None else np.nan)

            # Compute Jacobian eigenvalues if not already present
            if res.J_eigvals is not None:
                result.J_eigvals.append(res.J_eigvals)
            else:
                try:
                    from numerics.core.jacobian import JacobianBuilder
                    builder = self.model.build_jacobian_builder(params)
                    if builder is not None:
                        from numerics.core.r_matrix import R_matrix_to_vector
                        r_vec = R_matrix_to_vector(res.R)
                        J = builder(r_vec, params)
                        result.J_eigvals.append(np.linalg.eigvals(J))
                    else:
                        result.J_eigvals.append(np.full(self.model.dim**2, np.nan))
                except Exception:
                    result.J_eigvals.append(np.full(self.model.dim**2, np.nan))

            # Update continuation guess
            if res.success:
                guess = res.R
            # If failed, keep previous guess for next step

        return result
