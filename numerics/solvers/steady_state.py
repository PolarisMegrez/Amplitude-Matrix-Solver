"""
Unified steady-state solvers for the nonlinear R-matrix equation.

Methods:
    - "root": solve the real-vector equation L(R)=0 with scipy.optimize.root.
    - "cholesky": parameterize R = L L^dagger and solve the residual equation,
      guaranteeing positive semidefiniteness.
    - "auto": try root first, fall back to cholesky on failure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from numerics.models.base import Model, SteadyStateResult

from numerics.core.r_matrix import R_matrix_to_vector, vector_to_R_matrix
from numerics.core.frequency import omega_frequency
from numerics.core.jacobian import numerical_jacobian
from numerics.utils.validation import convergence_message, is_positive_semidefinite
from numerics.solvers.backends import to_numpy


def _vec_to_L(vec: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct lower-triangular complex L from a real vector."""
    L = np.zeros((n, n), dtype=complex)
    idx = 0
    for i in range(n):
        L[i, i] = vec[idx]
        idx += 1
        for j in range(i):
            L[i, j] = vec[idx] + 1j * vec[idx + 1]
            idx += 2
    return L


def _L_to_vec(L: np.ndarray, n: int) -> np.ndarray:
    """Flatten lower-triangular L to a real vector."""
    vec = []
    for i in range(n):
        vec.append(float(np.real(L[i, i])))
        for j in range(i):
            vec.append(float(np.real(L[i, j])))
            vec.append(float(np.imag(L[i, j])))
    return np.array(vec)


def _cholesky_residual_func(model: "Model", params: dict):
    """Build residual F(l_vec) = L(R) in the Cholesky parameterization."""
    n = model.dim

    def residual(l_vec: np.ndarray) -> np.ndarray:
        L = _vec_to_L(l_vec, n)
        R = L @ L.conj().T
        H = model.H(R, params)
        D = model.D(R, params)
        # L(R) = -i H R + i R H^dagger + D
        eq = -1j * H @ R + 1j * R @ H.conj().T + D
        res = []
        for i in range(n):
            res.append(float(np.real(eq[i, i])))
            for j in range(i):
                res.append(float(np.real(eq[i, j])))
                res.append(float(np.imag(eq[i, j])))
        return np.array(res)

    return residual


def _root_residual_func(model: "Model", params: dict):
    """Build residual F(r_vec) = real-vector form of L(R)."""
    n = model.dim

    def residual(r_vec: np.ndarray) -> np.ndarray:
        R = vector_to_R_matrix(r_vec, n)
        H = model.H(R, params)
        D = model.D(R, params)
        eq = -1j * H @ R + 1j * R @ H.conj().T + D
        return R_matrix_to_vector(eq, n)

    return residual


def _make_root_jacobian(model: "Model", params: dict):
    """Return a Jacobian callable for the root method if available."""
    builder = model.build_jacobian_builder(params)
    if builder is None:
        return None
    n = model.dim

    def jac(r_vec: np.ndarray) -> np.ndarray:
        return builder(r_vec, params)

    return jac


def _solve_cholesky_once(
    model: "Model",
    params: dict,
    guess_R: np.ndarray,
    method: str,
    tol: float,
    **kwargs,
) -> "SteadyStateResult":
    # cholesky path does not reuse root residual/jacobian
    """Single Cholesky solve attempt."""
    from numerics.models.base import SteadyStateResult

    n = model.dim
    try:
        L_guess = np.linalg.cholesky(guess_R)
    except np.linalg.LinAlgError:
        L_guess = np.diag(np.sqrt(np.abs(np.diag(guess_R))))

    l_vec_guess = _L_to_vec(L_guess, n)
    residual = _cholesky_residual_func(model, params)

    sol = root(residual, l_vec_guess, method=method, tol=tol, **kwargs)
    L_final = _vec_to_L(sol.x, n)
    R_final = L_final @ L_final.conj().T
    max_res = float(np.max(np.abs(sol.fun))) if hasattr(sol, "fun") else np.inf
    success = bool(sol.success and max_res <= tol)

    H = model.H(R_final, params)
    omega = omega_frequency(H, R_final)

    return SteadyStateResult(
        R=R_final,
        params=params,
        success=success,
        residual=max_res,
        method=f"cholesky-{method}",
        message=convergence_message(sol.success, max_res, tol),
        omega=omega,
        J_eigvals=None,
        iterations=sol.nit if hasattr(sol, "nit") else None,
    )


def solve_cholesky_liouvillian(
    model: "Model",
    params: dict,
    guess_R: np.ndarray | None = None,
    tol: float = 1e-10,
    **kwargs,
) -> "SteadyStateResult":
    """
    Solve L(R)=0 using Cholesky parameterization R = L L^dagger.

    Tries the Levenberg-Marquardt method first; if it fails, falls back to
    the hybrid Powell method, matching the behavior of the original notebooks.

    Parameters
    ----------
    model : Model
        The model to solve.
    params : dict
        Model parameters.
    guess_R : np.ndarray, optional
        Initial guess for R. Defaults to identity.
    tol : float
        Convergence tolerance.

    Returns
    -------
    SteadyStateResult
    """
    if guess_R is None:
        guess_R = np.eye(model.dim, dtype=complex)

    res_lm = _solve_cholesky_once(model, params, guess_R, "lm", tol, **kwargs)
    if res_lm.success:
        return res_lm

    res_hybr = _solve_cholesky_once(model, params, guess_R, "hybr", tol, **kwargs)
    if res_hybr.success or res_hybr.residual < res_lm.residual:
        return res_hybr
    return res_lm


def solve_root_liouvillian(
    model: "Model",
    params: dict,
    guess_R: np.ndarray | None = None,
    root_method: str = "hybr",
    tol: float = 1e-10,
    use_jacobian: bool = True,
    residual_func: Callable | None = None,
    jac_func: Callable | None = None,
    **kwargs,
) -> "SteadyStateResult":
    """
    Solve L(R)=0 directly on the real-vector representation of R.

    Parameters
    ----------
    model : Model
        The model to solve.
    params : dict
        Model parameters.
    guess_R : np.ndarray, optional
        Initial guess for R. Defaults to identity.
    root_method : str
        Method passed to scipy.optimize.root.
    tol : float
        Convergence tolerance.
    use_jacobian : bool
        If True, use symbolic Jacobian when available.

    Returns
    -------
    SteadyStateResult
    """
    from numerics.models.base import SteadyStateResult

    n = model.dim
    if guess_R is None:
        guess_R = np.eye(n, dtype=complex)

    r_guess = R_matrix_to_vector(guess_R, n)
    residual = residual_func if residual_func is not None else _root_residual_func(model, params)
    if jac_func is not None:
        jac = jac_func if use_jacobian else None
    else:
        jac = _make_root_jacobian(model, params) if use_jacobian else None

    sol = root(residual, r_guess, jac=jac, method=root_method, tol=tol, **kwargs)
    R_final = vector_to_R_matrix(sol.x, n)
    max_res = float(np.max(np.abs(sol.fun))) if hasattr(sol, "fun") else np.inf
    success = bool(sol.success and max_res <= tol)

    H = model.H(R_final, params)
    omega = omega_frequency(H, R_final)

    # Jacobian eigenvalues
    if jac is not None:
        J = jac(sol.x)
    else:
        J = numerical_jacobian(residual, sol.x)
    J_eigvals = np.linalg.eigvals(J)

    return SteadyStateResult(
        R=R_final,
        params=params,
        success=success,
        residual=max_res,
        method=f"root-{root_method}",
        message=convergence_message(sol.success, max_res, tol),
        omega=omega,
        J_eigvals=J_eigvals,
        iterations=sol.nit if hasattr(sol, "nit") else None,
    )


def solve_steady_state(
    model: "Model",
    params: dict,
    guess: np.ndarray | None = None,
    method: str = "auto",
    tol: float = 1e-10,
    residual_func: Callable | None = None,
    jac_func: Callable | None = None,
    **kwargs,
) -> "SteadyStateResult":
    """
    Unified dispatcher for steady-state solvers.

    Parameters
    ----------
    model : Model
        Model instance.
    params : dict
        Model parameters.
    guess : np.ndarray, optional
        Initial guess for R (Hermitian matrix).
    method : {"auto", "root", "cholesky"}
        Solver strategy.
    tol : float
        Convergence tolerance.
    **kwargs
        Additional options passed to the underlying scipy.optimize.root call.

    Returns
    -------
    SteadyStateResult
    """
    if method == "root":
        return solve_root_liouvillian(
            model, params, guess, tol=tol,
            residual_func=residual_func, jac_func=jac_func, **kwargs
        )
    if method == "cholesky":
        return solve_cholesky_liouvillian(model, params, guess, tol=tol, **kwargs)
    if method == "auto":
        res_root = solve_root_liouvillian(
            model, params, guess, tol=tol,
            residual_func=residual_func, jac_func=jac_func, **kwargs
        )
        if res_root.success:
            return res_root
        res_chol = solve_cholesky_liouvillian(model, params, guess, tol=tol, **kwargs)
        if res_chol.success:
            return res_chol
        # Return the better of the two
        return res_root if res_root.residual <= res_chol.residual else res_chol

    raise ValueError(f"Unknown steady-state method: {method!r}")
