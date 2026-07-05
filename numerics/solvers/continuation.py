"""
Pseudo-arclength continuation for R-matrix steady-state branches.

Traces solution curves of :math:`F(x, \\lambda) = 0` where :math:`x` is the
real-vector representation of :math:`R` and :math:`\\lambda` is a scalar
control parameter (e.g. ``omega_a`` or ``gamma_b``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from numerics.core.frequency import omega_frequency
from numerics.core.jacobian import JacobianBuilder, numerical_jacobian
from numerics.core.liouvillian import liouvillian_vector
from numerics.core.r_matrix import R_matrix_to_vector, vector_to_R_matrix


@dataclass
class ContinuationOptions:
    """Tuning parameters for pseudo-arclength continuation."""

    ds: float = 0.002
    ds_min: float = 1e-7
    ds_max: float = 0.02
    max_steps: int = 2000
    tol: float = 1e-10
    max_corrector_iter: int = 16
    corrector_damping: float = 1.0
    adapt_growth: float = 1.1
    adapt_shrink: float = 0.5
    param_eps: float = 1e-6


@dataclass
class ContinuationResult:
    """Container for a traced branch."""

    param_name: str
    lambdas: np.ndarray = field(default_factory=lambda: np.array([]))
    states: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    omegas: np.ndarray = field(default_factory=lambda: np.array([]))
    J_eigvals: list[np.ndarray] = field(default_factory=list)
    arclengths: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        return len(self.lambdas)

    def to_arrays(self) -> dict:
        """Return a dictionary of NumPy arrays for plotting."""
        return {
            "param_name": self.param_name,
            "lambdas": self.lambdas,
            "states": self.states,
            "residuals": self.residuals,
            "omegas": self.omegas,
            "arclengths": self.arclengths,
            "converged": self.converged,
            "J_eigvals": np.array(self.J_eigvals) if self.J_eigvals else np.array([]),
        }


def _params_with(model, params: dict, param_name: str, lam: float) -> dict:
    """Return a parameter dictionary with ``param_name`` set to ``lam``."""
    new_params = params.copy()
    new_params[param_name] = float(lam)
    return new_params


def _residual(model, params: dict, x: np.ndarray, param_name: str, lam: float) -> np.ndarray:
    """Evaluate F(x, lambda)."""
    p = _params_with(model, params, param_name, lam)
    return liouvillian_vector(model, x, p)


def _jacobian_x(
    model,
    params: dict,
    x: np.ndarray,
    param_name: str,
    lam: float,
    jac_builder: JacobianBuilder | None,
) -> np.ndarray:
    """Evaluate the Jacobian of F with respect to x."""
    p = _params_with(model, params, param_name, lam)
    if jac_builder is not None:
        return jac_builder(x, p)
    return numerical_jacobian(lambda v: liouvillian_vector(model, v, p), x)


def _param_derivative(
    model,
    params: dict,
    x: np.ndarray,
    param_name: str,
    lam: float,
    eps: float,
) -> np.ndarray:
    """Evaluate dF/dlambda by central finite differences."""
    Fp = _residual(model, params, x, param_name, lam + eps)
    Fm = _residual(model, params, x, param_name, lam - eps)
    return (Fp - Fm) / (2.0 * eps)


def _nullspace_vector(A: np.ndarray, preference: np.ndarray | None = None) -> np.ndarray:
    """
    Return a unit vector in the nullspace of A with sign chosen to align with
    ``preference`` when provided.
    """
    U, s, Vh = np.linalg.svd(A)
    tangent = Vh[-1, :].conj()
    tangent = tangent / (np.linalg.norm(tangent) + 1e-30)
    if preference is not None and np.dot(tangent, preference) < 0.0:
        tangent = -tangent
    return tangent


def _corrector_step(
    z: np.ndarray,
    z_pred: np.ndarray,
    tangent: np.ndarray,
    model,
    params: dict,
    param_name: str,
    jac_builder: JacobianBuilder | None,
    opts: ContinuationOptions,
) -> tuple[np.ndarray, bool]:
    """
    Apply one Newton corrector iteration for the augmented system.

    Returns the updated state and a flag indicating convergence on the
    original residual.
    """
    m = len(z) - 1
    x = z[:m]
    lam = float(z[-1])

    F = _residual(model, params, x, param_name, lam)
    Jx = _jacobian_x(model, params, x, param_name, lam, jac_builder)
    Flam = _param_derivative(model, params, x, param_name, lam, opts.param_eps)

    N = float(np.dot(tangent, z - z_pred))

    # Augmented Jacobian
    aug = np.zeros((m + 1, m + 1), dtype=float)
    aug[:m, :m] = Jx
    aug[:m, m] = Flam
    aug[m, :] = tangent

    rhs = np.zeros(m + 1, dtype=float)
    rhs[:m] = -F
    rhs[m] = -N

    try:
        dz = np.linalg.solve(aug, rhs)
    except np.linalg.LinAlgError:
        dz, *_ = np.linalg.lstsq(aug, rhs, rcond=None)

    z_new = z + opts.corrector_damping * dz
    F_new = _residual(model, params, z_new[:m], param_name, float(z_new[-1]))
    converged = float(np.linalg.norm(F_new)) <= opts.tol
    return z_new, converged


def _corrector(
    z_pred: np.ndarray,
    tangent: np.ndarray,
    model,
    params: dict,
    param_name: str,
    jac_builder: JacobianBuilder | None,
    opts: ContinuationOptions,
) -> tuple[np.ndarray, bool, int]:
    """
    Run the corrector loop.

    Returns the converged state, a success flag, and the number of iterations.
    """
    z = z_pred.copy()
    for it in range(opts.max_corrector_iter):
        z, converged = _corrector_step(
            z, z_pred, tangent, model, params, param_name, jac_builder, opts
        )
        if converged:
            return z, True, it + 1
    # Final check
    F = _residual(model, params, z[:-1], param_name, float(z[-1]))
    return z, float(np.linalg.norm(F)) <= opts.tol, opts.max_corrector_iter


def _tangent_from_state(
    model,
    params: dict,
    z: np.ndarray,
    param_name: str,
    jac_builder: JacobianBuilder | None,
    preference: np.ndarray | None,
    opts: ContinuationOptions,
) -> np.ndarray:
    """Compute a normalized tangent at z."""
    m = len(z) - 1
    x = z[:m]
    lam = float(z[-1])
    Jx = _jacobian_x(model, params, x, param_name, lam, jac_builder)
    Flam = _param_derivative(model, params, x, param_name, lam, opts.param_eps)
    A = np.hstack([Jx, Flam[:, None]])  # (m, m+1)
    return _nullspace_vector(A, preference=preference)


def _trace_one_direction(
    model,
    params: dict,
    param_name: str,
    x0: np.ndarray,
    lam0: float,
    direction: float,
    p_low: float,
    p_high: float,
    jac_builder: JacobianBuilder | None,
    opts: ContinuationOptions,
) -> ContinuationResult:
    """Trace the branch in one direction (direction = +1 or -1)."""
    m = len(x0)
    z0 = np.zeros(m + 1, dtype=float)
    z0[:m] = x0
    z0[-1] = lam0

    # Initial tangent oriented so that the parameter component has the sign of
    # ``direction``.  Subsequent tangents are oriented to preserve arclength
    # direction, which lets the corrector step through folds.
    tangent = _tangent_from_state(
        model, params, z0, param_name, jac_builder,
        preference=np.concatenate([np.zeros(m), [direction]]),
        opts=opts,
    )

    lambdas: list[float] = []
    states: list[np.ndarray] = []
    residuals: list[float] = []
    arclengths: list[float] = []
    converged_flags: list[bool] = []
    omegas: list[float] = []
    J_eigvals: list[np.ndarray] = []

    def record(z: np.ndarray, s: float, ok: bool):
        x = z[:m]
        lam = float(z[-1])
        p_step = _params_with(model, params, param_name, lam)
        R = vector_to_R_matrix(x, model.dim)
        try:
            H = model.H(R, p_step)
            omegas.append(float(omega_frequency(H, R)))
        except Exception:
            omegas.append(np.nan)
        try:
            J = jac_builder(x, p_step) if jac_builder is not None else None
            if J is None:
                J = numerical_jacobian(lambda v: liouvillian_vector(model, v, p_step), x)
            J_eigvals.append(np.linalg.eigvals(J))
        except Exception:
            J_eigvals.append(np.zeros(model.dim * model.dim, dtype=complex))

        lambdas.append(lam)
        states.append(x.copy())
        residuals.append(float(np.linalg.norm(_residual(model, params, x, param_name, lam))))
        arclengths.append(s)
        converged_flags.append(ok)

    record(z0, 0.0, True)

    ds = opts.ds
    s = 0.0
    z = z0.copy()

    for _ in range(opts.max_steps):
        if ds < opts.ds_min:
            break

        z_pred = z + ds * tangent
        lam_pred = float(z_pred[-1])
        if (direction > 0 and lam_pred > p_high) or (direction < 0 and lam_pred < p_low):
            # Try to hit the boundary with a smaller step.
            # Estimate the fraction of the predictor step that stays inside.
            lam = float(z[-1])
            denom = lam_pred - lam
            if direction > 0:
                frac = (p_high - lam) / denom if denom != 0.0 else 0.0
            else:
                frac = (p_low - lam) / denom if denom != 0.0 else 0.0
            if 0.0 < frac < 1.0:
                ds_boundary = frac * ds
                z_pred = z + direction * ds_boundary * tangent
                z_corr, ok, _ = _corrector(
                    z_pred, tangent, model, params, param_name, jac_builder, opts
                )
                if ok:
                    s += ds_boundary
                    z = z_corr
                    record(z, s, True)
            break

        z_corr, ok, n_iter = _corrector(
            z_pred, tangent, model, params, param_name, jac_builder, opts
        )

        if not ok:
            ds *= opts.adapt_shrink
            continue

        # Update arclength and state.
        s += abs(ds)
        z = z_corr

        # Tangent update, preserving orientation.
        tangent = _tangent_from_state(
            model, params, z, param_name, jac_builder, preference=tangent, opts=opts
        )

        record(z, s, True)

        # Adapt step size.
        if n_iter <= 3:
            ds = min(ds * opts.adapt_growth, opts.ds_max)
        elif n_iter >= opts.max_corrector_iter - 2:
            ds = max(ds * opts.adapt_shrink, opts.ds_min)

    return ContinuationResult(
        param_name=param_name,
        lambdas=np.array(lambdas),
        states=np.array(states),
        residuals=np.array(residuals),
        omegas=np.array(omegas),
        J_eigvals=J_eigvals,
        arclengths=np.array(arclengths),
        converged=np.array(converged_flags),
    )


def trace_branch_pseudo_arclength(
    model,
    params: dict,
    param_name: str,
    x0: np.ndarray | None = None,
    lam0: float | None = None,
    p_range: Sequence[float] | None = None,
    opts: ContinuationOptions | None = None,
) -> ContinuationResult:
    """
    Trace a steady-state branch in both directions using pseudo-arclength
    continuation.

    Parameters
    ----------
    model : Model
        R-matrix model.
    params : dict
        Base model parameters.  The value of ``param_name`` is overridden by
        the continuation parameter.
    param_name : str
        Name of the scalar parameter to vary.
    x0 : np.ndarray, optional
        Initial real-vector state.  If omitted, a steady-state solve at the
        initial parameter value is attempted.
    lam0 : float, optional
        Initial parameter value.  Defaults to ``params[param_name]``.
    p_range : sequence of float, optional
        ``[low, high]`` bounds for the continuation parameter.  Defaults to
        ``[lam0 - 0.1, lam0 + 0.1]``.
    opts : ContinuationOptions, optional
        Tuning parameters.

    Returns
    -------
    ContinuationResult
        Concatenated forward + backward trace, sorted by parameter value.
    """
    opts = opts or ContinuationOptions()

    if lam0 is None:
        lam0 = float(params[param_name])
    if p_range is None:
        p_range = [lam0 - 0.1, lam0 + 0.1]
    p_low, p_high = float(p_range[0]), float(p_range[1])

    if x0 is None:
        from numerics.solvers.steady_state import solve_steady_state
        R_guess = np.eye(model.dim, dtype=complex)
        res = solve_steady_state(
            model, _params_with(model, params, param_name, lam0),
            guess=R_guess, method="root", tol=opts.tol, use_jacobian=True,
        )
        if not res.success:
            raise RuntimeError(f"Could not find initial solution at {param_name}={lam0}")
        x0 = R_matrix_to_vector(res.R, model.dim)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.size != model.dim * model.dim:
            raise ValueError("x0 must be a real-vector of length dim**2")

    jac_builder = None
    try:
        jac_builder = model.build_jacobian_builder(params, modules="numpy")
    except Exception:
        jac_builder = None

    backward = _trace_one_direction(
        model, params, param_name, x0, lam0, -1.0, p_low, p_high,
        jac_builder, opts,
    )
    forward = _trace_one_direction(
        model, params, param_name, x0, lam0, +1.0, p_low, p_high,
        jac_builder, opts,
    )

    # Merge: exclude the duplicated initial point from one side.
    lambdas = np.concatenate([backward.lambdas[::-1][:-1], forward.lambdas])
    states = np.concatenate([backward.states[::-1][:-1], forward.states])
    residuals = np.concatenate([backward.residuals[::-1][:-1], forward.residuals])
    arclengths = np.concatenate(
        [(backward.arclengths[-1] - backward.arclengths[::-1])[:-1],
         forward.arclengths]
    )
    omegas = np.concatenate([backward.omegas[::-1][:-1], forward.omegas])
    converged = np.concatenate([backward.converged[::-1][:-1], forward.converged])
    J_eigvals = backward.J_eigvals[::-1][:-1] + forward.J_eigvals

    # Sort by parameter value for predictable output.
    order = np.argsort(lambdas)
    return ContinuationResult(
        param_name=param_name,
        lambdas=lambdas[order],
        states=states[order],
        residuals=residuals[order],
        omegas=omegas[order],
        J_eigvals=[J_eigvals[i] for i in order],
        arclengths=arclengths[order],
        converged=converged[order],
    )
