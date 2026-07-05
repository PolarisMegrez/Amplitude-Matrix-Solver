"""
Batched steady-state solver for GPU/CPU backends.

Solves many independent L(R)=0 problems simultaneously using a batched
Newton-Raphson iteration with a symbolic Jacobian.  Designed for small dense
systems (2-mode, 3-mode) where the parallelism comes from the number of
parameter points and initial guesses rather than the size of each linear solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from numerics.core.backend import get_array_module, get_backend, set_backend, to_numpy
from numerics.core.frequency import omega_frequency
from numerics.core.r_matrix import R_matrices_to_vectors, vectors_to_R_matrices
from numerics.utils.validation import convergence_message

if TYPE_CHECKING:
    from numerics.models.base import Model, SteadyStateResult


def _batched_residual_func(model: "Model", params: dict):
    """Return F(x) of shape (N, n^2) for a batch of real parameter vectors x."""

    def residual(x):
        xp = get_array_module(x)
        x = xp.asarray(x)
        R = vectors_to_R_matrices(x, model.dim)  # (N, n, n)
        L = model.liouvillian(R, params)  # (N, n, n)
        return R_matrices_to_vectors(L, model.dim)  # (N, n^2)

    return residual


def solve_steady_state_batched(
    model: "Model",
    params_batch: dict,
    guesses: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
    line_search: bool = True,
    compute_eigvals: bool = True,
    jac_builder=None,
    verbose: bool = False,
) -> list[list["SteadyStateResult"]]:
    """
    Solve a batch of steady-state problems with multiple initial guesses each.

    Parameters
    ----------
    model : Model
        R-matrix model.
    params_batch : dict
        Model parameters broadcast over the batch.  Scalar values are replicated;
        array values must have shape (B,).
    guesses : np.ndarray
        Initial guesses of shape (B, G, n, n) where B is batch size and G is
        the number of guesses per point.
    max_iter : int
        Maximum Newton iterations per guess.
    tol : float
        Convergence tolerance on the residual norm.
    line_search : bool
        If True, use a simple backtracking line search (slower but more robust).
    compute_eigvals : bool
        If True, compute Jacobian eigenvalues for stability analysis on the
        active backend.
    jac_builder : JacobianBuilder, optional
        Pre-built backend-aware Jacobian builder.  If None, one is built from
        ``model.build_jacobian_builder`` using the active backend.

    Returns
    -------
    results : list[list[SteadyStateResult]]
        Outer list length B, inner list length G.  Unconverged results are still
        returned with success=False.
    """
    from numerics.models.base import SteadyStateResult

    xp = get_array_module()
    guesses = xp.asarray(guesses)
    if guesses.ndim != 4:
        raise ValueError("guesses must have shape (B, G, n, n)")

    B, G, n, _ = guesses.shape
    n2 = n * n

    # Broadcast scalar params to shape (B,), then repeat G times for the
    # flattened (B*G,) layout used inside the Newton loop.
    params_B: dict[str, np.ndarray] = {}
    for k, v in params_batch.items():
        arr = xp.asarray(v)
        if arr.ndim == 0:
            arr = xp.broadcast_to(arr, (B,))
        elif arr.shape != (B,):
            raise ValueError(f"Parameter '{k}' has shape {arr.shape}, expected (B,) or scalar")
        params_B[k] = arr

    params_BG = {k: xp.repeat(v, G) for k, v in params_B.items()}

    # Initial x: (B, G, n^2)
    x = R_matrices_to_vectors(guesses.reshape(B * G, n, n)).reshape(B, G, n2)

    residual = _batched_residual_func(model, params_BG)

    if jac_builder is None:
        backend_module = get_backend()
        jac_builder = model.build_jacobian_builder(params_BG, modules=backend_module)
        if jac_builder is None:
            raise ValueError(
                "Model does not provide a Jacobian builder; "
                "batched Newton requires a symbolic Jacobian."
            )

    # Convergence mask: (B, G)
    converged = xp.zeros((B, G), dtype=bool)
    residual_norms = xp.full((B, G), xp.inf)

    for it in range(max_iter):
        x_flat = x.reshape(B * G, n2)
        F_flat = residual(x_flat)  # (B*G, n2)
        F = F_flat.reshape(B, G, n2)

        norms = xp.linalg.norm(F, axis=-1)  # (B, G)
        residual_norms = xp.where(~converged, norms, residual_norms)
        newly_converged = norms < tol
        converged = converged | newly_converged

        if verbose and it % 10 == 0:
            print(
                f"  iter {it}: max residual {float(xp.max(norms)):.3e}, "
                f"converged {int(xp.sum(converged))}/{B * G}"
            )

        if xp.all(converged):
            break

        # Symbolic Jacobian evaluated for the whole batch
        J = jac_builder.evaluate_batched(x_flat, params_BG)  # (B*G, n2, n2)
        J = J.reshape(B, G, n2, n2)

        # Solve J dx = -F for each (b, g)
        dx = xp.linalg.solve(J, -F[..., None])[..., 0]  # (B, G, n2)

        if line_search:
            alpha = 1.0
            for _ in range(5):
                x_new = x + alpha * dx
                F_new = residual(x_new.reshape(B * G, n2)).reshape(B, G, n2)
                new_norms = xp.linalg.norm(F_new, axis=-1)
                improved = new_norms < norms
                if xp.all(~(~converged) | improved) or alpha < 0.125:
                    x = x + alpha * dx
                    break
                alpha *= 0.5
            else:
                x = x + alpha * dx
        else:
            x = x + dx

    # Final residual and converged status
    x_flat = x.reshape(B * G, n2)
    F_flat = residual(x_flat)
    F = F_flat.reshape(B, G, n2)
    final_norms = xp.linalg.norm(F, axis=-1)
    converged = final_norms < tol

    R_final = vectors_to_R_matrices(x_flat, n).reshape(B, G, n, n)
    omega_final = omega_frequency(
        model.H(R_final.reshape(B * G, n, n), params_BG),
        R_final.reshape(B * G, n, n),
    ).reshape(B, G)

    # Jacobian eigenvalues for stability, computed on the active backend
    J_eigvals_final = None
    if compute_eigvals:
        J_final = jac_builder.evaluate_batched(x_flat, params_BG)  # (B*G, n2, n2)
        J_eigvals_final = xp.linalg.eigvals(J_final).reshape(B, G, n2)

    results: list[list[SteadyStateResult]] = []
    for b in range(B):
        params_cpu = {k: float(to_numpy(v[b])) for k, v in params_B.items()}
        row = []
        for g in range(G):
            R_bg = to_numpy(R_final[b, g])
            success = bool(to_numpy(converged[b, g]))
            Je = to_numpy(J_eigvals_final[b, g]) if J_eigvals_final is not None else None
            row.append(SteadyStateResult(
                R=R_bg,
                params=params_cpu,
                success=success,
                residual=float(to_numpy(final_norms[b, g])),
                method="batched-newton",
                message=convergence_message(
                    success, float(to_numpy(final_norms[b, g])), tol
                ),
                omega=float(to_numpy(omega_final[b, g])),
                J_eigvals=Je,
                iterations=it + 1,
            ))
        results.append(row)

    return results


