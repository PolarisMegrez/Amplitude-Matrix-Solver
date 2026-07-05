"""
Vectorized post-processing of converged steady-state R matrices.

After all fixed points R have been discovered, this module computes
omega, Jacobian eigenvalues, PSD, and stability in large batched calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from numerics.core.backend import get_array_module, get_backend, to_numpy
from numerics.core.frequency import omega_frequency
from numerics.core.liouvillian import liouvillian_from_model
from numerics.core.r_matrix import (
    R_matrices_to_vectors,
    R_matrix_to_vector,
    vectors_to_R_matrices,
)
from numerics.scans.multistability import MultistabilityScanResult


def _params_grid_from_list(params_list: list[dict]) -> dict[str, np.ndarray]:
    """Turn a list of parameter dicts into a dict of 1-D arrays."""
    if not params_list:
        return {}
    keys = list(params_list[0].keys())
    return {k: np.array([p[k] for p in params_list]) for k in keys}


def compute_omegas(
    model,
    R_stack: np.ndarray,
    params_grid: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized omega for a stack of R matrices."""
    xp = get_array_module(R_stack)
    R_stack = xp.asarray(R_stack)
    params_grid = {k: xp.asarray(v) for k, v in params_grid.items()}
    H = model.H(R_stack, params_grid)
    return to_numpy(omega_frequency(H, R_stack))


def compute_jacobian_eigvals_symbolic(
    model,
    R_stack: np.ndarray,
    params_grid: dict[str, np.ndarray],
    modules: str = "numpy",
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Vectorized symbolic Jacobian eigenvalues for a stack of R matrices.

    Uses the model's symbolic Jacobian builder; falls back to numerical
    differences if the model does not provide one.
    """
    builder = model.build_jacobian_builder(params_grid, modules=modules)
    if builder is None:
        return compute_jacobian_eigvals_numerical(model, R_stack, params_grid, chunk_size=chunk_size)

    xp = get_array_module(modules)
    R_stack = xp.asarray(R_stack)
    params_grid = {k: xp.asarray(v) for k, v in params_grid.items()}
    N = int(R_stack.shape[0])
    n = model.dim
    n2 = n * n

    r_stack = R_matrices_to_vectors(R_stack, n)
    all_eigvals: list[np.ndarray] = []
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        r_chunk = r_stack[start:end]
        params_chunk = {k: v[start:end] for k, v in params_grid.items()}
        J = builder.evaluate_batched(r_chunk, params_chunk)  # (M, n2, n2)
        try:
            eigvals = xp.linalg.eigvals(J)
        except (xp.linalg.LinAlgError, ValueError):
            eigvals = xp.full((end - start, n2), xp.nan + 0j * xp.nan)
        all_eigvals.append(to_numpy(eigvals))
    return np.concatenate(all_eigvals, axis=0) if len(all_eigvals) > 1 else all_eigvals[0]


def compute_jacobian_eigvals_numerical(
    model,
    R_stack: np.ndarray,
    params_grid: dict[str, np.ndarray],
    eps: float = 1e-7,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Vectorized numerical Jacobian eigenvalues for a stack of R matrices.

    A forward-difference Jacobian is built for the whole stack in a few
    batched residual evaluations, then ``xp.linalg.eigvals`` is applied
    to each Jacobian matrix.
    """
    xp = get_array_module(R_stack)
    R_stack = xp.asarray(R_stack)
    params_grid = {k: xp.asarray(v) for k, v in params_grid.items()}
    n = model.dim
    n2 = n * n
    N = int(R_stack.shape[0])

    def residual_from_R(R: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        L = liouvillian_from_model(model, R, params)
        return R_matrices_to_vectors(L, n)

    def residual_from_x(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        R = vectors_to_R_matrices(x, n)
        return residual_from_R(R, params)

    all_eigvals: list[np.ndarray] = []
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        R_chunk = R_stack[start:end]
        params_chunk = {k: v[start:end] for k, v in params_grid.items()}

        x0 = R_matrices_to_vectors(R_chunk, n)  # (M, n2)
        M = int(x0.shape[0])

        F0 = residual_from_x(x0, params_chunk)  # (M, n2)

        # x_pert[m, j, :] = x0[m] + eps * e_j
        eye = xp.eye(n2, dtype=float)
        x_pert = x0[:, None, :] + eps * eye  # (M, n2, n2)
        x_pert_flat = x_pert.reshape(M * n2, n2)
        params_pert = {k: xp.repeat(v, n2) for k, v in params_chunk.items()}
        F_pert = residual_from_x(x_pert_flat, params_pert)  # (M*n2, n2)
        F_pert = F_pert.reshape(M, n2, n2)

        # J[m, k, j] = dF_k / dx_j
        J = (F_pert - F0[:, None, :]) / eps
        J = xp.transpose(J, (0, 2, 1))  # (M, n2, n2)

        try:
            eigvals = xp.linalg.eigvals(J)  # (M, n2)
        except (xp.linalg.LinAlgError, ValueError):
            eigvals = xp.full((M, n2), xp.nan + 0j * xp.nan)
        all_eigvals.append(to_numpy(eigvals))

    return np.concatenate(all_eigvals, axis=0) if len(all_eigvals) > 1 else all_eigvals[0]


def compute_psd_and_stability(
    R_stack: np.ndarray,
    J_eigvals: np.ndarray,
    psd_atol: float = -1e-8,
    stable_atol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized PSD and stability flags."""
    xp = get_array_module(R_stack)
    R_stack = xp.asarray(R_stack)
    J_eigvals = xp.asarray(J_eigvals)
    psd = xp.array([xp.all(xp.linalg.eigvalsh(R) > psd_atol) for R in R_stack])
    stable = xp.real(J_eigvals).max(axis=-1) < stable_atol
    return to_numpy(psd), to_numpy(stable)


def postprocess_result(
    result: MultistabilityScanResult,
    model,
    chunk_size: int = 2048,
) -> MultistabilityScanResult:
    """
    Recompute omega, Jacobian eigenvalues, PSD, and stability for all
    converged branches in ``result`` using vectorized batched operations.
    """
    valid_mask = ~np.isnan(result.R_matrices[..., 0, 0])
    if not np.any(valid_mask):
        return result

    indices = np.argwhere(valid_mask)
    R_stack = np.array([result.R_matrices[i, j, k] for i, j, k in indices])
    params_list = [{
        **result.base_params,
        list(result.axes.keys())[0]: float(result.axes[list(result.axes.keys())[0]][i]),
        list(result.axes.keys())[1]: float(result.axes[list(result.axes.keys())[1]][j]),
    } for i, j, _ in indices]
    params_grid = _params_grid_from_list(params_list)

    backend = get_backend()
    omegas = compute_omegas(model, R_stack, params_grid)
    J_eigvals = compute_jacobian_eigvals_symbolic(
        model, R_stack, params_grid, modules=backend, chunk_size=chunk_size
    )
    psd, stable = compute_psd_and_stability(R_stack, J_eigvals)

    for (i, j, k), o, Je, p, s in zip(indices, omegas, J_eigvals, psd, stable):
        result.omegas[i, j, k] = o
        result.J_eigvals[i, j, k] = Je
        result.is_psd[i, j, k] = bool(p)
        result.is_stable[i, j, k] = bool(s)

    return result
