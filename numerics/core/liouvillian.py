"""
Liouvillian superoperator L(R) = -i H(R) R + i R H^dagger(R) + D(R).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numerics.core.backend import get_array_module

if TYPE_CHECKING:
    from numerics.models.base import Model


def liouvillian(H, R, D):
    """
    Evaluate L(R) = -i H R + i R H^dagger + D.

    Parameters
    ----------
    H, R, D : np.ndarray
        Square complex matrices of the same shape, or batches of shape
        (..., n, n).

    Returns
    -------
    L : np.ndarray
        Liouvillian output matrix.
    """
    xp = get_array_module(H)
    H = xp.asarray(H)
    R = xp.asarray(R)
    D = xp.asarray(D)
    Hdag = xp.conj(xp.swapaxes(H, -1, -2))
    return -1j * H @ R + 1j * R @ Hdag + D


def liouvillian_from_model(model: "Model", R, params: dict):
    """
    Evaluate L(R) using a Model instance.

    Parameters
    ----------
    model : Model
        Model providing H(R, params) and D(R, params).
    R : np.ndarray
        Hermitian state matrix, or batch of matrices.
    params : dict
        Model parameters.

    Returns
    -------
    L : np.ndarray
        Liouvillian output.
    """
    H = model.H(R, params)
    D = model.D(R, params)
    return liouvillian(H, R, D)


def liouvillian_vector(model: "Model", r_vec: np.ndarray, params: dict) -> np.ndarray:
    """
    Evaluate the real-vector residual F(r_vec) = vec(L(R(r_vec))).

    Parameters
    ----------
    model : Model
        Model instance.
    r_vec : np.ndarray
        Real parameter vector of length n^2 (or batch of shape (..., n^2)).
    params : dict
        Model parameters.

    Returns
    -------
    F : np.ndarray
        Real-vector residual of the same shape as r_vec.
    """
    from numerics.core.r_matrix import R_matrices_to_vectors, vectors_to_R_matrices

    xp = get_array_module(r_vec)
    r_vec = xp.asarray(r_vec, dtype=float)
    was_1d = r_vec.ndim == 1
    if was_1d:
        r_vec = r_vec[None, :]
    R = vectors_to_R_matrices(r_vec, model.dim)
    L = liouvillian_from_model(model, R, params)
    F = R_matrices_to_vectors(L, model.dim)
    if was_1d:
        F = F[0]
    return xp.asarray(F, dtype=float)


def liouvillian_residual(L, ord: str | None = "fro") -> float:
    """
    Residual norm for the steady-state equation L(R0) = 0.

    Parameters
    ----------
    L : np.ndarray
        Liouvillian output.
    ord : str or None
        Norm order passed to numpy.linalg.norm.

    Returns
    -------
    residual : float
        Norm of L.
    """
    xp = get_array_module(L)
    L = xp.asarray(L)
    return float(xp.linalg.norm(L, ord=ord))
