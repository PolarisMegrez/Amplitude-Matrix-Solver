"""
Real-vector parameterization of Hermitian matrices.

Ordering convention (consistent with Tensor/jacobian_tools.py):
    1. Diagonal real entries: R_11, R_22, ..., R_nn
    2. Upper-triangle real parts: Re R_12, Re R_13, ..., Re R_(n-1,n)
    3. Upper-triangle imaginary parts: Im R_12, Im R_13, ..., Im R_(n-1,n)
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Union

from numerics.core.backend import get_array_module

Array = Union[np.ndarray]


def get_parameterized_R(n: int) -> tuple[sp.Matrix, sp.Matrix]:
    """
    Symbolic n x n Hermitian matrix R and the corresponding real parameter vector.

    Returns
    -------
    R : sympy.Matrix
        Hermitian matrix with symbolic entries.
    r_vec : sympy.Matrix
        Column vector of real parameters in the standard order.
    """
    R = sp.zeros(n, n)
    r_vars = []

    # Diagonal entries
    for i in range(n):
        sym = sp.Symbol(f"R_{i+1}{i+1}", real=True)
        R[i, i] = sym
        r_vars.append(sym)

    # Upper-triangle real and imaginary parts
    re_vars = []
    im_vars = []
    for i in range(n):
        for j in range(i + 1, n):
            R_re = sp.Symbol(f"R_{i+1}{j+1}_re", real=True)
            R_im = sp.Symbol(f"R_{i+1}{j+1}_im", real=True)
            re_vars.append(R_re)
            im_vars.append(R_im)
            R[i, j] = R_re + sp.I * R_im
            R[j, i] = R_re - sp.I * R_im

    r_vec = sp.Matrix(r_vars + re_vars + im_vars)
    return R, r_vec


def R_matrix_to_vector(R: Array, n: int | None = None) -> np.ndarray:
    """
    Flatten a Hermitian matrix into a real vector.

    Ordering convention (consistent with get_parameterized_R):
        1. Diagonal real entries: R_11, R_22, ..., R_nn
        2. Upper-triangle real parts: Re R_12, Re R_13, ..., Re R_(n-1,n)
        3. Upper-triangle imaginary parts: Im R_12, Im R_13, ..., Im R_(n-1,n)

    Parameters
    ----------
    R : np.ndarray
        n x n Hermitian matrix.
    n : int, optional
        Matrix dimension. Inferred if not given.

    Returns
    -------
    vec : np.ndarray
        Real vector of length n + n*(n-1).
    """
    xp = get_array_module(R)
    R = xp.asarray(R)
    if n is None:
        n = int(R.shape[0])

    diag = xp.real(xp.diag(R))

    re_parts = []
    im_parts = []
    for i in range(n):
        for j in range(i + 1, n):
            re_parts.append(xp.real(R[i, j]))
            im_parts.append(xp.imag(R[i, j]))

    return xp.concatenate([diag, re_parts, im_parts])


def vector_to_R_matrix(vec: Array, n: int | None = None) -> np.ndarray:
    """
    Reconstruct a Hermitian matrix from its real-vector parameterization.

    Parameters
    ----------
    vec : np.ndarray
        Real vector of length n + n*(n-1).
    n : int, optional
        Matrix dimension. Inferred from vector length if not given.

    Returns
    -------
    R : np.ndarray
        n x n Hermitian matrix.
    """
    xp = get_array_module(vec)
    vec = xp.asarray(vec, dtype=float)
    if n is None:
        n = int(xp.sqrt(len(vec)))
        if n * n != len(vec):
            raise ValueError(f"Vector length {len(vec)} is not a perfect square.")

    R = xp.zeros((n, n), dtype=xp.complex128)

    for i in range(n):
        R[i, i] = vec[i]

    n_upper = n * (n - 1) // 2
    idx = n
    for i in range(n):
        for j in range(i + 1, n):
            re = vec[idx]
            im = vec[idx + n_upper]
            idx += 1
            R[i, j] = re + 1j * im
            R[j, i] = re - 1j * im

    return R


def R_matrices_to_vectors(R: Array, n: int | None = None) -> np.ndarray:
    """
    Batched version of R_matrix_to_vector.

    Parameters
    ----------
    R : np.ndarray
        Array of shape (..., n, n).
    n : int, optional
        Matrix dimension.

    Returns
    -------
    vec : np.ndarray
        Array of shape (..., n*n).
    """
    xp = get_array_module(R)
    R = xp.asarray(R)
    if n is None:
        n = int(R.shape[-1])

    batch_shape = R.shape[:-2]
    diag = xp.real(xp.diagonal(R, axis1=-2, axis2=-1))  # (..., n)

    re_parts = []
    im_parts = []
    for i in range(n):
        for j in range(i + 1, n):
            re_parts.append(xp.real(R[..., i, j]))
            im_parts.append(xp.imag(R[..., i, j]))

    re_stack = xp.stack(re_parts, axis=-1)  # (..., n_upper)
    im_stack = xp.stack(im_parts, axis=-1)  # (..., n_upper)
    return xp.concatenate([diag, re_stack, im_stack], axis=-1)


def vectors_to_R_matrices(vec: Array, n: int | None = None) -> np.ndarray:
    """
    Batched version of vector_to_R_matrix.

    Parameters
    ----------
    vec : np.ndarray
        Array of shape (..., n*n).
    n : int, optional
        Matrix dimension.

    Returns
    -------
    R : np.ndarray
        Array of shape (..., n, n).
    """
    xp = get_array_module(vec)
    vec = xp.asarray(vec, dtype=float)
    if n is None:
        n = int(xp.sqrt(vec.shape[-1]))
        if n * n != vec.shape[-1]:
            raise ValueError(f"Vector length {vec.shape[-1]} is not a perfect square.")

    batch_shape = vec.shape[:-1]
    R = xp.zeros(batch_shape + (n, n), dtype=xp.complex128)

    for i in range(n):
        R[..., i, i] = vec[..., i]

    n_upper = n * (n - 1) // 2
    idx = n
    for i in range(n):
        for j in range(i + 1, n):
            re = vec[..., idx]
            im = vec[..., idx + n_upper]
            idx += 1
            R[..., i, j] = re + 1j * im
            R[..., j, i] = re - 1j * im

    return R


def is_hermitian(R: Array, atol: float = 1e-10) -> bool:
    """Check whether a matrix is Hermitian within tolerance."""
    xp = get_array_module(R)
    R = xp.asarray(R)
    return bool(xp.allclose(R, R.conj().T, atol=atol))
