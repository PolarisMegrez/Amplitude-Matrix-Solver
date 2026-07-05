"""
Frequency extraction for steady states.

The physical frequency associated with a fixed point R0 is

    omega = Re Tr(H(R0) R0) / Tr(R0).

For rank-one R0 = alpha alpha^dagger, this coincides with the eigenvalue of H
along the coherent state.
"""

from __future__ import annotations

from numerics.core.backend import get_array_module


def omega_frequency(H, R):
    """
    Compute omega = Re Tr(H R) / Tr(R).

    Supports single matrices or batches.  For batched input H and R have
    shape (..., n, n); the returned array has shape (...).

    Parameters
    ----------
    H : np.ndarray
        Effective Hamiltonian at R.
    R : np.ndarray
        Hermitian steady-state matrix.

    Returns
    -------
    omega : float or np.ndarray
        Oscillation frequency.
    """
    xp = get_array_module(H)
    H = xp.asarray(H)
    R = xp.asarray(R)

    # batched trace of matrix product: trace(H @ R, axis1=-2, axis2=-1)
    numerator = xp.trace(H @ R, axis1=-2, axis2=-1)
    denominator = xp.trace(R, axis1=-2, axis2=-1)

    # Avoid division by zero
    safe = xp.abs(denominator) >= 1e-300
    omega = xp.where(safe, xp.real(numerator / denominator), 0.0)

    if omega.shape == ():
        return float(omega)
    return omega


def omega_from_H_eigvals(H):
    """
    Return the real parts of all eigenvalues of H.

    Useful for comparing with omega_frequency for rank-one states.
    """
    xp = get_array_module(H)
    H = xp.asarray(H)
    return xp.real(xp.linalg.eigvals(H))
