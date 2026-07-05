"""
Adaptive initial-guess bounds for R-matrix steady-state solvers.

The idea is to exploit the *diagonal balance* of the Liouvillian:

    L_ii(R) = -i H_ii(R) R_ii + i R_ii H_ii^dagger(R) + D_ii(R)
            = 2 R_ii Im H_ii(R) + D_ii(R)

For a diagonal R this is a small system of real equations that depends only on
model parameters and the diagonal entries.  Solving it gives characteristic
scales (and signs) of the steady-state diagonal elements.  Those scales are
used to build a much better random-guess distribution than a fixed ``scale``.

This is model-agnostic: it only calls ``model.H`` and ``model.D`` and works for
any R-matrix model (Kerr, van der Pol, Hopf, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import root


@dataclass
class GuessBounds:
    """
    Per-element sampling bounds for generating Hermitian initial guesses.

    Parameters
    ----------
    diag_lower, diag_upper : np.ndarray
        Lower / upper bound for each diagonal entry.
    offdiag_scale : np.ndarray
        Symmetric matrix of typical scales for the upper-triangle entries.
    diag_candidates : np.ndarray | None
        Representative diagonal vectors obtained from the diagonal balance.
        These are used to build a deterministic seed grid.
    """

    diag_lower: np.ndarray
    diag_upper: np.ndarray
    offdiag_scale: np.ndarray
    diag_candidates: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.diag_lower = np.asarray(self.diag_lower, dtype=float)
        self.diag_upper = np.asarray(self.diag_upper, dtype=float)
        self.offdiag_scale = np.asarray(self.offdiag_scale, dtype=float)
        if self.diag_candidates is not None:
            self.diag_candidates = np.asarray(self.diag_candidates, dtype=float)

    @property
    def dim(self) -> int:
        return int(self.diag_lower.shape[0])

    def sample(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """Generate ``n_samples`` random Hermitian guesses inside the bounds."""
        rng = np.random.default_rng(seed)
        n = self.dim
        guesses: list[np.ndarray] = []
        for _ in range(n_samples):
            R = np.zeros((n, n), dtype=complex)
            # Diagonal entries: uniform over the inferred signed range
            for i in range(n):
                R[i, i] = rng.uniform(self.diag_lower[i], self.diag_upper[i])
            # Off-diagonal entries: normal with per-pair scale
            for i in range(n):
                for j in range(i + 1, n):
                    s = self.offdiag_scale[i, j]
                    re = rng.normal(0.0, s)
                    im = rng.normal(0.0, s)
                    R[i, j] = re + 1j * im
                    R[j, i] = re - 1j * im
            guesses.append(R)
        return guesses


def _diagonal_residual(model, params: dict):
    """Return F(r_diag) = Re diag(L(diag(r_diag), params))."""
    n = model.dim

    def F(r: np.ndarray) -> np.ndarray:
        R = np.diag(np.asarray(r, dtype=float)).astype(complex)
        H = model.H(R, params)
        D = model.D(R, params)
        # L_ii = 2 R_ii Im(H_ii) + D_ii
        return 2.0 * np.real(np.diag(R)) * np.imag(np.diag(H)) + np.real(np.diag(D))

    return F


def _unique_rows(arr: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Deduplicate rows of ``arr`` by Euclidean distance."""
    if arr.size == 0:
        return arr
    unique: list[np.ndarray] = []
    for row in arr:
        if not unique:
            unique.append(row)
            continue
        dists = [np.linalg.norm(row - u) for u in unique]
        if min(dists) > tol * max(1.0, np.linalg.norm(row)):
            unique.append(row)
    return np.array(unique)


def infer_diagonal_roots(
    model,
    params: dict,
    r_min: float = 1e-4,
    r_max: float = 1e6,
    n_log: int = 7,
    residual_tol: float = 1e-6,
) -> np.ndarray:
    """
    Find real roots of the diagonal balance L_ii(diag(r)) = 0.

    A logarithmic grid of starting points (positive and negative, plus zero) is
    used so that both small and large roots are recovered without model-specific
    tuning.

    Returns
    -------
    roots : np.ndarray
        Array of shape ``(n_roots, dim)`` with the recovered diagonal vectors.
        May be empty if no roots are found.
    """
    F = _diagonal_residual(model, params)
    n = model.dim
    log_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_log)
    per_dim = [np.concatenate([[0.0], log_vals, -log_vals]) for _ in range(n)]
    starts = np.array(np.meshgrid(*per_dim, indexing="ij")).reshape(n, -1).T

    roots_found: list[np.ndarray] = []
    for s in starts:
        try:
            sol = root(F, np.real(s), method="hybr", tol=residual_tol)
        except Exception:
            continue
        if not sol.success:
            continue
        r = np.real(sol.x)
        if np.linalg.norm(F(r)) <= 10.0 * residual_tol:
            roots_found.append(r)

    if not roots_found:
        return np.empty((0, n))
    return _unique_rows(np.array(roots_found), tol=1e-3)


def _fallback_scale(model, params: dict) -> float:
    """Parameter-aware fallback scale when diagonal roots cannot be found."""
    n = model.dim
    R0 = np.zeros((n, n), dtype=complex)
    try:
        H0 = model.H(R0, params)
        D0 = model.D(R0, params)
        imH = np.imag(np.diag(H0))
        d = np.real(np.diag(D0))
        safe = np.abs(imH) > 1e-12
        scales = np.where(
            safe,
            0.5 * np.abs(d) / np.abs(imH),
            np.full(n, 1.0),
        )
        return float(np.max(scales))
    except Exception:
        return 100.0


def infer_guess_bounds(
    model,
    params: dict,
    fallback_scale: float = 100.0,
    margin: float = 2.0,
    r_max: float = 1e6,
) -> GuessBounds:
    """
    Infer model- and parameter-aware initial-guess bounds.

    The bounds are built from the diagonal-balance roots.  They automatically
    adapt to the order of magnitude and sign of the steady-state diagonal
    entries.

    Parameters
    ----------
    model : Model
        R-matrix model.
    params : dict
        Parameter dictionary.
    fallback_scale : float
        Scale used when the diagonal balance yields no roots.
    margin : float
        Extra multiplicative margin applied on each side of the root range.
    r_max : float
        Largest magnitude scanned by ``infer_diagonal_roots``.

    Returns
    -------
    GuessBounds
        Sampling bounds and diagonal candidates for seed generation.
    """
    n = model.dim
    roots = infer_diagonal_roots(model, params, r_max=r_max)

    if len(roots) == 0:
        scale = max(fallback_scale, _fallback_scale(model, params))
        lower = np.full(n, -scale)
        upper = np.full(n, scale)
        candidates = np.zeros((1, n))
    else:
        # Always include zero so non-PSD / trivial branches are not excluded.
        candidates = _unique_rows(
            np.vstack([roots, np.zeros((1, n))]), tol=1e-3
        )
        lower_raw = np.min(candidates, axis=0)
        upper_raw = np.max(candidates, axis=0)

        lower = np.where(lower_raw < 0, lower_raw * margin, -fallback_scale * 0.1)
        upper = np.where(upper_raw > 0, upper_raw * margin, fallback_scale * 0.1)

    # Per-pair off-diagonal scale: geometric mean of the two diagonal ranges.
    offdiag = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ri = upper[i] - lower[i]
            rj = upper[j] - lower[j]
            offdiag[i, j] = offdiag[j, i] = 0.5 * np.sqrt(max(ri * rj, 1e-8))

    return GuessBounds(
        diag_lower=lower,
        diag_upper=upper,
        offdiag_scale=offdiag,
        diag_candidates=candidates,
    )


def merge_guess_bounds(bounds_list: Sequence[GuessBounds]) -> GuessBounds:
    """Merge several bounds objects by taking the widest ranges."""
    if not bounds_list:
        raise ValueError("Cannot merge empty list of GuessBounds")
    n = bounds_list[0].dim
    lower = np.min([b.diag_lower for b in bounds_list], axis=0)
    upper = np.max([b.diag_upper for b in bounds_list], axis=0)
    offdiag = np.max([b.offdiag_scale for b in bounds_list], axis=0)
    candidates = np.vstack(
        [b.diag_candidates for b in bounds_list if b.diag_candidates is not None]
    )
    candidates = _unique_rows(candidates, tol=1e-3) if len(candidates) else None
    return GuessBounds(
        diag_lower=lower,
        diag_upper=upper,
        offdiag_scale=offdiag,
        diag_candidates=candidates,
    )
