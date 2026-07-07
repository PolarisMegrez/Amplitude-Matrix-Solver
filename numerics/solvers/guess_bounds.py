"""
Adaptive initial-guess bounds for R-matrix steady-state solvers.

Two complementary strategies are provided:

1. **Diagonal-balance inference** (cheap, analytic-ish):
   solve the scalar equations ``L_ii(diag(r)) = 0`` to get typical diagonal
   scales.  This works for any model but can underestimate the range when
   off-diagonal coupling or bifurcations push solutions far from the diagonal
   subspace.

2. **Exploration-based inference** (more expensive but robust):
   run a short multi-start solve with heavy-tailed log-uniform guesses, then
   measure the actual range of the converged solutions.  Because it uses the
   full model, it naturally captures coupling effects and bifurcation
   branches that the diagonal balance misses.

By default ``infer_guess_bounds`` uses exploration and falls back to the
 diagonal balance if the cheap solve does not find anything.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        Representative diagonal vectors (e.g. from roots or from converged
        solutions) used to build a deterministic seed grid.
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
        tail_fraction: float = 0.25,
        tail_orders: float = 3.0,
    ) -> list[np.ndarray]:
        """
        Generate ``n_samples`` random Hermitian guesses.

        The distribution is a mixture:

        * ``1 - tail_fraction`` of the guesses are sampled uniformly inside the
          inferred bounds (exploitation around the typical scale).
        * ``tail_fraction`` are sampled from a signed log-uniform tail spanning
          ``tail_orders`` of magnitude beyond the bounds.  This catches
          bifurcation branches whose R elements are much larger than the
          diagonal-balance estimate.
        """
        rng = np.random.default_rng(seed)
        n = self.dim
        guesses: list[np.ndarray] = []

        # Typical scales used for the heavy tail.
        typical = np.maximum(np.abs(self.diag_lower), np.abs(self.diag_upper))
        typical = np.where(typical > 0.0, typical, 1.0)

        for _ in range(n_samples):
            R = np.zeros((n, n), dtype=complex)
            use_tail = rng.random() < tail_fraction

            for i in range(n):
                if use_tail:
                    # Signed log-uniform: several orders of magnitude around the
                    # inferred typical scale.
                    log_mag = rng.uniform(
                        np.log10(typical[i]) - tail_orders,
                        np.log10(typical[i]) + tail_orders,
                    )
                    mag = 10.0**log_mag
                    sign = rng.choice([-1.0, 1.0])
                    R[i, i] = sign * mag
                else:
                    R[i, i] = rng.uniform(self.diag_lower[i], self.diag_upper[i])

            for i in range(n):
                for j in range(i + 1, n):
                    s = self.offdiag_scale[i, j]
                    if use_tail:
                        s_tail = max(s, 10.0 ** (rng.uniform(
                            np.log10(max(s, 1e-8)) - tail_orders,
                            np.log10(max(s, 1e-8)) + tail_orders,
                        )))
                        s = s_tail
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


def _guess_bounds_from_roots(
    model,
    params: dict,
    fallback_scale: float = 100.0,
    margin: float = 2.0,
    r_max: float = 1e6,
) -> GuessBounds:
    """Build bounds from diagonal-balance roots."""
    n = model.dim
    roots = infer_diagonal_roots(model, params, r_max=r_max)

    if len(roots) == 0:
        scale = max(fallback_scale, _fallback_scale(model, params))
        lower = np.full(n, -scale)
        upper = np.full(n, scale)
        candidates = np.zeros((1, n))
    else:
        candidates = _unique_rows(np.vstack([roots, np.zeros((1, n))]), tol=1e-3)
        lower_raw = np.min(candidates, axis=0)
        upper_raw = np.max(candidates, axis=0)

        lower = np.where(lower_raw < 0, lower_raw * margin, -fallback_scale * 0.1)
        upper = np.where(upper_raw > 0, upper_raw * margin, fallback_scale * 0.1)

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


def _make_log_uniform_guesses(
    dim: int,
    n_samples: int,
    r_min: float = 1e-4,
    r_max: float = 1e6,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate heavy-tailed Hermitian guesses with log-uniform diagonals."""
    rng = np.random.default_rng(seed)
    guesses: list[np.ndarray] = []
    log_min, log_max = np.log10(r_min), np.log10(r_max)
    for _ in range(n_samples):
        R = np.zeros((dim, dim), dtype=complex)
        diag_mags = []
        for i in range(dim):
            sign = rng.choice([-1.0, 1.0])
            mag = 10.0 ** rng.uniform(log_min, log_max)
            R[i, i] = sign * mag
            diag_mags.append(abs(mag))
        for i in range(dim):
            for j in range(i + 1, dim):
                s = 0.5 * np.sqrt(diag_mags[i] * diag_mags[j])
                re = rng.normal(0.0, max(s, r_min))
                im = rng.normal(0.0, max(s, r_min))
                R[i, j] = re + 1j * im
                R[j, i] = re - 1j * im
        guesses.append(R)
    return guesses


def explore_solution_bounds(
    model,
    params: dict,
    n_samples: int = 300,
    r_min: float = 1e-4,
    r_max: float = 1e6,
    seed: int = 0,
    residual_tol: float = 1e-4,
    margin: float = 2.0,
) -> GuessBounds | None:
    """
    Infer guess bounds by actually solving the model from heavy-tailed guesses.

    This is more expensive than ``_guess_bounds_from_roots`` but captures
    branches whose R elements are driven far from the diagonal-balance scale by
    coupling or bifurcations.

    Returns ``None`` if no solutions are found, so the caller can fall back.
    """
    n = model.dim
    guesses = _make_log_uniform_guesses(n, n_samples, r_min, r_max, seed)
    candidates: list = []

    # Prefer the batched solver for speed; fall back to sequential solves if
    # the model does not expose a symbolic Jacobian builder.
    try:
        from numerics.solvers.batched import solve_steady_state_batched

        params_batch = {k: np.array([v]) for k, v in params.items()}
        guesses_arr = np.asarray(guesses).reshape(1, n_samples, n, n)
        results = solve_steady_state_batched(
            model,
            params_batch,
            guesses_arr,
            max_iter=50,
            tol=1e-10,
            line_search=True,
            compute_eigvals=False,
        )
        candidates = [r for r in results[0] if r.success and r.residual <= residual_tol]
    except Exception:
        pass

    if not candidates:
        from numerics.solvers.steady_state import solve_steady_state

        for g in guesses:
            try:
                res = solve_steady_state(
                    model, params, guess=g, method="root", tol=1e-10, use_jacobian=False
                )
                if res.success and res.residual <= residual_tol:
                    candidates.append(res)
            except Exception:
                continue

    if not candidates:
        return None

    # Deduplicate converged solutions.
    from numerics.core.r_matrix import R_matrix_to_vector

    seen: list[np.ndarray] = []
    unique: list = []
    for r in candidates:
        vec = R_matrix_to_vector(r.R)
        if all(np.linalg.norm(vec - s) >= 1.0 for s in seen):
            seen.append(vec)
            unique.append(r)

    diag_vals = np.array([np.real(np.diag(r.R)) for r in unique])
    lower = np.min(diag_vals, axis=0)
    upper = np.max(diag_vals, axis=0)
    lower = np.minimum(lower, 0.0)
    upper = np.maximum(upper, 0.0)

    # Widen by ``margin`` away from zero; also keep a small negative/positive
    # window on the opposite side so branches crossing zero are not lost.
    max_abs = max(np.max(np.abs(lower)), np.max(np.abs(upper)), 1.0)
    lower = np.where(lower < 0, lower * margin, -0.1 * max_abs)
    upper = np.where(upper > 0, upper * margin, 0.1 * max_abs)

    offdiag = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vals = [abs(r.R[i, j]) for r in unique]
            offdiag[i, j] = offdiag[j, i] = margin * max(vals + [1e-8])

    return GuessBounds(
        diag_lower=lower,
        diag_upper=upper,
        offdiag_scale=offdiag,
        diag_candidates=diag_vals,
    )


def infer_guess_bounds(
    model,
    params: dict,
    fallback_scale: float = 100.0,
    margin: float = 2.0,
    r_max: float = 1e6,
    explore: bool = True,
    explore_samples: int = 300,
) -> GuessBounds:
    """
    Infer model- and parameter-aware initial-guess bounds.

    By default this first tries a short exploratory solve with heavy-tailed
    log-uniform guesses.  If that fails to find any solutions, it falls back to
    the cheaper diagonal-balance root estimate.
    """
    if explore:
        bounds = explore_solution_bounds(
            model,
            params,
            n_samples=explore_samples,
            r_min=1e-4,
            r_max=r_max,
            seed=0,
            residual_tol=1e-4,
            margin=margin,
        )
        if bounds is not None:
            return bounds

    return _guess_bounds_from_roots(
        model, params, fallback_scale=fallback_scale, margin=margin, r_max=r_max
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
