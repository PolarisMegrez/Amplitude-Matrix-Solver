"""Utilities for representing, deduplicating and matching fixed points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from numerics.core.backend import get_array_module


@dataclass
class FixedPoint:
    """A single fixed point of the R-matrix dynamics."""

    R: np.ndarray
    residual: float
    omega: float | None = None
    J_eigvals: np.ndarray | None = None

    @property
    def is_psd(self) -> bool:
        xp = get_array_module(self.R)
        return bool(xp.all(xp.linalg.eigvalsh(self.R) > -1e-8))

    @property
    def is_stable(self) -> bool:
        if self.J_eigvals is None:
            return False
        xp = get_array_module(self.J_eigvals)
        return bool(xp.all(xp.real(self.J_eigvals) < 1e-10))

    @property
    def trace(self) -> float:
        xp = get_array_module(self.R)
        return float(xp.trace(self.R).real)

    def to_vector(self) -> np.ndarray:
        """Flatten R to a real vector for distance comparisons."""
        from numerics.core.r_matrix import R_matrix_to_vector
        return R_matrix_to_vector(self.R)


def result_to_fixedpoint(res) -> FixedPoint:
    """Convert a SteadyStateResult into our local FixedPoint representation."""
    return FixedPoint(
        R=res.R,
        residual=res.residual,
        omega=res.omega,
        J_eigvals=res.J_eigvals,
    )


def deduplicate_fixed_points(
    candidates: Sequence[FixedPoint],
    distance_tol: float = 3.0,
) -> list[FixedPoint]:
    """Merge fixed points that are numerically identical."""
    unique: list[FixedPoint] = []
    for fp in candidates:
        vec = fp.to_vector()
        if all(np.linalg.norm(vec - u.to_vector()) >= distance_tol for u in unique):
            unique.append(fp)
    return unique


def get_neighbor_solutions(
    grid: list[list[list[FixedPoint] | None]],
    i: int,
    j: int,
) -> list[FixedPoint]:
    """Collect converged solutions from direct neighbors (up/down/left/right)."""
    neighbors: list[FixedPoint] = []
    ni, nj = len(grid), len(grid[0]) if grid else 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ii, jj = i + di, j + dj
        if 0 <= ii < ni and 0 <= jj < nj and grid[ii][jj] is not None:
            neighbors.extend(grid[ii][jj])
    return neighbors


def match_to_neighbors(
    current: Sequence[FixedPoint],
    neighbors: Sequence[FixedPoint],
    distance_tol: float = 10.0,
) -> list[int]:
    """
    Return branch indices for current points based on nearest-neighbor matching.
    Unmatched points get new indices.
    """
    if not neighbors:
        return list(range(len(current)))

    neighbor_vecs = np.array([fp.to_vector() for fp in neighbors])
    indices = []
    used: set[int] = set()
    for fp in current:
        vec = fp.to_vector()
        dists = np.linalg.norm(neighbor_vecs - vec, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] < distance_tol and idx not in used:
            indices.append(idx)
            used.add(idx)
        else:
            new_idx = max(list(used) + list(range(len(current)))) + 1
            indices.append(new_idx)
            used.add(new_idx)
    return indices


def solve_from_seed(
    model,
    params: dict,
    seed_R: np.ndarray,
    residual_tol: float = 1e-4,
    **solver_kwargs,
) -> FixedPoint | None:
    """Try to find a fixed point starting from a single seed R."""
    from numerics.solvers.steady_state import solve_steady_state

    try:
        res = solve_steady_state(
            model, params, guess=seed_R.copy(), method="root",
            tol=1e-10, use_jacobian=False, **solver_kwargs
        )
    except Exception:
        return None
    if not res.success or res.residual > residual_tol:
        return None
    if res.J_eigvals is None:
        return None
    return FixedPoint(
        R=res.R, residual=res.residual,
        omega=res.omega, J_eigvals=res.J_eigvals
    )
