"""
Validation utilities: positive-semidefinite checks, convergence helpers,
solution clustering/deduplication.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from numerics.core.backend import get_array_module, to_numpy


def is_positive_semidefinite(R, atol: float = 1e-10) -> bool:
    """Check whether a Hermitian matrix is positive semidefinite."""
    xp = get_array_module(R)
    R = xp.asarray(R)
    if R.size == 1:
        return float(xp.real(R[0, 0])) >= -atol
    try:
        eigvals = xp.linalg.eigvalsh(R)
    except Exception:
        return False
    return bool(xp.all(eigvals >= -atol))


def convergence_message(success: bool, residual: float, tol: float) -> str:
    """Return a human-readable convergence diagnostic."""
    if success and residual <= tol:
        return f"Converged with residual {residual:.3e} <= tol {tol:.3e}."
    if success and residual > tol:
        return (
            f"Solver reported success but residual {residual:.3e} > tol {tol:.3e}; "
            "result may be unreliable."
        )
    return f"Solver failed with residual {residual:.3e} > tol {tol:.3e}."


def cluster_solutions(
    solutions: Sequence,
    residuals: Sequence[float] | None = None,
    distance_tol: float = 1e-6,
):
    """
    Deduplicate a list of solution vectors by Euclidean distance.

    Parameters
    ----------
    solutions : sequence of np.ndarray
        Candidate solution vectors.
    residuals : sequence of float, optional
        Corresponding residuals; used to pick the best representative.
    distance_tol : float
        Two solutions are considered identical if their distance is below this.

    Returns
    -------
    clustered : list[(solution, residual)]
        List of unique solutions (best residual representative).
    """
    if residuals is None:
        residuals = [0.0] * len(solutions)

    # Clustering is easiest on CPU with NumPy
    sols = [to_numpy(s).astype(float) for s in solutions]
    ress = [float(r) for r in residuals]

    clusters: list[list[tuple[np.ndarray, float]]] = []
    for sol, res in zip(sols, ress):
        assigned = False
        for cluster in clusters:
            ref = cluster[0][0]
            if float(np.linalg.norm(sol - ref)) < distance_tol:
                cluster.append((sol, res))
                assigned = True
                break
        if not assigned:
            clusters.append([(sol, res)])

    result = []
    for cluster in clusters:
        best = min(cluster, key=lambda x: x[1])
        result.append(best)
    return result
