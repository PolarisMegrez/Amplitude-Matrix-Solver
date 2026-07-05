"""
Search for multiple steady-state solutions of a nonlinear R-matrix model.

Strategy:
    1. Generate a grid or random sample of initial guesses;
    2. Solve from each guess;
    3. Cluster/deduplicate solutions by distance;
    4. Optionally refine each unique solution.
"""

from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from numerics.models.base import Model, SteadyStateResult

from numerics.solvers.steady_state import solve_steady_state, _root_residual_func, _make_root_jacobian
from numerics.utils.validation import cluster_solutions


def _make_guess_grid(
    dim: int,
    n_samples: int,
    scale: float = 1.0,
    seed: int | None = None,
    include_zero: bool = True,
    include_eye: bool = True,
) -> list[np.ndarray]:
    """
    Generate a list of Hermitian initial guesses.

    For a d x d Hermitian matrix there are d^2 real degrees of freedom.
    We sample diagonal and off-diagonal entries separately.
    """
    rng = np.random.default_rng(seed)
    guesses = []

    if include_zero:
        guesses.append(np.zeros((dim, dim), dtype=complex))
    if include_eye:
        guesses.append(np.eye(dim, dtype=complex) * scale)

    n_random = max(0, n_samples - len(guesses))
    for _ in range(n_random):
        # Diagonal entries ~ Exp(0.5) * scale
        diag = rng.exponential(scale=scale, size=dim)
        R = np.diag(diag).astype(complex)
        # Upper-triangle complex entries
        for i in range(dim):
            for j in range(i + 1, dim):
                re = rng.normal(0.0, scale)
                im = rng.normal(0.0, scale)
                R[i, j] = re + 1j * im
                R[j, i] = re - 1j * im
        guesses.append(R)
    return guesses


def find_steady_states(
    model: "Model",
    params: dict,
    guesses: Sequence[np.ndarray] | None = None,
    n_samples: int = 10,
    scale: float = 1.0,
    seed: int | None = 42,
    solver_method: str = "auto",
    distance_tol: float = 1e-6,
    residual_tol: float = 1e-6,
    early_stop_unique: int | None = None,
    patience: int = 10,
    parallel: bool = False,
    n_jobs: int = 4,
    **solver_kwargs,
) -> list["SteadyStateResult"]:
    """
    Search for multiple steady-state solutions from a set of initial guesses.

    Parameters
    ----------
    model : Model
        The model to solve.
    params : dict
        Model parameters.
    guesses : sequence of np.ndarray, optional
        User-provided initial guesses. If None, a random grid is generated.
    n_samples : int
        Number of random guesses if `guesses` is None.
    scale : float
        Scale for random guess generation.
    seed : int or None
        Random seed.
    solver_method : str
        Method passed to solve_steady_state.
    distance_tol : float
        Solutions closer than this are merged.
    residual_tol : float
        Only accept solutions with residual below this.
    early_stop_unique : int, optional
        If set, stop once this many unique solutions have been found and
        `patience` consecutive guesses have failed to produce a new one.
    patience : int
        Number of consecutive non-new successful guesses required before
        early-stopping.
    parallel : bool
        If True, solve guesses concurrently within a single parameter point
        using a thread pool.  This is useful when each Newton solve is
        CPU-bound and releases the GIL (e.g. scipy.optimize.root).
    n_jobs : int
        Number of threads when ``parallel=True``.

    Returns
    -------
    results : list[SteadyStateResult]
        Deduplicated successful steady states, sorted by residual.
    """
    if guesses is None:
        guesses = _make_guess_grid(model.dim, n_samples, scale=scale, seed=seed)

    use_jacobian = solver_kwargs.get("use_jacobian", True)
    residual_func = _root_residual_func(model, params)
    jac_func = _make_root_jacobian(model, params) if use_jacobian else None

    # Keep our own running deduplication so we can early-stop cheaply.
    unique_vecs: list[np.ndarray] = []
    unique_results: list["SteadyStateResult"] = []
    streak = 0

    def process_result(res: "SteadyStateResult") -> None:
        nonlocal streak
        if not (res.success and res.residual <= residual_tol):
            return
        from numerics.core.r_matrix import R_matrix_to_vector
        vec = R_matrix_to_vector(res.R)
        is_new = all(
            np.linalg.norm(vec - u) >= distance_tol for u in unique_vecs
        )
        if is_new:
            unique_vecs.append(vec)
            unique_results.append(res)
            streak = 0
        else:
            streak += 1

    def should_stop() -> bool:
        return (
            early_stop_unique is not None
            and len(unique_results) >= early_stop_unique
            and streak >= patience
        )

    if not parallel:
        for idx, guess in enumerate(guesses):
            try:
                res = solve_steady_state(
                    model, params, guess=guess, method=solver_method,
                    residual_func=residual_func, jac_func=jac_func,
                    **solver_kwargs,
                )
            except Exception:
                continue
            process_result(res)
            if should_stop():
                break
    else:
        chunk_size = max(1, n_jobs * 2)
        idx = 0
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            while idx < len(guesses) and not should_stop():
                chunk = guesses[idx : idx + chunk_size]
                idx += chunk_size
                futures = {
                    executor.submit(
                        solve_steady_state,
                        model, params, guess, solver_method,
                        residual_func, jac_func,
                        **solver_kwargs,
                    ): guess
                    for guess in chunk
                }
                for fut in as_completed(futures):
                    try:
                        process_result(fut.result())
                    except Exception:
                        pass
                    if should_stop():
                        for f in futures:
                            f.cancel()
                        break

    # Final clustering step keeps compatibility with the old pipeline and
    # catches any near-duplicates that slipped through the running filter.
    from numerics.core.r_matrix import R_matrix_to_vector
    sol_vecs = [R_matrix_to_vector(res.R) for res in unique_results]
    residuals = [res.residual for res in unique_results]
    clustered = cluster_solutions(sol_vecs, residuals, distance_tol=distance_tol)

    representative_set = {tuple(np.round(v, 12)) for v, _ in clustered}
    final_results = []
    seen = set()
    for res in unique_results:
        v = R_matrix_to_vector(res.R)
        key = tuple(np.round(v, 12))
        if key in representative_set and key not in seen:
            seen.add(key)
            final_results.append(res)

    final_results.sort(key=lambda r: r.residual)
    return final_results
