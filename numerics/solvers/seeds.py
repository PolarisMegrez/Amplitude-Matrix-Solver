"""Initial-guess generators and dense local search for steady-state solvers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from numerics.solvers.guess_bounds import (
    GuessBounds,
    infer_guess_bounds,
    merge_guess_bounds,
)
from numerics.solvers.steady_state import solve_steady_state
from numerics.solvers.multi_search import find_steady_states
from numerics.utils.fixed_points import (
    FixedPoint,
    deduplicate_fixed_points,
    result_to_fixedpoint,
)


def make_random_guesses(
    dim: int,
    n_samples: int,
    scale: float = 100.0,
    seed: int | None = None,
    bounds: GuessBounds | None = None,
) -> list[np.ndarray]:
    """Generate random Hermitian initial guesses for multi-start search.

    If ``bounds`` is supplied, it overrides ``scale`` and the guesses are
    sampled inside the model-inferred ranges (including negative diagonal
    entries when the model admits them).
    """
    if bounds is not None:
        return bounds.sample(n_samples, seed=seed)

    rng = np.random.default_rng(seed)
    guesses: list[np.ndarray] = []
    for _ in range(n_samples):
        diag = rng.exponential(scale=scale, size=dim)
        R = np.diag(diag).astype(complex)
        for i in range(dim):
            for j in range(i + 1, dim):
                re = rng.normal(0.0, scale)
                im = rng.normal(0.0, scale)
                R[i, j] = re + 1j * im
                R[j, i] = re - 1j * im
        guesses.append(R)
    return guesses


def discover_seed_guesses(
    model,
    params: dict,
    R11_vals: Sequence[float] | None = None,
    R22_vals: Sequence[float] | None = None,
    amplitude_vals: Sequence[float] | None = None,
    n_phases: int = 8,
    bounds: GuessBounds | None = None,
) -> list[np.ndarray]:
    """Use a coarse grid with the root solver to discover seed fixed points.

    If ``bounds`` is provided, the diagonal grid is built from the
    model-inferred diagonal candidates instead of the hard-coded lists, which
    makes the search naturally cover negative as well as positive diagonal
    entries.
    """
    n = model.dim
    phase_vals = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)

    if bounds is None:
        R11_vals = R11_vals or [0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]
        R22_vals = R22_vals or [0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0]
        amplitude_vals = amplitude_vals or [0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0]
        diag_combos = np.array(np.meshgrid(R11_vals, R22_vals)).reshape(2, -1).T
    else:
        candidates = bounds.diag_candidates
        if candidates is None or len(candidates) == 0:
            per_dim = [
                np.linspace(bounds.diag_lower[i], bounds.diag_upper[i], 5)
                for i in range(n)
            ]
            candidates = np.array(np.meshgrid(*per_dim, indexing="ij")).reshape(n, -1).T
        diag_combos = candidates
        amp_scale = float(np.mean(bounds.offdiag_scale[bounds.offdiag_scale > 0]) or 1.0)
        amplitude_vals = amplitude_vals or [0.0, 0.5 * amp_scale, amp_scale]

    amplitude_vals = list(amplitude_vals)

    candidates: list[FixedPoint] = []
    for diag in diag_combos:
        for amp in amplitude_vals:
            for phase in phase_vals:
                R = np.diag(np.asarray(diag, dtype=float)).astype(complex)
                for i in range(n):
                    for j in range(i + 1, n):
                        val = amp * np.exp(1j * phase)
                        R[i, j] = val
                        R[j, i] = val.conj()
                if np.min(np.linalg.eigvalsh(R)) < -1.0:
                    continue
                try:
                    res = solve_steady_state(
                        model, params, guess=R.copy(), method="root",
                        tol=1e-10, use_jacobian=False
                    )
                except Exception:
                    continue
                if res.success and res.residual <= 1e-4:
                    candidates.append(result_to_fixedpoint(res))

    # Simple deduplication before returning raw R matrices
    from numerics.utils.fixed_points import deduplicate_fixed_points
    unique = deduplicate_fixed_points(candidates, distance_tol=5.0)
    return [fp.R for fp in unique]


def discover_seed_guesses_multi_point(
    model,
    param_points: Sequence[dict],
    bounds: GuessBounds | None = None,
) -> list[np.ndarray]:
    """
    Discover seed fixed points at several parameter locations and merge them.

    This is important when a branch does not exist at the nominal center point
    but appears elsewhere in the scan window.

    If ``bounds`` is ``"auto"`` or None, bounds are inferred from the first
    parameter point (or the supplied ``"auto"`` string triggers inference).
    """
    from numerics.utils.fixed_points import deduplicate_fixed_points

    if bounds is None or bounds == "auto":
        bounds = infer_guess_bounds(model, param_points[0])

    all_candidates: list[FixedPoint] = []
    for p in param_points:
        for R in discover_seed_guesses(model, p, bounds=bounds, n_phases=8):
            all_candidates.append(FixedPoint(
                R=R, residual=0.0, omega=0.0,
                J_eigvals=np.linalg.eigvals(
                    model.build_jacobian_builder(p)(
                        np.array([
                            R[0, 0].real, R[1, 1].real,
                            R[0, 1].real, R[0, 1].imag
                        ]),
                        p,
                    )
                ),
            ))
    unique = deduplicate_fixed_points(all_candidates, distance_tol=5.0)
    return [fp.R for fp in unique]


def independent_search_point(
    model,
    params: dict,
    n_samples: int = 200,
    scale: float = 100.0,
    seed: int = 42,
    distance_tol: float = 3.0,
    residual_tol: float = 1e-4,
    bounds: GuessBounds | None = None,
) -> list[FixedPoint]:
    """
    Run an independent multi-start search at a single parameter point.

    Uses the backend-aware batched Newton solver so the same code path runs
    on CPU and GPU.
    """
    from numerics.solvers.batched import solve_steady_state_batched

    guesses = make_random_guesses(
        model.dim, n_samples, scale=scale, seed=seed, bounds=bounds
    )
    guesses_arr = np.array(guesses).reshape(1, n_samples, model.dim, model.dim)
    try:
        results = solve_steady_state_batched(
            model,
            params,
            guesses_arr,
            max_iter=50,
            tol=1e-10,
            line_search=True,
            compute_eigvals=True,
        )
    except Exception:
        # Fall back to the scipy-based multi-start search for models that do
        # not expose a symbolic Jacobian builder.
        try:
            results = find_steady_states(
                model, params,
                n_samples=n_samples,
                scale=scale,
                seed=seed,
                solver_method="root",
                distance_tol=distance_tol,
                residual_tol=residual_tol,
                tol=1e-10,
                use_jacobian=False,
            )
        except Exception:
            return []
        return [result_to_fixedpoint(r) for r in results]

    candidates: list[FixedPoint] = []
    for res in results[0]:
        if res.success and res.residual <= residual_tol:
            candidates.append(result_to_fixedpoint(res))
    return deduplicate_fixed_points(candidates, distance_tol=distance_tol)


def local_dense_search(
    model,
    params: dict,
    center_guess: np.ndarray,
    n_perturbations: int = 20,
    scale: float = 5.0,
    seed: int = 0,
) -> list[FixedPoint]:
    """Perturb a central guess randomly to recover solutions near a difficult point."""
    from numerics.utils.fixed_points import deduplicate_fixed_points, solve_from_seed

    rng = np.random.default_rng(seed)
    candidates: list[FixedPoint] = []
    fp = solve_from_seed(model, params, center_guess)
    if fp is not None:
        candidates.append(fp)

    for _ in range(n_perturbations):
        R = center_guess.copy()
        R[0, 0] = max(0.0, R[0, 0].real + rng.normal(0, scale))
        R[1, 1] = max(0.0, R[1, 1].real + rng.normal(0, scale))
        amp = abs(R[0, 1]) + rng.exponential(scale)
        phase = rng.uniform(0, 2 * np.pi)
        R[0, 1] = amp * np.exp(1j * phase)
        R[1, 0] = R[0, 1].conj()
        fp = solve_from_seed(model, params, R)
        if fp is not None:
            candidates.append(fp)

    return deduplicate_fixed_points(candidates, distance_tol=3.0)
