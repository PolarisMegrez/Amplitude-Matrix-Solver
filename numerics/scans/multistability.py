"""
Generic multi-stability parameter scans for R-matrix models.

This module provides a model-agnostic 2-D (and eventually N-D) scanner that
finds all coexisting steady-state fixed points on a parameter grid and
returns a structured result container.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from numerics.core.backend import get_array_module, get_backend, to_numpy
from numerics.core.r_matrix import R_matrix_to_vector
from numerics.models.base import Model
from numerics.scans.parallel import TileScanRunner, make_tiles
from numerics.scans.traversal import spiral_order
from numerics.solvers.batched import solve_steady_state_batched
from numerics.solvers.multi_search import find_steady_states
from numerics.solvers.guess_bounds import (
    GuessBounds,
    infer_guess_bounds,
    merge_guess_bounds,
)
from numerics.solvers.seeds import (
    discover_seed_guesses_multi_point,
    independent_search_point,
    make_random_guesses,
)
from numerics.utils.fixed_points import (
    FixedPoint,
    deduplicate_fixed_points,
    get_neighbor_solutions,
    match_to_neighbors,
    result_to_fixedpoint,
)
from numerics.utils.progress import ProgressTracker


@dataclass(frozen=True)
class ScanTolerances:
    """Numerical tolerances used throughout a multistability scan."""

    residual_tol: float = 1e-4
    solver_tol: float = 1e-10
    distance_tol: float = 3.0
    branch_match_tol: float = 10.0


@dataclass
class MultistabilityScanResult:
    """
    Container for a completed multistability parameter scan.

    All array fields have shape ``(*grid_shape, max_branches, ...)``.
    """

    axes: dict[str, np.ndarray]
    base_params: dict
    n_solutions: np.ndarray
    R_matrices: np.ndarray
    residuals: np.ndarray
    omegas: np.ndarray
    J_eigvals: np.ndarray
    is_psd: np.ndarray
    is_stable: np.ndarray
    branch_ids: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(self.n_solutions.shape)

    @property
    def max_branches(self) -> int:
        return int(self.R_matrices.shape[-3])

    def save_npz(self, path: str | Path) -> None:
        """Save the scan result to an NPZ archive."""
        path = Path(path)
        save_dict = {
            "base_params": json.dumps(self.base_params),
            "grid_shape": np.array(self.grid_shape),
            "max_branches": np.array(self.R_matrices.shape[-3]),
            "n_solutions": self.n_solutions,
            "R_matrices": self.R_matrices,
            "residuals": self.residuals,
            "omegas": self.omegas,
            "J_eigvals": self.J_eigvals,
            "is_psd": self.is_psd,
            "is_stable": self.is_stable,
            "branch_ids": self.branch_ids,
            "metadata": json.dumps(self.metadata),
        }
        for name, arr in self.axes.items():
            save_dict[f"axis_{name}"] = arr
        np.savez(path, **save_dict)

    @classmethod
    def load_npz(cls, path: str | Path) -> "MultistabilityScanResult":
        """Load a scan result from an NPZ archive."""
        data = np.load(path, allow_pickle=True)
        axes = {}
        base_params = json.loads(str(data["base_params"]))
        for key in data.files:
            if key.startswith("axis_"):
                axes[key[5:]] = data[key]
        return cls(
            axes=axes,
            base_params=base_params,
            n_solutions=data["n_solutions"],
            R_matrices=data["R_matrices"],
            residuals=data["residuals"],
            omegas=data["omegas"],
            J_eigvals=data["J_eigvals"],
            is_psd=data["is_psd"],
            is_stable=data["is_stable"],
            branch_ids=data["branch_ids"],
            metadata=json.loads(str(data["metadata"])),
        )

    def to_fixedpoints(self, index: tuple[int, ...]) -> list[FixedPoint]:
        """Return the fixed points at a given grid index."""
        fps: list[FixedPoint] = []
        for k in range(int(self.n_solutions[index])):
            R = self.R_matrices[(*index, k)]
            if np.isnan(R[0, 0]):
                break
            fps.append(FixedPoint(
                R=R,
                residual=float(self.residuals[(*index, k)]),
                omega=float(self.omegas[(*index, k)]),
                J_eigvals=self.J_eigvals[(*index, k)],
            ))
        return fps


@dataclass
class ParallelConfig:
    """Configuration for tile-based process parallelism."""

    n_workers: int = 8
    n_tiles: int | None = None


class MultistabilityScan2D:
    """
    Model-agnostic 2-D multistability scan.

    Parameters
    ----------
    model : Model
        R-matrix model to scan.
    base_params : dict
        Base parameter dictionary.
    axes : dict[str, np.ndarray]
        Mapping from two parameter names to 1-D axis arrays.  The scan grid is
        the Cartesian product of these axes.
    n_random_guesses : int
        Number of random guesses added to the warm-start pool at each point.
    tolerances : ScanTolerances
        Numerical tolerances.
    max_branches : int
        Maximum number of coexisting branches to store per grid point.
    backend : str
        ``"auto"`` selects cupy if available, otherwise numpy.  ``"numpy"`` or
        ``"cupy"`` forces the backend.
    parallel : ParallelConfig | None
        If given, use tile-based process parallelism.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        model: Model,
        base_params: dict,
        axes: dict[str, np.ndarray],
        n_random_guesses: int = 50,
        tolerances: ScanTolerances | None = None,
        max_branches: int = 8,
        backend: str = "auto",
        parallel: ParallelConfig | None = None,
        symmetry_axis: str | None = None,
        guess_bounds: GuessBounds | Literal["auto"] | None = None,
        verbose: bool = True,
    ) -> None:
        if len(axes) != 2:
            raise ValueError("MultistabilityScan2D currently requires exactly two axes")
        if symmetry_axis is not None and symmetry_axis not in axes:
            raise ValueError(
                f"symmetry_axis '{symmetry_axis}' must be one of {list(axes.keys())}"
            )
        self.model = model
        self.base_params = dict(base_params)
        self.axes = {k: np.asarray(v) for k, v in axes.items()}
        self.n_random_guesses = n_random_guesses
        self.tolerances = tolerances or ScanTolerances()
        self.max_branches = max_branches
        self.backend = self._resolve_backend(backend)
        self.parallel = parallel
        self.symmetry_axis = symmetry_axis
        self._guess_bounds_input = guess_bounds
        self.guess_bounds = self._resolve_guess_bounds(guess_bounds)
        self.verbose = verbose

        self.axis_names = list(self.axes.keys())
        self.axis_lengths = tuple(len(v) for v in self.axes.values())
        self.n_rows, self.n_cols = self.axis_lengths

    def _resolve_guess_bounds(
        self,
        guess_bounds: GuessBounds | Literal["auto"] | None,
    ) -> GuessBounds | None:
        """Resolve the user-supplied guess-bound policy.

        ``"auto"`` is intentionally resolved later, over the seed parameter
        points, so the bounds reflect the whole scan window rather than only the
        nominal center.
        """
        if guess_bounds is None or guess_bounds == "auto":
            return None
        if isinstance(guess_bounds, GuessBounds):
            return guess_bounds
        raise ValueError(f"Unknown guess_bounds policy: {guess_bounds!r}")

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            try:
                import cupy  # noqa: F401
                return "cupy"
            except ImportError:
                return "numpy"
        if backend not in {"numpy", "cupy"}:
            raise ValueError(f"Unknown backend: {backend!r}")
        if backend == "cupy":
            try:
                import cupy  # noqa: F401
            except ImportError as exc:
                raise ImportError("CuPy requested but not installed") from exc
        return backend

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> MultistabilityScanResult:
        """Run the full scan and return a populated result container."""
        from numerics.core.backend import set_backend

        original_backend = get_backend()
        set_backend(self.backend)
        try:
            seed_Rs = self._discover_seeds()
            if self.parallel is not None:
                arrays = self._run_parallel(seed_Rs)
            else:
                arrays = self._run_sequential(seed_Rs)
            n_solutions, R_matrices, residuals, branch_ids = arrays
            result = self._build_result(
                n_solutions, R_matrices, residuals, branch_ids
            )
            result = self._refine(result)
            if self.symmetry_axis is not None:
                result = self._enforce_reflection_symmetry(result)
            result.metadata.update({
                "backend": self.backend,
                "n_random_guesses": self.n_random_guesses,
                "tolerances": {
                    k: getattr(self.tolerances, k)
                    for k in self.tolerances.__dataclass_fields__
                },
                "guess_bounds": (
                    {
                        "diag_lower": self.guess_bounds.diag_lower.tolist(),
                        "diag_upper": self.guess_bounds.diag_upper.tolist(),
                    }
                    if self.guess_bounds is not None
                    else None
                ),
            })
            return result
        finally:
            set_backend(original_backend)

    # -------------------------------------------------------------------------
    # Seed discovery
    # -------------------------------------------------------------------------

    def _discover_seeds(self) -> list[np.ndarray]:
        """Discover global seed fixed points at representative parameter points."""
        names = self.axis_names
        ax0, ax1 = self.axes[names[0]], self.axes[names[1]]
        seed_points = [
            {**self.base_params, names[0]: ax0[0], names[1]: ax1[0]},
            {**self.base_params, names[0]: ax0[-1], names[1]: ax1[0]},
            {**self.base_params, names[0]: ax0[0], names[1]: ax1[-1]},
            {**self.base_params, names[0]: ax0[-1], names[1]: ax1[-1]},
            self.base_params,
        ]
        t0 = time.perf_counter()
        if self.verbose:
            print("=== Discovering seed fixed points ===")

        # If the user asked for automatic bounds, infer them from the seed
        # parameter points and merge, so the bounds cover the whole scan window
        # instead of just the nominal center.
        if self._guess_bounds_input == "auto" and self.guess_bounds is None:
            if self.verbose:
                print("  (inferring adaptive guess bounds from seed points)")
            bounds_list = [
                infer_guess_bounds(self.model, p, explore_samples=200)
                for p in seed_points
            ]
            self.guess_bounds = merge_guess_bounds(bounds_list)
            if self.verbose:
                print(
                    f"  inferred diag bounds: "
                    f"lower={self.guess_bounds.diag_lower.tolist()}, "
                    f"upper={self.guess_bounds.diag_upper.tolist()}"
                )

        seed_Rs = discover_seed_guesses_multi_point(
            self.model, seed_points, bounds=self.guess_bounds
        )
        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"Found {len(seed_Rs)} unique seed fixed points in {elapsed:.1f}s\n")
        return seed_Rs

    # -------------------------------------------------------------------------
    # Sequential scan
    # -------------------------------------------------------------------------

    def _run_sequential(
        self,
        seed_Rs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the 2-D scan sequentially with neighbor continuation."""
        n_rows, n_cols = self.n_rows, self.n_cols
        n_solutions, R_matrices, residuals, branch_ids = self._allocate_arrays(
            (n_rows, n_cols)
        )
        grid: list[list[list[FixedPoint] | None]] = [
            [None for _ in range(n_cols)] for _ in range(n_rows)
        ]

        center_i, center_j = n_rows // 2, n_cols // 2
        order = spiral_order(n_rows, n_cols, center_i, center_j)

        progress = ProgressTracker(
            len(order), label="2D scan", interval=2.0, enabled=self.verbose
        )
        failed_points: list[tuple[int, int]] = []

        for idx, (i, j) in enumerate(order):
            params = self._params_at(i, j)
            neighbor_fps = get_neighbor_solutions(grid, i, j)
            fps = self._solve_point(params, neighbor_fps, seed_Rs, idx)

            if not fps:
                fps = independent_search_point(
                    self.model, params,
                    n_samples=200, scale=200.0, seed=idx,
                    distance_tol=self.tolerances.distance_tol,
                )

            if not fps:
                failed_points.append((i, j))
                progress.update(failed=1)
                continue

            grid[i][j] = fps
            self._store_point(
                n_solutions, R_matrices, residuals, branch_ids,
                grid, i, j, fps, neighbor_fps,
            )
            progress.update(failed=0)

        progress.finish()

        if failed_points:
            self._retry_failed_points(
                n_solutions, R_matrices, residuals, branch_ids,
                grid, failed_points,
            )

        return n_solutions, R_matrices, residuals, branch_ids

    # -------------------------------------------------------------------------
    # Parallel tile scan
    # -------------------------------------------------------------------------

    def _run_parallel(
        self,
        seed_Rs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the 2-D scan in parallel over independent tiles."""
        n_rows, n_cols = self.n_rows, self.n_cols
        n_tiles = self.parallel.n_tiles or max(self.parallel.n_workers * 4, 16)
        tiles = make_tiles(n_rows, n_cols, n_tiles)

        n_solutions, R_matrices, residuals, branch_ids = self._allocate_arrays(
            (n_rows, n_cols)
        )

        def merge(tile, res):
            row_start, row_end, col_start, col_end, _, _ = tile
            ns, Rm, resids, bids = res
            n_solutions[row_start:row_end, col_start:col_end] = ns
            R_matrices[row_start:row_end, col_start:col_end] = Rm
            residuals[row_start:row_end, col_start:col_end] = resids
            branch_ids[row_start:row_end, col_start:col_end] = bids

        runner = TileScanRunner(
            _tile_worker,
            n_workers=self.parallel.n_workers,
            initializer=_init_tile_worker,
            initargs=(
                self.model,
                self.base_params,
                self.axes,
                seed_Rs,
                self.n_random_guesses,
                self.tolerances,
                self.max_branches,
                self.guess_bounds,
                self.verbose,
            ),
            verbose=self.verbose,
        )
        tile_args = [
            (row_start, row_end, col_start, col_end, local_i, local_j)
            for row_start, row_end, col_start, col_end, local_i, local_j in tiles
        ]
        t0 = time.perf_counter()
        runner.run(tiles, tile_args, merge)
        if self.verbose:
            elapsed = time.perf_counter() - t0
            print(f"  Parallel tile scan phase took {elapsed:.1f}s\n")
        return n_solutions, R_matrices, residuals, branch_ids

    # -------------------------------------------------------------------------
    # Batched tile solve logic
    # -------------------------------------------------------------------------

    def _params_at(self, i: int, j: int) -> dict:
        """Build the parameter dict for grid index (i, j)."""
        names = self.axis_names
        params = dict(self.base_params)
        params[names[0]] = float(self.axes[names[0]][i])
        params[names[1]] = float(self.axes[names[1]][j])
        return params

    def _solve_point(
        self,
        params: dict,
        neighbor_fps: Sequence[FixedPoint],
        seed_Rs: Sequence[np.ndarray],
        seed: int,
    ) -> list[FixedPoint]:
        """Find fixed points at a single parameter point."""
        seed_guesses = [fp.R for fp in neighbor_fps]
        seed_guesses += [
            R for R in seed_Rs
            if not any(
                np.linalg.norm(R.flatten() - s.flatten()) < 1e-6
                for s in seed_guesses
            )
        ]
        if not seed_guesses:
            return []

        scale = 200.0 if self.guess_bounds is None else 100.0
        combined = list(seed_guesses) + make_random_guesses(
            self.model.dim,
            self.n_random_guesses,
            scale=scale,
            seed=seed,
            bounds=self.guess_bounds,
        )
        results = find_steady_states(
            self.model, params,
            guesses=combined,
            n_samples=0,
            scale=200.0,
            seed=seed,
            solver_method="root",
            distance_tol=self.tolerances.distance_tol,
            residual_tol=self.tolerances.residual_tol,
            tol=self.tolerances.solver_tol,
            use_jacobian=False,
            early_stop_unique=self.max_branches,
            patience=10,
        )
        return [result_to_fixedpoint(r) for r in results]

    # -------------------------------------------------------------------------
    # Allocation and storage
    # -------------------------------------------------------------------------

    def _allocate_arrays(
        self, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_rows, n_cols = shape
        n = self.model.dim
        n_solutions = np.zeros((n_rows, n_cols), dtype=int)
        R_matrices = np.full(
            (n_rows, n_cols, self.max_branches, n, n),
            np.nan + 0j * np.nan,
            dtype=complex,
        )
        residuals = np.full((n_rows, n_cols, self.max_branches), np.nan)
        branch_ids = np.full((n_rows, n_cols, self.max_branches), -1, dtype=int)
        return n_solutions, R_matrices, residuals, branch_ids

    def _store_point(
        self,
        n_solutions: np.ndarray,
        R_matrices: np.ndarray,
        residuals: np.ndarray,
        branch_ids: np.ndarray,
        grid: list[list[list[FixedPoint] | None]],
        i: int,
        j: int,
        fps: list[FixedPoint],
        neighbor_fps: Sequence[FixedPoint],
    ) -> None:
        """Store a point's fixed points into the packed arrays."""
        n_solutions[i, j] = len(fps)
        bidx = match_to_neighbors(fps, neighbor_fps, self.tolerances.branch_match_tol)
        for k, (fp, bid) in enumerate(zip(fps, bidx)):
            if k >= self.max_branches:
                break
            R_matrices[i, j, k] = fp.R
            residuals[i, j, k] = fp.residual
            branch_ids[i, j, k] = bid
        grid[i][j] = fps

    # -------------------------------------------------------------------------
    # Failed-point retry
    # -------------------------------------------------------------------------

    def _retry_failed_points(
        self,
        n_solutions: np.ndarray,
        R_matrices: np.ndarray,
        residuals: np.ndarray,
        branch_ids: np.ndarray,
        grid: list[list[list[FixedPoint] | None]],
        failed_points: list[tuple[int, int]],
    ) -> None:
        if self.verbose:
            print(f"\nRetrying {len(failed_points)} failed points with dense search...")
        progress = ProgressTracker(
            len(failed_points), label="Retry", interval=2.0, enabled=self.verbose
        )
        for idx, (i, j) in enumerate(failed_points):
            params = self._params_at(i, j)
            candidates = independent_search_point(
                self.model, params,
                n_samples=200, scale=200.0, seed=idx,
                distance_tol=self.tolerances.distance_tol,
                bounds=self.guess_bounds,
            )
            if candidates:
                neighbor_fps = get_neighbor_solutions(grid, i, j)
                self._store_point(
                    n_solutions, R_matrices, residuals, branch_ids,
                    grid, i, j, candidates, neighbor_fps,
                )
            progress.update(failed=0 if candidates else 1)
        progress.finish()

    # -------------------------------------------------------------------------
    # Post-processing: omega, Jacobian eigenvalues, PSD, stability
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        n_solutions: np.ndarray,
        R_matrices: np.ndarray,
        residuals: np.ndarray,
        branch_ids: np.ndarray,
    ) -> MultistabilityScanResult:
        """Create a result container and run vectorized post-processing."""
        n_rows, n_cols = n_solutions.shape
        n = self.model.dim
        omegas = np.full((n_rows, n_cols, self.max_branches), np.nan)
        J_eigvals = np.full(
            (n_rows, n_cols, self.max_branches, n * n),
            np.nan + 0j * np.nan,
            dtype=complex,
        )
        is_psd = np.full((n_rows, n_cols, self.max_branches), False)
        is_stable = np.full((n_rows, n_cols, self.max_branches), False)

        # Collect converged (R, params) pairs for vectorized post-processing
        valid_mask = ~np.isnan(R_matrices[..., 0, 0])
        if np.any(valid_mask):
            indices = np.argwhere(valid_mask)
            R_stack = np.array([R_matrices[i, j, k] for i, j, k in indices])
            params_list = [self._params_at(i, j) for i, j, _ in indices]
            params_grid = {k: np.array([p[k] for p in params_list]) for k in params_list[0]}

            omegas_stack, J_eigvals_stack, psd_stack, stable_stack = self._postprocess(
                R_stack, params_grid
            )
            for (i, j, k), o, Je, psd, stab in zip(
                indices, omegas_stack, J_eigvals_stack, psd_stack, stable_stack
            ):
                omegas[i, j, k] = o
                J_eigvals[i, j, k] = Je
                is_psd[i, j, k] = psd
                is_stable[i, j, k] = stab

        return MultistabilityScanResult(
            axes=self.axes,
            base_params=self.base_params,
            n_solutions=n_solutions,
            R_matrices=R_matrices,
            residuals=residuals,
            omegas=omegas,
            J_eigvals=J_eigvals,
            is_psd=is_psd,
            is_stable=is_stable,
            branch_ids=branch_ids,
        )

    def _postprocess(
        self,
        R_stack: np.ndarray,
        params_grid: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized computation of omega, Jacobian eigenvalues, PSD, stability."""
        from numerics.postprocess import (
            compute_jacobian_eigvals_numerical,
            compute_omegas,
            compute_psd_and_stability,
        )
        omegas = compute_omegas(self.model, R_stack, params_grid)
        J_eigvals = compute_jacobian_eigvals_numerical(self.model, R_stack, params_grid)
        psd, stable = compute_psd_and_stability(R_stack, J_eigvals)
        return omegas, J_eigvals, psd, stable

    # -------------------------------------------------------------------------
    # Batched multi-point search helper
    # -------------------------------------------------------------------------

    def _batched_search_at_points(
        self,
        params_list: list[dict],
        seed_guess_lists: list[list[np.ndarray]],
        n_random: int,
        scale: float = 100.0,
        seed: int = 0,
    ) -> list[list[FixedPoint]]:
        """
        Run one batched Newton solve for many parameter points at once.

        ``seed_guess_lists[i]`` supplies warm-start R matrices for
        ``params_list[i]``; random guesses are appended to reach a common
        batch width.
        """
        from numerics.core.backend import set_backend

        if not params_list:
            return []

        n = self.model.dim
        rng = np.random.default_rng(seed)

        # Determine the number of guesses per point and pad.
        max_g = 0
        padded_guesses: list[list[np.ndarray]] = []
        for seeds in seed_guess_lists:
            guesses = list(seeds)
            if not guesses:
                guesses = [np.eye(n, dtype=complex)]
            random_Rs = make_random_guesses(
                n,
                n_random,
                scale=scale,
                seed=int(rng.integers(0, 2**31)),
                bounds=self.guess_bounds,
            )
            guesses += random_Rs
            padded_guesses.append(guesses)
            max_g = max(max_g, len(guesses))

        B = len(params_list)
        guesses_arr = np.empty((B, max_g, n, n), dtype=complex)
        for b, guesses in enumerate(padded_guesses):
            for g, R in enumerate(guesses):
                guesses_arr[b, g] = R
            for g in range(len(guesses), max_g):
                guesses_arr[b, g] = guesses[-1]

        params_batch = {
            k: np.array([p[k] for p in params_list])
            for k in self.base_params
        }

        original_backend = get_backend()
        set_backend(self.backend)
        try:
            results = solve_steady_state_batched(
                self.model,
                params_batch,
                guesses_arr,
                max_iter=50,
                tol=self.tolerances.solver_tol,
                line_search=True,
                compute_eigvals=False,
            )
        finally:
            set_backend(original_backend)

        point_roots: list[list[FixedPoint]] = []
        for b, row in enumerate(results):
            candidates: list[FixedPoint] = []
            for res in row:
                if res.success and res.residual <= self.tolerances.residual_tol:
                    candidates.append(result_to_fixedpoint(res))
            unique = deduplicate_fixed_points(
                candidates, distance_tol=self.tolerances.distance_tol
            )
            point_roots.append(unique[: self.max_branches])
        return point_roots

    # -------------------------------------------------------------------------
    # Refinement and symmetry enforcement
    # -------------------------------------------------------------------------

    def _refine(self, result: MultistabilityScanResult) -> MultistabilityScanResult:
        """Refine points whose solution count jumps by more than one vs. neighbors."""
        n_rows, n_cols = result.grid_shape
        suspicious: list[tuple[int, int]] = []
        for i in range(n_rows):
            for j in range(n_cols):
                current = int(result.n_solutions[i, j])
                neighbor_counts = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n_rows and 0 <= jj < n_cols:
                        neighbor_counts.append(int(result.n_solutions[ii, jj]))
                if neighbor_counts and any(
                    abs(current - nc) > 1 for nc in neighbor_counts
                ):
                    suspicious.append((i, j))

        if not suspicious:
            return result

        if self.verbose:
            print(f"\nPost-refining {len(suspicious)} suspicious points...")
        progress = ProgressTracker(
            len(suspicious), label="Refine", interval=2.0, enabled=self.verbose
        )
        grid = self._reconstruct_grid(result)
        params_list = [self._params_at(i, j) for i, j in suspicious]
        seed_lists = [
            [fp.R for fp in (grid[i][j] if grid[i][j] is not None else [])]
            + [fp.R for fp in get_neighbor_solutions(grid, i, j)]
            for i, j in suspicious
        ]
        batched_roots = self._batched_search_at_points(
            params_list, seed_lists, n_random=50, scale=100.0, seed=0
        )

        for (i, j), candidates in zip(suspicious, batched_roots):
            neighbor_fps = get_neighbor_solutions(grid, i, j)
            current_fps = grid[i][j] if grid[i][j] is not None else []
            merged = deduplicate_fixed_points(
                list(current_fps) + candidates,
                distance_tol=self.tolerances.distance_tol,
            )
            if len(merged) != len(current_fps):
                grid[i][j] = merged
                self._update_result_point(result, grid, i, j, merged, neighbor_fps)
            progress.update()
        progress.finish()
        return result

    def _enforce_reflection_symmetry(
        self, result: MultistabilityScanResult
    ) -> MultistabilityScanResult:
        """
        Enforce reflection symmetry across the configured ``symmetry_axis``.

        This is opt-in because reflection symmetry is a property of specific
        models/parameters (e.g. the 2-mode VdP oscillator in ``omega_a``), not
        a generic feature of all R-matrix models.
        """
        ax = self.axes[self.symmetry_axis]
        if not np.allclose(ax, -ax[::-1]):
            if self.verbose:
                print(
                    f"\nWarning: axis '{self.symmetry_axis}' is not symmetric; "
                    "skipping symmetry enforcement."
                )
            return result

        axis_idx = self.axis_names.index(self.symmetry_axis)
        n_rows, n_cols = result.grid_shape
        n_along = n_rows if axis_idx == 0 else n_cols

        for _ in range(3):
            asym: set[tuple[int, int]] = set()
            for i in range(n_rows):
                for j in range(n_cols):
                    if axis_idx == 0:
                        ii = n_rows - 1 - i
                        jj = j
                    else:
                        ii = i
                        jj = n_cols - 1 - j
                    if ii < i or jj < j:
                        continue
                    if result.n_solutions[i, j] != result.n_solutions[ii, jj]:
                        asym.add((i, j))
                        asym.add((ii, jj))
            if not asym:
                break

            if self.verbose:
                print(f"\nEnforcing symmetry: {len(asym)} asymmetric cells...")
            grid = self._reconstruct_grid(result)
            params_list = [self._params_at(i, j) for i, j in asym]
            seed_lists = [
                [fp.R for fp in (grid[i][j] if grid[i][j] is not None else [])]
                + [fp.R for fp in get_neighbor_solutions(grid, i, j)]
                for i, j in asym
            ]
            batched_roots = self._batched_search_at_points(
                params_list, seed_lists, n_random=50, scale=100.0, seed=1
            )

            progress = ProgressTracker(
                len(asym), label="Symmetry", interval=2.0, enabled=self.verbose
            )
            for (i, j), candidates in zip(asym, batched_roots):
                neighbor_fps = get_neighbor_solutions(grid, i, j)
                current_fps = grid[i][j] if grid[i][j] is not None else []
                merged = deduplicate_fixed_points(
                    list(current_fps) + candidates,
                    distance_tol=self.tolerances.distance_tol,
                )
                grid[i][j] = merged
                self._update_result_point(result, grid, i, j, merged, neighbor_fps)
                progress.update()
            progress.finish()
        return result

    def _reconstruct_grid(
        self, result: MultistabilityScanResult
    ) -> list[list[list[FixedPoint] | None]]:
        """Rebuild a grid of FixedPoint lists from packed arrays."""
        n_rows, n_cols = result.grid_shape
        grid: list[list[list[FixedPoint] | None]] = [
            [None for _ in range(n_cols)] for _ in range(n_rows)
        ]
        for i in range(n_rows):
            for j in range(n_cols):
                fps = result.to_fixedpoints((i, j))
                if fps:
                    grid[i][j] = fps
        return grid

    def _update_result_point(
        self,
        result: MultistabilityScanResult,
        grid: list[list[list[FixedPoint] | None]],
        i: int,
        j: int,
        fps: list[FixedPoint],
        neighbor_fps: Sequence[FixedPoint],
    ) -> None:
        """Recompute and store all derived quantities for a single grid point."""
        n = self.model.dim
        if not fps:
            result.n_solutions[i, j] = 0
            for k in range(self.max_branches):
                result.R_matrices[i, j, k] = np.nan + 0j * np.nan
                result.residuals[i, j, k] = np.nan
                result.omegas[i, j, k] = np.nan
                result.J_eigvals[i, j, k] = np.nan + 0j * np.nan
                result.is_psd[i, j, k] = False
                result.is_stable[i, j, k] = False
                result.branch_ids[i, j, k] = -1
            grid[i][j] = []
            return

        result.n_solutions[i, j] = len(fps)
        bidx = match_to_neighbors(fps, neighbor_fps, self.tolerances.branch_match_tol)

        # Reset slots
        for k in range(self.max_branches):
            result.R_matrices[i, j, k] = np.nan + 0j * np.nan
            result.residuals[i, j, k] = np.nan
            result.omegas[i, j, k] = np.nan
            result.J_eigvals[i, j, k] = np.nan + 0j * np.nan
            result.is_psd[i, j, k] = False
            result.is_stable[i, j, k] = False
            result.branch_ids[i, j, k] = -1

        # Recompute post-processing for the new point
        params = self._params_at(i, j)
        R_stack = np.array([fp.R for fp in fps])
        params_grid = {k: np.full(len(R_stack), params[k]) for k in params}
        omegas, J_eigvals, psd, stable = self._postprocess(R_stack, params_grid)

        for k, (fp, bid) in enumerate(zip(fps, bidx)):
            if k >= self.max_branches:
                break
            result.R_matrices[i, j, k] = fp.R
            result.residuals[i, j, k] = fp.residual
            result.omegas[i, j, k] = omegas[k]
            result.J_eigvals[i, j, k] = J_eigvals[k]
            result.is_psd[i, j, k] = bool(psd[k])
            result.is_stable[i, j, k] = bool(stable[k])
            result.branch_ids[i, j, k] = bid

        grid[i][j] = fps


# -----------------------------------------------------------------------------
# Tile-worker machinery for process parallelism
# -----------------------------------------------------------------------------

_WORKER_SCANNER: MultistabilityScan2D | None = None
_WORKER_SEED_RS: list[np.ndarray] | None = None


def _init_tile_worker(
    model,
    base_params,
    axes,
    seed_Rs,
    n_random_guesses,
    tolerances,
    max_branches,
    guess_bounds,
    verbose,
) -> None:
    """Initialize per-worker globals (avoids repeated pickling)."""
    global _WORKER_SCANNER, _WORKER_SEED_RS
    _WORKER_SCANNER = MultistabilityScan2D(
        model=model,
        base_params=base_params,
        axes=axes,
        n_random_guesses=n_random_guesses,
        tolerances=tolerances,
        max_branches=max_branches,
        backend="numpy",  # workers solve on CPU
        parallel=None,
        guess_bounds=guess_bounds,
        verbose=False,
    )
    _WORKER_SEED_RS = list(seed_Rs)


def _tile_worker(
    tile: tuple[int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a single tile inside a worker process."""
    row_start, row_end, col_start, col_end, local_i, local_j = tile
    scanner = _WORKER_SCANNER
    seed_Rs = _WORKER_SEED_RS
    n_rows, n_cols = row_end - row_start, col_end - col_start

    ns, Rm, resids, bids = scanner._allocate_arrays((n_rows, n_cols))
    grid: list[list[list[FixedPoint] | None]] = [
        [None for _ in range(n_cols)] for _ in range(n_rows)
    ]

    order = spiral_order(n_rows, n_cols, local_i, local_j)
    failed_points: list[tuple[int, int]] = []

    for idx, (ii, jj) in enumerate(order):
        global_i, global_j = row_start + ii, col_start + jj
        params = scanner._params_at(global_i, global_j)
        neighbor_fps = get_neighbor_solutions(grid, ii, jj)
        fps = scanner._solve_point(params, neighbor_fps, seed_Rs, idx)

        if not fps:
            fps = independent_search_point(
                scanner.model, params,
                n_samples=200, scale=200.0, seed=idx,
                distance_tol=scanner.tolerances.distance_tol,
                bounds=scanner.guess_bounds,
            )

        if not fps:
            failed_points.append((ii, jj))
            continue

        scanner._store_point(ns, Rm, resids, bids, grid, ii, jj, fps, neighbor_fps)

    if failed_points:
        scanner._retry_failed_points(ns, Rm, resids, bids, grid, failed_points)

    return ns, Rm, resids, bids

