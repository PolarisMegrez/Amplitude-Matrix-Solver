"""Benchmark batched GPU/CPU steady-state solver."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numerics.core.backend import set_backend, get_array_module
from numerics.models.vdp_2mode import VdP2Mode
from numerics.solvers.batched import solve_steady_state_batched
from numerics.solvers.multi_search import find_steady_states


def benchmark_cpu(model, params_list, seeds_per_point=4, scale=100.0):
    """Sequential CPU multi-search."""
    t0 = time.time()
    all_results = []
    for params in params_list:
        sols = find_steady_states(
            model, params,
            n_samples=seeds_per_point,
            scale=scale,
            seed=42,
            solver_method="root",
            distance_tol=3.0,
            residual_tol=1e-4,
            tol=1e-10,
            use_jacobian=False,
        )
        all_results.append(sols)
    elapsed = time.time() - t0
    return elapsed, all_results


def benchmark_batched(model, params_batch, guesses):
    """Batched Newton solver."""
    t0 = time.time()
    results = solve_steady_state_batched(
        model, params_batch, guesses,
        max_iter=30, tol=1e-10,
    )
    elapsed = time.time() - t0
    return elapsed, results


def make_param_batch(B, omega_a_center=0.02, gamma_b=0.9):
    return {
        "omega_a": np.full(B, omega_a_center),
        "omega_b": np.zeros(B),
        "gamma_a": np.full(B, 2.0),
        "gamma_b": np.full(B, gamma_b),
        "Gamma": np.full(B, 0.01),
        "g": np.full(B, 0.5),
        "D": np.full(B, 1.0),
    }


def make_guesses(B, G, scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    guesses = np.zeros((B, G, 2, 2), dtype=complex)
    for b in range(B):
        for g in range(G):
            diag = rng.exponential(scale=scale, size=2)
            re = rng.normal(0, scale)
            im = rng.normal(0, scale)
            guesses[b, g] = [[diag[0], re + 1j * im], [re - 1j * im, diag[1]]]
    return guesses


def main():
    model = VdP2Mode()
    B_values = [64, 256, 1024]
    G = 4

    print("=" * 60)
    print("Batched solver benchmark")
    print("=" * 60)

    # CPU benchmark
    print("\n--- CPU (sequential find_steady_states) ---")
    for B in B_values:
        params_list = [{
            "omega_a": 0.02,
            "omega_b": 0.0,
            "gamma_a": 2.0,
            "gamma_b": 0.9,
            "Gamma": 0.01,
            "g": 0.5,
            "D": 1.0,
        } for _ in range(B)]
        elapsed, results = benchmark_cpu(model, params_list, seeds_per_point=G)
        total_solves = sum(len(r) for r in results)
        print(f"B={B:5d}: {elapsed:.3f}s  ({B/elapsed:.1f} pts/s, {total_solves} converged)")

    # GPU benchmark
    print("\n--- GPU (batched Newton) ---")
    set_backend("cupy")
    xp = get_array_module()
    for B in B_values:
        params_batch = make_param_batch(B)
        guesses = make_guesses(B, G, seed=42)
        elapsed, results = benchmark_batched(model, params_batch, guesses)
        total_solves = sum(len(row) for row in results) * G
        print(f"B={B:5d}: {elapsed:.3f}s  ({B/elapsed:.1f} pts/s, batch processed)")
    set_backend("numpy")


if __name__ == "__main__":
    main()
