"""Benchmark sequential vs parallel tile scan using the new framework."""

from __future__ import annotations

import time

import numpy as np

from numerics.models.vdp_2mode import VdP2Mode
from numerics.scans.multistability import MultistabilityScan2D, ParallelConfig


def run(n: int, parallel: ParallelConfig | None):
    model = VdP2Mode()
    base_params = {
        "omega_a": 0.0, "omega_b": 0.0,
        "gamma_a": 2.0, "gamma_b": 1.0,
        "Gamma": 0.01, "g": 0.5, "D": 1.0,
    }
    axes = {
        "gamma_b": np.linspace(0.8, 1.4, n),
        "omega_a": np.linspace(-0.3, 0.3, n),
    }
    scan = MultistabilityScan2D(
        model, base_params, axes,
        n_random_guesses=50,
        max_branches=5,
        backend="numpy",
        parallel=parallel,
        symmetry_axis="omega_a",
        verbose=True,
    )
    t0 = time.time()
    result = scan.run()
    elapsed = time.time() - t0
    return elapsed, result


def main():
    n = 31
    print(f"Grid: {n} x {n} = {n*n} points\n")

    print("Sequential...")
    t_seq, r_seq = run(n, None)
    print(f"  {t_seq:.1f}s  ({n*n/t_seq:.1f} pts/s)")
    print("  counts:", {int(k): int(np.sum(r_seq.n_solutions == k)) for k in np.unique(r_seq.n_solutions)})

    configs = [
        (8, 64),
        (16, 256),
    ]
    for nw, nt in configs:
        print(f"\nParallel: {nw} workers, {nt} tiles...")
        t_par, r_par = run(n, ParallelConfig(n_workers=nw, n_tiles=nt))
        print(f"  {t_par:.1f}s  ({n*n/t_par:.1f} pts/s)")
        print(f"  speedup: {t_seq/t_par:.2f}x")
        print("  counts:", {int(k): int(np.sum(r_par.n_solutions == k)) for k in np.unique(r_par.n_solutions)})
        diff = int(np.sum(r_seq.n_solutions != r_par.n_solutions))
        print(f"  different-count points vs seq: {diff}")


if __name__ == "__main__":
    main()
