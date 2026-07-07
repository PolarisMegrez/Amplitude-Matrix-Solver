"""
High-resolution multistability scan of the 2-mode van der Pol oscillator.

This script is now a thin driver around ``numerics.scans.multistability``.
All scanning, post-processing, and symmetry-enforcement logic lives in the
library so it can be reused for other models.
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numerics.models.vdp_2mode import VdP2Mode
from numerics.scans.multistability import (
    MultistabilityScan2D,
    ParallelConfig,
    ScanTolerances,
)


def main():
    parser = argparse.ArgumentParser(
        description="High-resolution multistability scan of the 2-mode VdP oscillator."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the CuPy ring-batch GPU solver (requires CuPy + NVIDIA GPU).",
    )
    args = parser.parse_args()

    model = VdP2Mode()
    base_params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 2.0,
        "gamma_b": 1.0,
        "Gamma": 0.01,
        "g": 0.5,
        "D": 1.0,
    }

    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    # Grid definition.  These can later be moved to a YAML/JSON config file or
    # argparse without touching the scanning logic.
    N_OA = 1801
    N_GB = 1801

    scan = MultistabilityScan2D(
        model=model,
        base_params=base_params,
        axes={
            "gamma_b": np.linspace(0.3, 1.2, N_GB),
            "omega_a": np.linspace(-0.3, 0.3, N_OA),
        },
        n_random_guesses=100,
        tolerances=ScanTolerances(
            residual_tol=1e-4,
            solver_tol=1e-10,
            distance_tol=3.0,
            branch_match_tol=10.0,
        ),
        max_branches=5,
        backend="cupy" if args.gpu else "numpy",
        parallel=None if args.gpu else ParallelConfig(n_workers=32, n_tiles=288),
        guess_bounds="auto",
        gpu_ring_batch=args.gpu,
        verbose=True,
    )

    print(f"\n=== 2D scan: {N_GB} x {N_OA} = {N_GB * N_OA} points ===", flush=True)
    t0 = time.time()
    result = scan.run()
    elapsed = time.time() - t0
    print(f"Scan completed in {elapsed:.1f}s ({elapsed / (N_GB * N_OA):.3f}s per point)")

    print("\nSolution count statistics:")
    max_n = int(np.max(result.n_solutions))
    for n in range(max_n + 1):
        count = int(np.sum(result.n_solutions == n))
        print(f"  {n} solutions: {count}")

    valid = result.branch_ids >= 0
    n_psd = int(np.sum(result.is_psd & valid))
    n_stable = int(np.sum(result.is_stable & valid))
    n_total_slots = int(np.sum(valid))
    print("\nPhysical/stable summary:")
    print(f"  Total converged branches: {n_total_slots}")
    print(f"  Positive-semidefinite R:  {n_psd} ({100 * n_psd / n_total_slots:.1f}%)")
    print(f"  Stable (all Re lambda < 0): {n_stable} ({100 * n_stable / n_total_slots:.1f}%)")

    result.metadata["elapsed_seconds"] = elapsed
    result.save_npz(outdir / "vdp_2mode_scan_results.npz")
    print(f"\nSaved {outdir / 'vdp_2mode_scan_results.npz'}")


if __name__ == "__main__":
    main()
