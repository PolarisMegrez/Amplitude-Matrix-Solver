"""
Example: scan gamma_a in the 2-mode van der Pol oscillator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numerics.models.vdp_2mode import VdP2Mode
from numerics.scans.continuation import ParameterScan
from numerics.core.r_matrix import vector_to_R_matrix


def main():
    model = VdP2Mode()
    base_params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "gamma_a": 2.0,
        "gamma_b": 0.5,
        "Gamma": 0.0001,
        "g": 0.5,
        "D": 1.0,
    }
    gamma_a_vals = np.linspace(2.0, 10.0, 201)

    guess = vector_to_R_matrix(
        np.array([20000.0, 20000.0, 20000.0, -20000.0])
    )

    scan = ParameterScan(
        model, base_params, "gamma_a", gamma_a_vals,
        solver_method="cholesky", fallback_guess=guess, tol=1e-10
    )
    result = scan.run(initial_guess=guess)

    arr = result.to_arrays()
    valid = arr["valid_mask"]
    print(f"Valid points: {valid.sum()}/{len(gamma_a_vals)}")

    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        arr["control_values"][valid],
        arr["steady_states"][valid, 0],
        ".-", markersize=2, label=r"$R_{11}$"
    )
    axes[0].plot(
        arr["control_values"][valid],
        arr["steady_states"][valid, 1],
        ".-", markersize=2, label=r"$R_{22}$"
    )
    axes[0].set_xlabel(r"$\gamma_a$")
    axes[0].set_ylabel("Steady-state populations")
    axes[0].legend()

    J_re = np.array([
        np.max(np.real(ev)) if not np.isnan(ev).all() else np.nan
        for ev in arr["J_eigvals"]
    ])
    axes[1].semilogy(arr["control_values"][valid], -J_re[valid], ".-", markersize=2)
    axes[1].set_xlabel(r"$\gamma_a$")
    axes[1].set_ylabel(r"$-\max \mathrm{Re}\,\lambda$")
    fig.tight_layout()
    fig.savefig(outdir / "vdp_2mode_sweep.png", dpi=150)
    print(f"Saved {outdir / 'vdp_2mode_sweep.png'}")


if __name__ == "__main__":
    main()
