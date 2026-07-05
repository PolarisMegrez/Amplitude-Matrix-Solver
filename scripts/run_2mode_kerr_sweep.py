"""
Example: scan the detuning s in the 2-mode Kerr model and locate the EP.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numerics.models.kerr_2mode import Kerr2Mode
from numerics.scans.continuation import ParameterScan
from numerics.scans.bifurcation import BifurcationLocator


def main():
    model = Kerr2Mode()
    base_params = {
        "omega_A": 1.0,
        "omega_B": 1.0,
        "kappa_A": 0.1,
        "kappa_B": 0.1,
        "g": 0.2,
        "s": 0.0,
    }
    s_vals = np.linspace(-0.5, 0.5, 201)

    scan = ParameterScan(
        model, base_params, "s", s_vals,
        solver_method="cholesky", tol=1e-10
    )
    result = scan.run(initial_guess=np.eye(2))

    arr = result.to_arrays()
    valid = arr["valid_mask"]
    print(f"Valid points: {valid.sum()}/{len(s_vals)}")

    # Plot
    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(arr["control_values"][valid], arr["omegas"][valid], ".-", markersize=2)
    axes[0].set_xlabel("s")
    axes[0].set_ylabel(r"$\omega$")
    axes[0].set_title("2-mode Kerr frequency vs detuning")

    J_re = np.array([
        np.max(np.real(ev)) if not np.isnan(ev).all() else np.nan
        for ev in arr["J_eigvals"]
    ])
    axes[1].semilogy(arr["control_values"][valid], -J_re[valid], ".-", markersize=2)
    axes[1].set_xlabel("s")
    axes[1].set_ylabel(r"$-\max \mathrm{Re}\,\lambda$")
    axes[1].set_title("Stability vs detuning")
    fig.tight_layout()
    fig.savefig(outdir / "kerr_2mode_sweep.png", dpi=150)
    print(f"Saved {outdir / 'kerr_2mode_sweep.png'}")

    # Bifurcation candidates
    loc = BifurcationLocator(result)
    idx = loc.critical_indices()
    if len(idx) > 0:
        print("Most critical s:", arr["control_values"][idx[0]])


if __name__ == "__main__":
    main()
