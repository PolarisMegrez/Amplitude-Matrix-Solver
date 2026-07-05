"""
Example: scan kappa_c in the 3-mode Kerr/Hopf model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numerics.models.kerr_3mode_hopf import Kerr3ModeHopf
from numerics.scans.continuation import ParameterScan
from numerics.scans.bifurcation import BifurcationLocator


def main():
    model = Kerr3ModeHopf()
    base_params = {
        "omega_a": 0.0,
        "omega_b": 1.0,
        "omega_c": 1.05,
        "kappa_a": 0.05,
        "kappa_b": 0.06,
        "kappa_c": 0.056,
        "g_b": 0.2,
        "g_c": 0.2,
        "chi": 0.0001,
    }
    kappa_c_vals = np.linspace(0.056, 0.06, 201)

    scan = ParameterScan(
        model, base_params, "kappa_c", kappa_c_vals,
        solver_method="cholesky", fallback_guess=np.eye(3) * 1000.0,
        tol=1e-10
    )
    result = scan.run(initial_guess=np.eye(3) * 1000.0)

    arr = result.to_arrays()
    valid = arr["valid_mask"]
    print(f"Valid points: {valid.sum()}/{len(kappa_c_vals)}")

    outdir = Path(__file__).parent / "output"
    outdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(arr["control_values"][valid], arr["omegas"][valid], ".-", markersize=2)
    axes[0].set_xlabel(r"$\kappa_c$")
    axes[0].set_ylabel(r"$\omega$")
    axes[0].set_title("3-mode Hopf frequency vs loss")

    J_re = np.array([
        np.max(np.real(ev)) if not np.isnan(ev).all() else np.nan
        for ev in arr["J_eigvals"]
    ])
    axes[1].semilogy(arr["control_values"][valid], -J_re[valid], ".-", markersize=2)
    axes[1].set_xlabel(r"$\kappa_c$")
    axes[1].set_ylabel(r"$-\max \mathrm{Re}\,\lambda$")
    axes[1].set_title("Stability vs loss")
    fig.tight_layout()
    fig.savefig(outdir / "kerr_3mode_hopf_sweep.png", dpi=150)
    print(f"Saved {outdir / 'kerr_3mode_hopf_sweep.png'}")

    loc = BifurcationLocator(result)
    idx = loc.critical_indices()
    if len(idx) > 0:
        print("Most critical kappa_c:", arr["control_values"][idx[0]])


if __name__ == "__main__":
    main()
