# Nonlinear EP Numerics

A unified Python package for the numerical analysis of nonlinear exceptional-point and bifurcation dynamics in the coherent-amplitude (R-matrix) formalism.

## Equation

The package solves the nonlinear matrix equation

```
Ḋ = L(R) = -i H(R) R + i R H†(R) + D(R)
```

for Hermitian steady-state matrices `R₀`, and extracts the associated frequency

```
ω = Re Tr[H(R₀) R₀] / Tr(R₀)
```

and the Jacobian spectrum for stability / bifurcation analysis.

## Installation

```bash
cd Numerical
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

For GPU acceleration (CUDA 12):

```bash
pip install -e ".[gpu]"
```

## Quick start

```python
import numpy as np
from numerics.models.kerr_3mode_hopf import Kerr3ModeHopf
from numerics.solvers.steady_state import solve_steady_state
from numerics.scans.continuation import ParameterScan

model = Kerr3ModeHopf()
params = {
    "omega_a": 0.0, "omega_b": 1.0, "omega_c": 1.05,
    "kappa_a": 0.05, "kappa_b": 0.06, "kappa_c": 0.056,
    "g_b": 0.2, "g_c": 0.2, "chi": 0.0001,
}

# Single steady state
res = solve_steady_state(model, params, guess=np.eye(3) * 1e3, method="cholesky")
print(res.omega, res.R.diagonal().real)

# Parameter sweep
scan = ParameterScan(
    model, params, "kappa_c", np.linspace(0.056, 0.06, 101),
    solver_method="cholesky"
)
result = scan.run(initial_guess=np.eye(3) * 1e3)
```

## Package structure

```
numerics/
├── core/          # R-matrix math, Liouvillian, frequency, Jacobian
├── models/        # Physical models (2-mode Kerr/vdP, 3-mode Hopf, Kerr-3PA)
├── solvers/       # Steady-state solvers, multi-solution search, backends
├── scans/         # Parameter continuation and bifurcation detection
├── io/            # External-data loaders (npz/csv/pkl/jld2)
└── utils/         # Validation / clustering helpers
```

## Models

| Model | File | Coupling | Key parameters |
|-------|------|----------|----------------|
| `Kerr2Mode` | `models/kerr_2mode.py` | two-mode | `s, omega_A/B, kappa_A/B, g` |
| `VdP2Mode` | `models/vdp_2mode.py` | two-mode vdP | `omega_a/b, gamma_a/b, Gamma, g, D` |
| `Kerr3ModeHopf` | `models/kerr_3mode_hopf.py` | star (a-b, a-c) | `omega_a/b/c, kappa_a/b/c, g_b/c, chi` |
| `Kerr3ModeChain` | `models/kerr_3mode_chain.py` | chain (a-b-c) | `omega_a/b/c, kappa_a/b/c, g_ab/ac, chi` |
| `Kerr3PA` | `models/kerr_3pa.py` | single-mode Hopf | `omega_0, chi, kappa_3, mu, g1` |

## Tests

```bash
pytest tests -v
```

## Example scripts

```bash
python scripts/run_2mode_kerr_sweep.py
python scripts/run_2mode_vdp_sweep.py
python scripts/run_3mode_hopf_sweep.py
```

Outputs are written to `scripts/output/`.

## GPU support

The package keeps NumPy as the default backend. To use CuPy, pass the module
explicitly to backend-aware functions:

```python
from numerics.solvers.backends import get_array_module
xp = get_array_module("cupy")
```

Currently the heavy optimization loops (`scipy.optimize.root`) still run on the
CPU; CuPy is used for array-level kernels where implemented. Full GPU
off-loading of the solvers is a future extension.
