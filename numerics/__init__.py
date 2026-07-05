"""
numerics: Unified numerical package for nonlinear exceptional-point dynamics.

The package solves the nonlinear matrix equation

    dot(R) = L(R) = -i H(R) R + i R H^dagger(R) + D(R)

for Hermitian matrices R, supporting multiple steady-state search,
parameter sweeps, bifurcation detection, and optional GPU acceleration.
"""

__version__ = "0.1.0"

from numerics.core.r_matrix import R_matrix_to_vector, vector_to_R_matrix, get_parameterized_R
from numerics.core.liouvillian import liouvillian, liouvillian_residual
from numerics.core.frequency import omega_frequency, omega_from_H_eigvals
from numerics.core.jacobian import JacobianBuilder, numerical_jacobian
from numerics.models.base import Model, SteadyStateResult, ScanResult
from numerics.solvers.steady_state import solve_steady_state
from numerics.solvers.multi_search import find_steady_states
from numerics.solvers.deflation import find_roots_deflation
from numerics.solvers.continuation import trace_branch_pseudo_arclength
from numerics.solvers.backends import get_array_module
from numerics.scans.continuation import ParameterScan
from numerics.scans.bifurcation import BifurcationLocator
from numerics.scans.multistability import (
    MultistabilityScan2D,
    MultistabilityScanResult,
    ParallelConfig,
    ScanTolerances,
)

__all__ = [
    "R_matrix_to_vector",
    "vector_to_R_matrix",
    "get_parameterized_R",
    "liouvillian",
    "liouvillian_residual",
    "omega_frequency",
    "omega_from_H_eigvals",
    "JacobianBuilder",
    "numerical_jacobian",
    "Model",
    "SteadyStateResult",
    "ScanResult",
    "solve_steady_state",
    "find_steady_states",
    "find_roots_deflation",
    "trace_branch_pseudo_arclength",
    "get_array_module",
    "ParameterScan",
    "BifurcationLocator",
    "MultistabilityScan2D",
    "MultistabilityScanResult",
    "ParallelConfig",
    "ScanTolerances",
]
