"""Abstract base class for models and result containers."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from numerics.core.backend import get_array_module, to_numpy


@dataclass
class SteadyStateResult:
    """Container for a steady-state solution."""

    R: np.ndarray
    params: dict
    success: bool
    residual: float
    method: str
    message: str = ""
    omega: float | None = None
    J_eigvals: np.ndarray | None = None
    iterations: int | None = None

    def is_stable(self, atol: float = 1e-10) -> bool:
        """Return True if all Jacobian eigenvalues have negative real part."""
        if self.J_eigvals is None:
            return False
        xp = get_array_module(self.J_eigvals)
        return bool(xp.all(xp.real(self.J_eigvals) < atol))

    def to_numpy(self) -> "SteadyStateResult":
        """Return a copy with all arrays converted to NumPy."""
        return SteadyStateResult(
            R=to_numpy(self.R),
            params=self.params.copy(),
            success=self.success,
            residual=self.residual,
            method=self.method,
            message=self.message,
            omega=self.omega,
            J_eigvals=to_numpy(self.J_eigvals) if self.J_eigvals is not None else None,
            iterations=self.iterations,
        )


@dataclass
class ScanResult:
    """Container for parameter-scan results."""

    control_param: str
    control_values: np.ndarray
    steady_states: list[np.ndarray] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    valid_mask: list[bool] = field(default_factory=list)
    omegas: list[float] = field(default_factory=list)
    J_eigvals: list[np.ndarray] = field(default_factory=list)

    def to_arrays(self) -> dict:
        """Convert scan results to NumPy arrays for plotting/analysis."""
        states = np.array(self.steady_states)
        return {
            "control_param": self.control_param,
            "control_values": np.asarray(self.control_values),
            "steady_states": states,
            "residuals": np.asarray(self.residuals),
            "valid_mask": np.asarray(self.valid_mask),
            "omegas": np.asarray(self.omegas),
            "J_eigvals": np.asarray(self.J_eigvals),
        }


class Model(ABC):
    """Abstract base class for R-matrix models."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the R matrix."""
        ...

    @abstractmethod
    def H(self, R: np.ndarray, params: dict) -> np.ndarray:
        """Return the effective Hamiltonian H(R, params)."""
        ...

    @abstractmethod
    def D(self, R: np.ndarray, params: dict) -> np.ndarray:
        """Return the diffusion matrix D(R, params)."""
        ...

    def control_params(self) -> list[str]:
        """Return the list of commonly scanned control parameters."""
        return []

    def liouvillian(self, R: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate L(R) for this model."""
        from numerics.core.liouvillian import liouvillian
        return liouvillian(self.H(R, params), R, self.D(R, params))

    def build_jacobian_builder(self, params: dict | None = None) -> Callable | None:
        """
        Optionally return a JacobianBuilder for symbolic Jacobians.

        Subclasses should override this if symbolic H and D are available.
        """
        return None
