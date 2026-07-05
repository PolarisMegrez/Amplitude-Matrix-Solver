"""
Abstract base classes for external data I/O.

The formats actually used in the project today are:
    - NPZ (NumPy archives) for trajectories and distributions
    - CSV for PSD/fit tables
    - Pickle for radial-distribution statistics
    - JLD2 for Julia time-series outputs

This module defines a common interface so that future format changes only
require adding a new adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulationData:
    """
    Generic container for imported simulation data.

    Attributes
    ----------
    trajectories : np.ndarray, optional
        Time-series array, typically shape (n_ic, n_traj, n_steps, n_modes).
    t0 : float, optional
        Initial time.
    dt : float, optional
        Time step.
    metadata : dict
        Parameter dictionary and any other run metadata.
    arrays : dict
        Additional named arrays loaded from the source file.
    """

    trajectories: np.ndarray | None = None
    t0: float | None = None
    dt: float | None = None
    metadata: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


class DataLoader(ABC):
    """Abstract loader for a single file format."""

    @abstractmethod
    def load(self, path: str) -> SimulationData:
        """Load a file and return a SimulationData object."""
        ...

    @abstractmethod
    def save(self, path: str, data: SimulationData) -> None:
        """Save a SimulationData object to a file."""
        ...

    @abstractmethod
    def extensions(self) -> list[str]:
        """Return the file extensions this loader handles."""
        ...
