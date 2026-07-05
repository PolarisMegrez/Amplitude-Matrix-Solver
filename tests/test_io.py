"""Tests for I/O adapters."""

import numpy as np
import tempfile
from pathlib import Path

from numerics.io.formats import NPZLoader, CSVLoader, PickleLoader
from numerics.io.base import SimulationData


def test_npz_round_trip():
    """NPZ loader can save and reload SimulationData."""
    loader = NPZLoader()
    data = SimulationData(
        trajectories=np.random.rand(2, 3, 4, 5),
        t0=0.0,
        dt=0.01,
        metadata={"model": "test", "epsilon": 0.1},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.npz"
        loader.save(str(path), data)
        loaded = loader.load(str(path))
        assert np.allclose(loaded.trajectories, data.trajectories)
        assert loaded.t0 == data.t0
        assert loaded.dt == data.dt
        assert loaded.metadata["epsilon"] == 0.1


def test_csv_loader_shape():
    """CSV loader reads a table with named columns."""
    loader = CSVLoader()
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fit_results.csv")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        pytest.skip("fit_results.csv not found")
    data = loader.load(path)
    assert "epsilon" in data.metadata["columns"]
    assert len(data.arrays["table"]) > 0


import pytest
