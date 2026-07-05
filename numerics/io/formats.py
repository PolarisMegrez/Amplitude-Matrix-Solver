"""
Concrete I/O adapters for formats used in the project.

- NPZ: NumPy archives, used for trajectories and 2-D histograms.
- CSV: Comma-separated tables, used for PSD/fit results.
- Pickle: Python object archives, used for radial-distribution statistics.
- JLD2: Julia HDF5-based format, used for time-series outputs.
"""

from __future__ import annotations

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Any

from numerics.io.base import DataLoader, SimulationData


class NPZLoader(DataLoader):
    """Loader for NumPy .npz archives."""

    def extensions(self) -> list[str]:
        return [".npz"]

    def load(self, path: str) -> SimulationData:
        data = np.load(path, allow_pickle=True)
        arrays = {key: data[key] for key in data.files}

        metadata = {}
        if "params" in arrays and isinstance(arrays["params"], np.ndarray):
            try:
                metadata = json.loads(str(arrays["params"]))
            except Exception:
                metadata = {"params": arrays["params"]}

        trajectories = arrays.pop("data", None)
        t0 = float(arrays.pop("t0", 0.0)) if "t0" in arrays else None
        dt = float(arrays.pop("dt", 0.0)) if "dt" in arrays else None

        return SimulationData(
            trajectories=trajectories,
            t0=t0,
            dt=dt,
            metadata=metadata,
            arrays=arrays,
        )

    def save(self, path: str, data: SimulationData) -> None:
        save_dict = {}
        if data.trajectories is not None:
            save_dict["data"] = data.trajectories
        if data.t0 is not None:
            save_dict["t0"] = data.t0
        if data.dt is not None:
            save_dict["dt"] = data.dt
        save_dict["params"] = json.dumps(data.metadata)
        save_dict.update(data.arrays)
        np.savez(path, **save_dict)


class CSVLoader(DataLoader):
    """Loader for CSV tables using numpy.genfromtxt."""

    def extensions(self) -> list[str]:
        return [".csv"]

    def load(self, path: str) -> SimulationData:
        table = np.genfromtxt(path, delimiter=",", names=True)
        return SimulationData(
            metadata={"source": path, "columns": table.dtype.names},
            arrays={"table": table},
        )

    def save(self, path: str, data: SimulationData) -> None:
        if "table" not in data.arrays:
            raise ValueError("CSV save requires data.arrays['table'].")
        table = data.arrays["table"]
        header = ",".join(table.dtype.names)
        np.savetxt(path, table, delimiter=",", header=header, comments="")


class PickleLoader(DataLoader):
    """Loader for Python pickle files."""

    def extensions(self) -> list[str]:
        return [".pkl", ".pickle"]

    def load(self, path: str) -> SimulationData:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, SimulationData):
            return obj

        if isinstance(obj, dict):
            return SimulationData(
                metadata=obj.get("metadata", {}),
                arrays=obj,
            )

        return SimulationData(arrays={"object": obj})

    def save(self, path: str, data: SimulationData) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f)


class JLD2Loader(DataLoader):
    """
    Loader for Julia JLD2 files.

    This is a thin wrapper that attempts to read JLD2 via h5py. If h5py is
    not available, it raises ImportError.
    """

    def extensions(self) -> list[str]:
        return [".jld2"]

    def load(self, path: str) -> SimulationData:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read JLD2 files. "
                "Install it with: pip install h5py"
            ) from exc

        arrays = {}
        metadata = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                item = f[key]
                if hasattr(item, "shape"):
                    arrays[key] = np.asarray(item)
                elif hasattr(item, "items"):
                    metadata[key] = {
                        k: np.asarray(v) for k, v in item.items()
                    }
                else:
                    try:
                        metadata[key] = item[()]
                    except Exception:
                        metadata[key] = str(item)

        return SimulationData(metadata=metadata, arrays=arrays)

    def save(self, path: str, data: SimulationData) -> None:
        raise NotImplementedError("JLD2 write is not supported from Python.")
