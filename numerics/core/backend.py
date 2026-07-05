"""
Array backend abstraction for NumPy / CuPy.

The default backend is NumPy.  Calling ``set_backend("cupy")`` switches the
entire package to CuPy.  Modules should import ``get_array_module`` (or the
lazy ``xp`` proxy) and use it instead of hard-coded ``numpy``.
"""

from __future__ import annotations

import threading
import numpy as np
from typing import Any


_state = {"backend": "numpy"}
_lock = threading.Lock()


def set_backend(name: str) -> None:
    """Set the global array backend to 'numpy' or 'cupy'."""
    name = name.lower()
    if name not in {"numpy", "cupy"}:
        raise ValueError(f"Unsupported backend: {name!r}. Use 'numpy' or 'cupy'.")
    if name == "cupy":
        import cupy  # noqa: F401 -- verify availability
    with _lock:
        _state["backend"] = name


def get_backend() -> str:
    """Return the current backend name."""
    with _lock:
        return _state["backend"]


def get_array_module(xp: Any | None = None) -> Any:
    """
    Return the active array module.

    Parameters
    ----------
    xp : module, ndarray, str, or None
        If None, returns the globally configured backend module.  If a module
        is passed, it is returned unchanged.  If an ndarray is passed, its
        module (numpy or cupy) is returned.
    """
    if xp is not None:
        if isinstance(xp, str):
            if xp == "cupy":
                import cupy as cp
                return cp
            if xp == "numpy":
                return np
            raise ValueError(f"Unsupported backend string: {xp!r}")
        if hasattr(xp, "ndim"):
            # xp is an array, infer its module
            module_name = type(xp).__module__.split(".")[0]
            if module_name == "cupy":
                import cupy as cp
                return cp
            return np
        return xp
    with _lock:
        if _state["backend"] == "cupy":
            import cupy as cp
            return cp
        return np


def to_numpy(arr: Any) -> np.ndarray:
    """Convert any backend array to a NumPy array."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


class _BackendProxy:
    """Lazy module proxy that dispatches attribute access to the active backend."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_array_module(), name)


xp = _BackendProxy()
