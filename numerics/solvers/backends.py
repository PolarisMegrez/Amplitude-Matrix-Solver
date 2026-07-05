"""Re-export the backend utilities for backward compatibility."""

from numerics.core.backend import get_array_module, set_backend, get_backend, to_numpy

__all__ = ["get_array_module", "set_backend", "get_backend", "to_numpy"]
