"""Tests for NumPy/CuPy backend abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from numerics.core.backend import get_array_module, set_backend, get_backend
from numerics.models.vdp_2mode import VdP2Mode


def test_backend_switch():
    set_backend("numpy")
    assert get_backend() == "numpy"
    xp = get_array_module()
    assert xp.__name__ == "numpy"

    cupy = pytest.importorskip("cupy")
    set_backend("cupy")
    assert get_backend() == "cupy"
    xp = get_array_module()
    assert xp.__name__ == "cupy"

    # Restore
    set_backend("numpy")


def test_vdp_model_numpy():
    set_backend("numpy")
    model = VdP2Mode()
    params = {"omega_a": 0.0, "omega_b": 0.0, "gamma_a": 2.0, "gamma_b": 1.0,
              "Gamma": 0.01, "g": 0.5, "D": 1.0}
    R = np.eye(2, dtype=complex)
    H = model.H(R, params)
    assert isinstance(H, np.ndarray)
    assert H.shape == (2, 2)


def test_vdp_model_cupy():
    cupy = pytest.importorskip("cupy")
    set_backend("cupy")
    try:
        model = VdP2Mode()
        params = {"omega_a": 0.0, "omega_b": 0.0, "gamma_a": 2.0, "gamma_b": 1.0,
                  "Gamma": 0.01, "g": 0.5, "D": 1.0}
        R = cupy.eye(2, dtype=complex)
        H = model.H(R, params)
        assert isinstance(H, cupy.ndarray)
        assert H.shape == (2, 2)

        # Batched
        Rb = cupy.stack([R, R])
        Hb = model.H(Rb, params)
        assert Hb.shape == (2, 2, 2)
    finally:
        set_backend("numpy")
