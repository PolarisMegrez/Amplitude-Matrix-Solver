"""
Jacobian of the Liouvillian vector field with respect to the real parameters of R.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Callable

from numerics.core.backend import get_array_module
from numerics.core.r_matrix import get_parameterized_R, R_matrix_to_vector, vector_to_R_matrix


class JacobianBuilder:
    """
    Build a callable Jacobian J(r_vec, params) for a model.

    The model must provide *symbolic* Hamiltonian and diffusion matrices
    H_sym(R, params) and D_sym(R, params) as sympy Matrix expressions.

    Note: the lambdified function uses NumPy.  When the active backend is CuPy,
    callers should either move inputs to CPU before calling or use
    ``numerical_jacobian`` on the GPU.
    """

    def __init__(
        self,
        H_sym: sp.Matrix,
        D_sym: sp.Matrix,
        R_sym: sp.Matrix,
        r_vec: sp.Matrix,
        param_symbols: list[sp.Symbol],
        modules: str | None = None,
    ):
        """
        Parameters
        ----------
        H_sym : sympy.Matrix
            Symbolic Hamiltonian as a function of R entries and parameters.
        D_sym : sympy.Matrix
            Symbolic diffusion matrix.
        R_sym : sympy.Matrix
            Parameterized Hermitian R.
        r_vec : sympy.Matrix
            Real parameter vector of R.
        param_symbols : list[sympy.Symbol]
            List of model parameter symbols in the order they will be passed.
        modules : str, optional
            Backend module passed to sympy.lambdify.  Defaults to "numpy".
        """
        self.n = R_sym.shape[0]
        self.r_vec = r_vec
        self.param_symbols = param_symbols
        self.modules = modules or "numpy"

        # L = -i H R + i R H^dagger + D
        L_sym = -sp.I * H_sym * R_sym + sp.I * R_sym * H_sym.H + D_sym
        self.L_sym = sp.simplify(L_sym)

        # Extract real-vector output from L
        l_vec = self._extract_real_vector(self.L_sym, self.n)
        self.J_sym = sp.simplify(l_vec.jacobian(r_vec))

        # Lambdify: all arguments in a single flat list
        all_symbols = list(r_vec) + param_symbols
        self.J_func = sp.lambdify(
            all_symbols,
            self.J_sym,
            modules=[{"ImmutableDenseMatrix": lambda x: x}, self.modules],
        )

    @staticmethod
    def _extract_real_vector(M: sp.Matrix, n: int) -> sp.Matrix:
        """Convert an n x n Hermitian matrix to a real vector."""
        entries = []
        for i in range(n):
            entries.append(M[i, i].expand())
        re_entries = []
        im_entries = []
        for i in range(n):
            for j in range(i + 1, n):
                re_part = sp.simplify((M[i, j] + M[j, i]) / 2)
                im_part = sp.simplify((M[i, j] - M[j, i]) / 2 / sp.I)
                re_entries.append(re_part)
                im_entries.append(im_part)
        return sp.Matrix(entries + re_entries + im_entries)

    def _pack_args(self, r_vec, params):
        """Return a flat tuple of argument arrays passed to J_func."""
        if isinstance(params, dict):
            p = tuple(params[s.name] for s in self.param_symbols)
        else:
            p = tuple(params)
        return tuple(r_vec) + p

    def _stack_output(self, out):
        """Convert the nested-list output of J_func into a dense array."""
        xp = get_array_module(self.modules)
        rows = []
        for row in out:
            arrays = [xp.asarray(entry) for entry in row]
            shape = np.broadcast_shapes(*(a.shape for a in arrays))
            arrays = [xp.broadcast_to(a, shape) for a in arrays]
            rows.append(xp.stack(arrays, axis=-1))
        return xp.stack(rows, axis=-2)

    def __call__(self, r_vec: np.ndarray, params: dict | tuple) -> np.ndarray:
        """
        Evaluate the Jacobian numerically.

        Parameters
        ----------
        r_vec : np.ndarray
            Real parameter vector of R.
        params : dict or tuple
            Model parameter values. If dict, keys must match param_symbols names;
            if tuple, values in the order of param_symbols.

        Returns
        -------
        J : np.ndarray
            Jacobian matrix (n**2 x n**2).
        """
        xp = get_array_module(self.modules)
        r_vec = xp.asarray(r_vec, dtype=float)
        args = self._pack_args(r_vec, params)
        out = self.J_func(*args)
        J = self._stack_output(out)
        return xp.asarray(J, dtype=float)

    def evaluate_batched(
        self,
        r_stack: np.ndarray,
        params: dict | dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Evaluate the Jacobian for a batch of real parameter vectors.

        Parameters
        ----------
        r_stack : np.ndarray
            Array of shape (N, n**2).
        params : dict or dict[str, np.ndarray]
            Model parameters broadcast over the batch.  Scalar values are
            replicated; array values must have shape (N,).

        Returns
        -------
        J : np.ndarray
            Jacobian array of shape (N, n**2, n**2).
        """
        xp = get_array_module(self.modules)
        r_stack = xp.asarray(r_stack, dtype=float)
        N = int(r_stack.shape[0])

        if isinstance(params, dict):
            param_arrays = []
            for s in self.param_symbols:
                v = params[s.name]
                arr = xp.asarray(v)
                if arr.ndim == 0:
                    arr = xp.broadcast_to(arr, (N,))
                elif arr.shape != (N,):
                    raise ValueError(
                        f"Parameter '{s.name}' has shape {arr.shape}, expected (N,) or scalar"
                    )
                param_arrays.append(arr)
        else:
            param_arrays = [xp.asarray(v) for v in params]

        args = tuple(r_stack[:, i] for i in range(r_stack.shape[1])) + tuple(param_arrays)
        out = self.J_func(*args)
        J = self._stack_output(out)
        return xp.asarray(J, dtype=float)


def numerical_jacobian(
    func: Callable,
    x,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Numerical Jacobian by central differences.

    Parameters
    ----------
    func : callable
        Vector-valued function f(x) returning an array of length m.
    x : np.ndarray
        Point at which to evaluate the Jacobian (length n).
    eps : float
        Step size.

    Returns
    -------
    J : np.ndarray
        m x n Jacobian matrix.
    """
    xp = get_array_module(x)
    x = xp.asarray(x, dtype=float)
    f0 = xp.asarray(func(x))
    m = int(f0.size)
    n = int(x.size)
    J = xp.zeros((m, n), dtype=float)

    for j in range(n):
        dx = xp.zeros_like(x)
        dx[j] = eps
        fp = xp.asarray(func(x + dx))
        fm = xp.asarray(func(x - dx))
        J[:, j] = (fp - fm) / (2 * eps)
    return J
