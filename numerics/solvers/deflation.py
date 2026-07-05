"""
Deflation-based multi-root discovery for R-matrix steady-state equations.

The deflation operator multiplies the original residual F(r) by a factor that
has a pole at every already-known root.  This repels Newton iterations so that
subsequent initial guesses converge to distinct solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from numerics.core.backend import get_array_module, to_numpy
from numerics.core.frequency import omega_frequency
from numerics.core.jacobian import JacobianBuilder, numerical_jacobian
from numerics.core.liouvillian import liouvillian_vector
from numerics.core.r_matrix import R_matrix_to_vector, vector_to_R_matrix
from numerics.utils.fixed_points import FixedPoint, deduplicate_fixed_points


@dataclass
class DeflationOptions:
    """Hyperparameters for deflated root finding."""

    alpha: float = 1e-2
    max_iter: int = 100
    tol: float = 1e-10
    residual_tol: float = 1e-8
    distance_tol: float = 1.0
    use_jacobian: bool = True
    damping: float = 1.0
    fallback_to_cholesky: bool = False


class DeflationOperator:
    """
    Multiply a residual by factors that introduce poles at known roots.

    For a known root :math:`r_k`, the contribution is

    .. math::
        m_k(x) = 1 + \\frac{\\alpha}{\\|x - r_k\\|^2},

    so that :math:`m_k \\to \\infty` as :math:`x \\to r_k`.  The combined
    deflation factor is :math:`M(x) = \\prod_k m_k(x)`.
    """

    def __init__(self, known_roots: np.ndarray | Sequence | None = None, alpha: float = 1e-2):
        self.alpha = float(alpha)
        if known_roots is None:
            self.known: np.ndarray = np.zeros((0, 0), dtype=float)
        else:
            self.known = np.asarray(known_roots, dtype=float)
            if self.known.ndim == 1 and self.known.size > 0:
                self.known = self.known[None, :]

    @property
    def n_known(self) -> int:
        return self.known.shape[0]

    def add_root(self, root: np.ndarray) -> None:
        """Register a newly found root."""
        root = np.asarray(root, dtype=float).reshape(1, -1)
        if self.n_known == 0:
            self.known = root
        else:
            self.known = np.concatenate([self.known, root], axis=0)

    def factor(self, x: np.ndarray) -> np.ndarray:
        """Evaluate M(x)."""
        xp = get_array_module(x)
        x = xp.asarray(x, dtype=float)
        if self.n_known == 0:
            return xp.ones(x.shape[:-1] if x.ndim > 1 else ())
        # x shape (..., m); known shape (K, m)
        diff = x[..., None, :] - self.known[None, ...]
        d2 = xp.sum(diff * diff, axis=-1)
        contrib = 1.0 + self.alpha / d2
        return xp.prod(contrib, axis=-1)

    def log_factor_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate :math:`\\nabla \\log M(x)`.

        For each contribution :math:`m_k = 1 + \\alpha / d_k`,

        .. math::
            \\nabla \\log m_k = -\\frac{2 \\alpha (x - r_k)}{d_k (d_k + \\alpha)}.
        """
        xp = get_array_module(x)
        x = xp.asarray(x, dtype=float)
        if self.n_known == 0:
            return xp.zeros_like(x)
        diff = x[..., None, :] - self.known[None, ...]
        d2 = xp.sum(diff * diff, axis=-1)  # (..., K)
        coeff = -2.0 * self.alpha / (d2 * (d2 + self.alpha))  # (..., K)
        grad = xp.sum(coeff[..., None] * diff, axis=-2)  # (..., m)
        return grad

    def residual(self, F: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the deflated residual M(x) F(x)."""
        return self.factor(x) * F

    def jacobian(self, J_F: np.ndarray, F: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Jacobian of the deflated residual.

        Using :math:`G = M F` and :math:`\\nabla M = M \\nabla \\log M`,

        .. math::
            J_G = M \\left( J_F + F \\nabla \\log M^\\top \\right).
        """
        xp = get_array_module(x)
        M = self.factor(x)
        grad = self.log_factor_gradient(x)
        if x.ndim == 1:
            return M * (J_F + xp.outer(F, grad))
        # Batched: x (..., m), F (..., m), J_F (..., m, m)
        return M[..., None, None] * (
            J_F + F[..., :, None] * grad[..., None, :]
        )


def _as_vector_guess(guess, dim: int):
    """Convert an R-matrix guess to a real vector if needed."""
    arr = np.asarray(guess)
    if arr.ndim == 1 and arr.size == dim * dim:
        return arr
    if arr.shape == (dim, dim):
        return R_matrix_to_vector(arr, dim)
    raise ValueError(f"Guess has unexpected shape {arr.shape}")


def _compute_omega_and_eigvals(
    model,
    params: dict,
    r_vec: np.ndarray,
    jac_builder: JacobianBuilder | None,
) -> tuple[float | None, np.ndarray | None]:
    """Compute omega and Jacobian eigenvalues for a converged solution."""
    R = vector_to_R_matrix(r_vec, model.dim)
    try:
        H = model.H(R, params)
        omega = omega_frequency(H, R)
    except Exception:
        omega = None
    try:
        if jac_builder is not None:
            J = to_numpy(jac_builder(r_vec, params))
        else:
            F_func = lambda rv: to_numpy(liouvillian_vector(model, rv, params))
            J = numerical_jacobian(F_func, r_vec)
        J_eigvals = np.linalg.eigvals(J)
    except Exception:
        J_eigvals = None
    return omega, J_eigvals


def find_roots_deflation(
    model,
    params: dict,
    guesses: Sequence[np.ndarray],
    known_roots: np.ndarray | Sequence | None = None,
    opts: DeflationOptions | None = None,
) -> list[FixedPoint]:
    """
    Discover multiple steady-state solutions using deflated Newton iterations.

    Parameters
    ----------
    model : Model
        R-matrix model.
    params : dict
        Model parameters.
    guesses : sequence of np.ndarray
        Initial guesses, either as R matrices or real vectors.
    known_roots : array-like, optional
        Roots that should be excluded from discovery.  If omitted, an empty
        deflation operator is used.
    opts : DeflationOptions, optional
        Tuning parameters.

    Returns
    -------
    roots : list[FixedPoint]
        Unique newly discovered roots (including those passed in ``known_roots``
        is not done; only roots reached from ``guesses`` are returned).
    """
    opts = opts or DeflationOptions()
    xp = np  # deflation currently runs on the CPU

    dim = model.dim
    m = dim * dim

    deflator = DeflationOperator(known_roots, alpha=opts.alpha)

    # Pre-build model residual and Jacobian.
    def residual_func(r: np.ndarray) -> np.ndarray:
        return liouvillian_vector(model, r, params)

    jac_builder = None
    if opts.use_jacobian:
        try:
            jac_builder = model.build_jacobian_builder(params, modules="numpy")
        except Exception:
            jac_builder = None

    def jacobian_func(r: np.ndarray) -> np.ndarray:
        if jac_builder is not None:
            return jac_builder(r, params)
        return numerical_jacobian(residual_func, r)

    found: list[FixedPoint] = []
    found_vecs: list[np.ndarray] = []

    def is_new_root(r_vec: np.ndarray) -> bool:
        if not found_vecs:
            return True
        r_vec = np.asarray(r_vec)
        return all(np.linalg.norm(r_vec - v) >= opts.distance_tol for v in found_vecs)

    for guess in guesses:
        x = _as_vector_guess(guess, dim).astype(float)
        try:
            x_new = _deflated_newton(
                x, residual_func, jacobian_func, deflator,
                max_iter=opts.max_iter,
                tol=opts.tol,
                damping=opts.damping,
            )
        except Exception:
            continue

        if x_new is None:
            continue

        res_norm = float(np.linalg.norm(residual_func(x_new)))
        if res_norm > opts.residual_tol:
            continue

        if not is_new_root(x_new):
            continue

        R = vector_to_R_matrix(x_new, dim)
        omega, J_eigvals = _compute_omega_and_eigvals(model, params, x_new, jac_builder)
        found_vecs.append(x_new.copy())
        found.append(FixedPoint(
            R=R,
            residual=res_norm,
            omega=omega,
            J_eigvals=J_eigvals,
        ))
        deflator.add_root(x_new)

    return deduplicate_fixed_points(found, distance_tol=opts.distance_tol)


def _deflated_newton(
    x0: np.ndarray,
    residual_func: callable,
    jacobian_func: callable,
    deflator: DeflationOperator,
    max_iter: int,
    tol: float,
    damping: float,
) -> np.ndarray | None:
    """Run a single deflated Newton solve from ``x0``."""
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        F = residual_func(x)
        norm_F = float(np.linalg.norm(F))
        if norm_F <= tol:
            return x

        J = jacobian_func(x)
        grad = deflator.log_factor_gradient(x)
        # Solve the *unscaled* correction equation
        # (J_F + F grad^T) dx = -F, which is equivalent to J_G dx = -G
        # while avoiding the possibly large factor M.
        A = J + np.outer(F, grad)
        try:
            dx = np.linalg.solve(A, -F)
        except np.linalg.LinAlgError:
            # Fall back to least-squares if the deflated Jacobian is singular.
            dx, *_ = np.linalg.lstsq(A, -F, rcond=None)

        x = x + damping * dx

    # Final residual check.
    F = residual_func(x)
    if float(np.linalg.norm(F)) <= tol:
        return x
    return None
