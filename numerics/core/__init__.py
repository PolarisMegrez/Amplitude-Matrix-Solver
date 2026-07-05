"""Core mathematical utilities for the R-matrix formalism."""

from .r_matrix import R_matrix_to_vector, vector_to_R_matrix, get_parameterized_R
from .liouvillian import liouvillian, liouvillian_residual
from .frequency import omega_frequency, omega_from_H_eigvals
from .jacobian import JacobianBuilder, numerical_jacobian

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
]
