"""Linear algebra operations submodule (`zero_keras.ops.linalg`)."""

from __future__ import annotations

from zero_keras.ops import (
    cholesky as cholesky,
    cholesky_inverse as cholesky_inverse,
    det as det,
    eig as eig,
    eigh as eigh,
    inv as inv,
    jvp as jvp,
    lstsq as lstsq,
    lu_factor as lu_factor,
    norm as norm,
    qr as qr,
    solve as solve,
    solve_triangular as solve_triangular,
    svd as svd,
)

__all__ = [
    "cholesky",
    "cholesky_inverse",
    "det",
    "eig",
    "eigh",
    "inv",
    "jvp",
    "lstsq",
    "lu_factor",
    "norm",
    "qr",
    "solve",
    "solve_triangular",
    "svd",
]
