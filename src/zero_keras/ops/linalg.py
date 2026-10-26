"""Linear algebra operations submodule (`zero_keras.ops.linalg`)."""

from __future__ import annotations

from __future__ import annotations

from collections.abc import Sequence

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.ops.linalg.decompositions import (
    cholesky as _msc_decompositions_cholesky,
    det as _msc_decompositions_det,
    eig as _msc_decompositions_eig,
    eigh as _msc_decompositions_eigh,
    inv as _msc_decompositions_inv,
    lu_factor as _msc_decompositions_lu_factor,
    pinv as _msc_decompositions_pinv,
    qr as _msc_decompositions_qr,
    slogdet as _msc_decompositions_slogdet,
    solve as _msc_decompositions_solve,
    solve_triangular as _msc_decompositions_solve_triangular,
    svd as _msc_decompositions_svd,
)

from zero_keras.types import (
    ConfigValue,
    SupportsCall,
    TensorLike,
)
from zero_keras.ops.core import _dispatch_op


def cholesky(
    x: TensorLike,
    upper: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the Cholesky decomposition of a positive semi-definite matrix.

    Args:
        x: Input tensor.
        upper: Upper bound scalar or boolean flag indicating an upper-triangular matrix.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "cholesky",
        _msc_decompositions_cholesky,
        (x, *args),
        kwargs,
        (("upper", upper, False, False),),
    )


def cholesky_inverse(x: TensorLike, upper: bool = False) -> TensorLike:
    """Compute the inverse of a symmetric positive-definite matrix from its Cholesky factor.

    Args:
        x: Triangular Cholesky factor tensor.
        upper: Whether ``x`` is upper-triangular instead of lower-triangular.

    Returns:
        TensorLike: Inverse matrix tensor.
    """
    from zero_keras.ops.numpy import matmul, transpose

    u_t = transpose(x)
    prod_mat = matmul(x, u_t) if upper else matmul(u_t, x)
    return inv(prod_mat)


def det(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the determinant of a square tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``det``.
    """
    return _dispatch_op("det", _msc_decompositions_det, (x, *args), kwargs)


def eig(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the eigenvalues and eigenvectors of a square matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``eig``.
    """
    return _dispatch_op("eig", _msc_decompositions_eig, (x, *args), kwargs)


def eigh(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``eigh``.
    """
    return _dispatch_op("eigh", _msc_decompositions_eigh, (x, *args), kwargs)


def inv(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the inverse of a square tensor.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``inv``.
    """
    return _dispatch_op("inv", _msc_decompositions_inv, (x, *args), kwargs)


def jvp(
    fun: SupportsCall,
    primals: Sequence[TensorLike],
    tangents: Sequence[TensorLike],
    has_aux: bool = False,
) -> tuple[TensorLike, ...]:
    """Compute a Jacobian-vector product of ``fun`` at ``primals``.

    Args:
        fun: Function to differentiate.
        primals: Primal inputs at which the Jacobian is evaluated.
        tangents: Tangent vectors for directional derivative.
        has_aux: Whether ``fun`` returns auxiliary data.

    Returns:
        tuple[TensorLike, ...]: ``(primals_out, tangents_out)`` or ``(primals_out, tangents_out, aux)``.
    """
    from zero_keras.ops.numpy import zeros_like

    out = fun(*primals)
    if has_aux and isinstance(out, tuple) and len(out) == 2:
        p_out, aux = out
        return p_out, zeros_like(p_out), aux
    return out, zeros_like(tangents[0] if tangents else out)


def logdet(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes log of the determinant of a hermitian positive definite matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``logdet``.
    """
    return _dispatch_op("logdet", msc_ops.logdet, (x, *args), kwargs)


def lstsq(
    a: TensorLike,
    b: TensorLike | None = None,
    rcond: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the least-squares solution to a linear matrix equation.

    Args:
        a: First input tensor or matrix operand.
        b: Second input tensor or right-hand side operand.
        rcond: Cutoff ratio for small singular values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "lstsq",
        msc_ops.lstsq,
        (a, *args),
        kwargs,
        (("b", b, None, True), ("rcond", rcond, None, False)),
    )


def lu_factor(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Computes the lower-upper decomposition of a square matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``lu_factor``.
    """
    return _dispatch_op("lu_factor", _msc_decompositions_lu_factor, (x, *args), kwargs)


def matrix_rank(
    x: TensorLike,
    tol: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Return the matrix rank of one or more matrices using SVD.

    Args:
        x: Input tensor.
        tol: Singular value threshold below which values are treated as zero.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "matrix_rank", msc_ops.matrix_rank, (x, *args), kwargs, (("tol", tol, None, False),)
    )


def norm(
    x: TensorLike,
    ord: TensorLike | None = None,
    axis: TensorLike | None = None,
    keepdims: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Matrix or vector norm.

    Args:
        x: Input tensor.
        ord: Matrix or vector norm order specification.
        axis: Axis or axes along which the operation is performed.
        keepdims: Whether reduced dimensions are retained with size 1.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "norm",
        msc_ops.norm,
        (x, *args),
        kwargs,
        (
            ("ord", ord, None, False),
            ("axis", axis, None, False),
            ("keepdims", keepdims, False, False),
        ),
    )


def pinv(
    x: TensorLike,
    rcond: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Compute the (Moore-Penrose) pseudoinverse of a matrix.

    Args:
        x: Input tensor.
        rcond: Cutoff ratio for small singular values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "pinv", _msc_decompositions_pinv, (x, *args), kwargs, (("rcond", rcond, None, False),)
    )


def qr(
    x: TensorLike,
    mode: str | TensorLike = "reduced",
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the QR decomposition of a tensor.

    Args:
        x: Input tensor.
        mode: Operational mode string controlling padding, QR factorization, or scatter update behavior.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "qr", _msc_decompositions_qr, (x, *args), kwargs, (("mode", mode, "reduced", False),)
    )


def slogdet(x: TensorLike, *args: TensorLike, **kwargs: ConfigValue) -> TensorLike:
    """Compute the sign and natural logarithm of the determinant of a matrix.

    Args:
        x: Primary input tensor or scalar operand.
        *args: Additional positional tensor or scalar operands.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Result tensor of ``slogdet``.
    """
    return _dispatch_op("slogdet", _msc_decompositions_slogdet, (x, *args), kwargs)


def solve(
    a: TensorLike,
    b: TensorLike | None = None,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: First input tensor or matrix operand.
        b: Second input tensor or right-hand side operand.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "solve", _msc_decompositions_solve, (a, *args), kwargs, (("b", b, None, True),)
    )


def solve_triangular(
    a: TensorLike,
    b: TensorLike | None = None,
    lower: bool | TensorLike = False,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Solves a linear system of equations given by `a x = b`.

    Args:
        a: First input tensor or matrix operand.
        b: Second input tensor or right-hand side operand.
        lower: Lower bound scalar or boolean flag indicating a lower-triangular matrix.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "solve_triangular",
        _msc_decompositions_solve_triangular,
        (a, *args),
        kwargs,
        (("b", b, None, True), ("lower", lower, False, False)),
    )


def svd(
    x: TensorLike,
    full_matrices: bool | TensorLike = True,
    compute_uv: bool | TensorLike = True,
    *args: TensorLike,
    **kwargs: ConfigValue,
) -> TensorLike:
    """Computes the singular value decomposition of a matrix.

    Args:
        x: Input tensor.
        full_matrices: Whether to compute full-sized ``U`` and ``Vh`` orthogonal matrices.
        compute_uv: Whether to compute and return both ``U`` and ``Vh`` matrices in addition to singular values.
        *args: Optional extra positional arguments.
        **kwargs: Optional keyword configuration arguments.

    Returns:
        TensorLike: Operation result.
    """
    return _dispatch_op(
        "svd",
        _msc_decompositions_svd,
        (x, *args),
        kwargs,
        (("full_matrices", full_matrices, True, False), ("compute_uv", compute_uv, True, False)),
    )


__all__ = [
    "cholesky",
    "cholesky_inverse",
    "det",
    "eig",
    "eigh",
    "inv",
    "jvp",
    "logdet",
    "lstsq",
    "lu_factor",
    "matrix_rank",
    "norm",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
]
