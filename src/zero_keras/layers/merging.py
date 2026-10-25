"""Merging layers (`Add`, `Subtract`, `Multiply`, `Average`, `Maximum`, `Minimum`, `Concatenate`, `Dot`) for zero_keras."""

from __future__ import annotations

from collections.abc import Sequence
from zero_keras import ops
from zero_keras.activations import _to_tensor, _wrap
from zero_keras.core_layers import Layer
from zero_keras.types import TensorLike


class Add(Layer):
    """Performs elementwise addition operation.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Add()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `added = keras.layers.add([x1, x2])`
    >>> added = keras.layers.Add()([x1, x2])
    >>> out = keras.layers.Dense(4)(added)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the elementwise sum of a sequence of input tensors.

        Args:
            inputs: Sequence of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise sum tensor.
        """
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.add(res, _to_tensor(t))
        return _wrap(res)


class Subtract(Layer):
    """Performs elementwise subtraction.

    It takes as input a list of tensors of size 2 both of the
    same shape, and returns a single tensor (inputs[0] - inputs[1])
    of same shape.

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Subtract()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `subtracted = keras.layers.subtract([x1, x2])`
    >>> subtracted = keras.layers.Subtract()([x1, x2])
    >>> out = keras.layers.Dense(4)(subtracted)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute ``inputs[0] - inputs[1]`` elementwise.

        Args:
            inputs: Pair of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise difference tensor.
        """
        return _wrap(ops.subtract(_to_tensor(inputs[0]), _to_tensor(inputs[1])))


class Multiply(Layer):
    """Performs elementwise multiplication.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Multiply()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.multiply([x1, x2])`
    >>> y = keras.layers.Multiply()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the elementwise product of a sequence of input tensors.

        Args:
            inputs: Sequence of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise product tensor.
        """
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.multiply(res, _to_tensor(t))
        return _wrap(res)


class Average(Layer):
    """Averages a list of inputs element-wise..

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Average()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.average([x1, x2])`
    >>> y = keras.layers.Average()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the elementwise mean of a sequence of input tensors.

        Args:
            inputs: Sequence of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise average tensor.
        """
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.add(res, _to_tensor(t))

        res = ops.divide(res, _to_tensor(float(len(inputs))))
        return _wrap(res)


class Maximum(Layer):
    """Computes element-wise maximum on a list of inputs.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Maximum()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.maximum([x1, x2])`
    >>> y = keras.layers.Maximum()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the elementwise maximum across a sequence of input tensors.

        Args:
            inputs: Sequence of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise maximum tensor.
        """
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.maximum(res, _to_tensor(t))
        return _wrap(res)


class Minimum(Layer):
    """Computes elementwise minimum on a list of inputs.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:
    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = keras.layers.Minimum()([x1, x2])

    Usage in a Keras model:

    >>> input1 = keras.layers.Input(shape=(16,))
    >>> x1 = keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = keras.layers.Input(shape=(32,))
    >>> x2 = keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = keras.layers.minimum([x1, x2])`
    >>> y = keras.layers.Minimum()([x1, x2])
    >>> out = keras.layers.Dense(4)(y)
    >>> model = keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the elementwise minimum across a sequence of input tensors.

        Args:
            inputs: Sequence of tensors with matching shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Elementwise minimum tensor.
        """
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.minimum(res, _to_tensor(t))
        return _wrap(res)


class Concatenate(Layer):
    """Concatenates a list of inputs.

    It takes as input a list of tensors, all of the same shape except
    for the concatenation axis, and returns a single tensor that is the
    concatenation of all inputs.

    Examples:
    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> keras.layers.Concatenate(axis=1)([x, y])

    Usage in a Keras model:

    >>> x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> y = keras.layers.Concatenate()([x1, x2])

    Args:
        axis: Axis along which to concatenate.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the concatenation of the inputs alongside axis `axis`.

    """

    def __init__(self, axis: int = -1, **kwargs: TensorLike) -> None:
        """Initialize the Concatenate layer.

        Args:
            axis: Axis along which to concatenate input tensors. Defaults to ``-1``.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Concatenate ``inputs`` along ``self.axis``.

        Args:
            inputs: Sequence of tensors to concatenate.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Concatenated tensor.
        """
        tensors = tuple(_to_tensor(t) for t in inputs)
        return _wrap(ops.concatenate(list(tensors), self.axis))


class Dot(Layer):
    """Computes element-wise dot product of two tensors.

    It takes a list of inputs of size 2, and the axes
    corresponding to each input along with the dot product
    is to be performed.

    Let's say `x` and `y` are the two input tensors with shapes
    `(2, 3, 5)` and `(2, 10, 3)`. The batch dimension should be
    of same size for both the inputs, and `axes` should correspond
    to the dimensions that have the same size in the corresponding
    inputs. e.g. with `axes=(1, 2)`, the dot product of `x`, and `y`
    will result in a tensor with shape `(2, 5, 10)`

    Example:
    >>> x = np.arange(10).reshape(1, 5, 2)
    >>> y = np.arange(10, 20).reshape(1, 2, 5)
    >>> keras.layers.Dot(axes=(1, 2))([x, y])

    Usage in a Keras model:

    >>> x1 = keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> y = keras.layers.Dot(axes=1)([x1, x2])

    Args:
        axes: Integer or tuple of integers, axis or axes along which to
            take the dot product. If a tuple, should be two integers
            corresponding to the desired axis from the first input and the
            desired axis from the second input, respectively. Note that the
            size of the two selected axes must match.
        normalize: Whether to L2-normalize samples along the dot product axis
            before taking the dot product. If set to `True`, then
            the output of the dot product is the cosine proximity
            between the two samples.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the dot product of the samples from the inputs.

    """

    def __init__(
        self, axes: int | Sequence[int], normalize: bool = False, **kwargs: TensorLike
    ) -> None:
        """Initialize the Dot product layer.

        Args:
            axes: Axis or pair of axes along which to compute the dot product.
            normalize: Whether to L2-normalize inputs before taking the dot product.
            **kwargs: Base layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axes = axes
        self.normalize = bool(normalize)

    def call(self, inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
        """Compute the dot product (or cosine similarity) of ``inputs[0]`` and ``inputs[1]``.

        Args:
            inputs: Pair of tensors ``[a, b]``.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorLike: Dot product tensor.
        """
        a, b = _to_tensor(inputs[0]), _to_tensor(inputs[1])
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes, axes)

        elif isinstance(axes, list):
            axes = tuple(axes)

        if self.normalize:
            a_norm = ops.sqrt(
                ops.sum(
                    ops.square(a),
                    axis=axes[0],
                    keepdims=True,
                )
            )
            b_norm = ops.sqrt(
                ops.sum(
                    ops.square(b),
                    axis=axes[1],
                    keepdims=True,
                )
            )
            a = ops.divide(a, ops.maximum(a_norm, 1e-7))
            b = ops.divide(b, ops.maximum(b_norm, 1e-7))

        import string

        einsum = ops.einsum
        chars = string.ascii_lowercase
        batch_char = chars[0]

        ndim1 = len(a.shape)
        ndim2 = len(b.shape)

        ax0 = axes[0] if axes[0] >= 0 else ndim1 + axes[0]
        ax1 = axes[1] if axes[1] >= 0 else ndim2 + axes[1]

        x1_chars = list(chars[0:ndim1])
        sum_char = x1_chars[ax0]

        fresh_start = ndim1
        x2_chars = []
        for i in range(ndim2):
            if i == 0:
                x2_chars.append(batch_char)
            elif i == ax1:
                x2_chars.append(sum_char)
            else:
                x2_chars.append(chars[fresh_start])
                fresh_start += 1

        out_chars = [c for c in x1_chars if c != sum_char] + [
            c for c in x2_chars if c != sum_char and c != batch_char
        ]

        eq = f"{''.join(x1_chars)},{''.join(x2_chars)}->{''.join(out_chars)}"
        out = einsum(eq, a, b)

        if len(out.shape) == 1:
            out = ops.expand_dims(out, -1)

        return _wrap(out)


def add(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Add`` layer.

    Args:
        inputs: Sequence of tensors to add elementwise.
        **kwargs: Keyword arguments forwarded to ``Add``.

    Returns:
        TensorLike: Elementwise sum tensor.
    """
    return Add(**kwargs)(inputs)


def subtract(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Subtract`` layer.

    Args:
        inputs: Pair of tensors ``[x1, x2]`` to subtract elementwise.
        **kwargs: Keyword arguments forwarded to ``Subtract``.

    Returns:
        TensorLike: Elementwise difference tensor ``x1 - x2``.
    """
    return Subtract(**kwargs)(inputs)


def multiply(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Multiply`` layer.

    Args:
        inputs: Sequence of tensors to multiply elementwise.
        **kwargs: Keyword arguments forwarded to ``Multiply``.

    Returns:
        TensorLike: Elementwise product tensor.
    """
    return Multiply(**kwargs)(inputs)


def average(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Average`` layer.

    Args:
        inputs: Sequence of tensors to average elementwise.
        **kwargs: Keyword arguments forwarded to ``Average``.

    Returns:
        TensorLike: Elementwise average tensor.
    """
    return Average(**kwargs)(inputs)


def maximum(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Maximum`` layer.

    Args:
        inputs: Sequence of tensors to take the elementwise maximum over.
        **kwargs: Keyword arguments forwarded to ``Maximum``.

    Returns:
        TensorLike: Elementwise maximum tensor.
    """
    return Maximum(**kwargs)(inputs)


def minimum(inputs: Sequence[TensorLike], **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Minimum`` layer.

    Args:
        inputs: Sequence of tensors to take the elementwise minimum over.
        **kwargs: Keyword arguments forwarded to ``Minimum``.

    Returns:
        TensorLike: Elementwise minimum tensor.
    """
    return Minimum(**kwargs)(inputs)


def concatenate(inputs: Sequence[TensorLike], axis: int = -1, **kwargs: TensorLike) -> TensorLike:
    """Functional interface to the ``Concatenate`` layer.

    Args:
        inputs: Sequence of tensors to concatenate.
        axis: Concatenation axis. Defaults to ``-1``.
        **kwargs: Keyword arguments forwarded to ``Concatenate``.

    Returns:
        TensorLike: Concatenated tensor.
    """
    return Concatenate(axis=axis, **kwargs)(inputs)


def dot(
    inputs: Sequence[TensorLike],
    axes: int | Sequence[int],
    normalize: bool = False,
    **kwargs: TensorLike,
) -> TensorLike:
    """Functional interface to the ``Dot`` layer.

    Args:
        inputs: Pair of tensors ``[x1, x2]``.
        axes: Axis or pair of axes along which to compute the dot product.
        normalize: Whether to L2-normalize inputs before taking the dot product.
        **kwargs: Keyword arguments forwarded to ``Dot``.

    Returns:
        TensorLike: Dot product tensor.
    """
    return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)
