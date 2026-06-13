"""Keras layers."""

from zero_keras.core_layers import Layer as BaseLayer

from .activations import _wrap


import ml_switcheroo_compiler.ops as ops
from zero_keras.activations import _to_tensor, get as get_activation


class Layer(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if not self.built:
            inputs = args[0] if args else kwargs.get("inputs")
            self.build(getattr(inputs, "shape", None))
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        return args[0] if args else None


class Dense(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        print("BUILD CALLED")
        if self.built:
            return
        import ml_switcheroo_compiler.core.dtype as dtypes

        in_dim = input_shape[-1]
        limit = (6.0 / (in_dim + self.units)) ** 0.5

        import ml_switcheroo_compiler.random as random

        key = random.PRNGKey(self._kwargs.get("seed", 42))
        self.kernel = random.uniform(
            key, (in_dim, self.units), dtypes.DType.Float32, -limit, limit
        )

        if self.use_bias:
            self.bias = ops.zeros((self.units,), dtype=dtypes.DType.Float32)
        self.built = True

    def get_weights(self):

        w = [self.kernel.numpy() if hasattr(self.kernel, "numpy") else self.kernel.data]
        if self.use_bias:
            w.append(
                self.bias.numpy() if hasattr(self.bias, "numpy") else self.bias.data
            )
        return w

    def set_weights(self, weights):
        import ml_switcheroo_compiler.core.dtype as dtypes

        self.kernel = ops.asarray(weights[0], dtype=dtypes.DType.Float32)
        if self.use_bias:
            self.bias = ops.asarray(weights[1], dtype=dtypes.DType.Float32)

    def call(self, inputs, *args, **kwargs):
        inputs = _to_tensor(inputs)
        if not self.built:
            self.build(inputs.shape)

        out = ops.matmul(inputs, self.kernel)
        if self.use_bias:
            out = ops.add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)
        return _wrap(out)


class Dropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None, **kwargs):
        inputs = _to_tensor(inputs)
        if training or training is None:
            import ml_switcheroo_compiler.random as random

            key = random.PRNGKey(42)
            mask_t = ops.cast(
                random.bernoulli(key, 1.0 - self.rate, inputs.shape), dtype=inputs.dtype
            ) / (1.0 - self.rate)
            return _wrap(ops.multiply(inputs, mask_t))
        return _wrap(inputs)


class Flatten(Layer):
    def call(self, inputs, **kwargs):
        inputs = _to_tensor(inputs)
        shape = inputs.shape
        if len(shape) <= 1:
            return _wrap(inputs)

        new_dim = 1
        for d in shape[1:]:
            new_dim *= d
        return _wrap(ops.reshape(inputs, (shape[0], int(new_dim))))


class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs, **kwargs):
        inputs = _to_tensor(inputs)
        shape = (inputs.shape[0],) + self.target_shape
        return _wrap(ops.reshape(inputs, shape))


class Permute(Layer):
    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)
        self.dims = tuple(dims)

    def call(self, inputs, **kwargs):
        inputs = _to_tensor(inputs)
        perm = [0] + [d for d in self.dims]
        return _wrap(ops.permute(inputs, perm))


class RepeatVector(Layer):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, inputs, **kwargs):
        inputs = _to_tensor(inputs)
        expanded = ops.expand_dims(inputs, 1)
        repeats = [1, self.n, 1]
        return _wrap(ops.tile(expanded, repeats))


class Masking(Layer):
    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        return _wrap(_to_tensor(inputs))


class Lambda(Layer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function

    def call(self, inputs, **kwargs):
        return self.function(inputs, **kwargs)


class LayerNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs, **kwargs):
        inputs = _to_tensor(inputs)
        mean = ops.mean(inputs, axis=self.axis, keepdims=True)
        var = ops.var(inputs, axis=self.axis, keepdims=True)
        return _wrap((inputs - mean) / ops.sqrt(var + self.epsilon))


class BatchNormalization(Layer):
    def __init__(self, axis=-1, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs, training=None, **kwargs):
        inputs = _to_tensor(inputs)
        if training:
            rank = len(inputs.shape)
            axis = self.axis if self.axis >= 0 else self.axis + rank
            axes = tuple(i for i in range(rank) if i != axis)
            mean = ops.mean(inputs, axis=axes, keepdims=True)
            var = ops.var(inputs, axis=axes, keepdims=True)
            return _wrap((inputs - mean) / ops.sqrt(var + self.epsilon))
        else:
            return _wrap(inputs / ops.sqrt(_to_tensor(1.0 + self.epsilon)))


class Add(Layer):
    def call(self, inputs, **kwargs):
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.add(res, _to_tensor(t))
        return _wrap(res)


class Subtract(Layer):
    def call(self, inputs, **kwargs):
        return _wrap(ops.subtract(_to_tensor(inputs[0]), _to_tensor(inputs[1])))


class Multiply(Layer):
    def call(self, inputs, **kwargs):
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.multiply(res, _to_tensor(t))
        return _wrap(res)


class Average(Layer):
    def call(self, inputs, **kwargs):
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.add(res, _to_tensor(t))

        res = ops.divide(res, _to_tensor(float(len(inputs))))
        return _wrap(res)


class Maximum(Layer):
    def call(self, inputs, **kwargs):
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.maximum(res, _to_tensor(t))
        return _wrap(res)


class Minimum(Layer):
    def call(self, inputs, **kwargs):
        res = _to_tensor(inputs[0])
        for t in inputs[1:]:
            res = ops.minimum(res, _to_tensor(t))
        return _wrap(res)


class Concatenate(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        tensors = tuple(_to_tensor(t) for t in inputs)
        return _wrap(ops.concatenate(list(tensors), self.axis))


class Dot(Layer):
    def __init__(self, axes, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.axes = axes
        self.normalize = normalize

    def call(self, inputs, **kwargs):
        a, b = _to_tensor(inputs[0]), _to_tensor(inputs[1])
        if self.normalize:
            a_norm = ops.sqrt(
                ops.sum(
                    ops.square(a),
                    axis=self.axes if isinstance(self.axes, int) else self.axes[0],
                    keepdims=True,
                )
            )
            b_norm = ops.sqrt(
                ops.sum(
                    ops.square(b),
                    axis=self.axes if isinstance(self.axes, int) else self.axes[1],
                    keepdims=True,
                )
            )
            a = ops.divide(a, ops.maximum(a_norm, 1e-7))
            b = ops.divide(b, ops.maximum(b_norm, 1e-7))

        if isinstance(self.axes, int) and self.axes == 1:
            out = ops.sum(ops.multiply(a, b), axis=1)
            out = ops.expand_dims(out, -1)
            return _wrap(out)
        return _wrap(ops.multiply(a, b))


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(**kwargs)
        from zero_keras.activations import get

        self.activation = get(activation)

    def call(self, inputs, **kwargs):
        return self.activation(inputs)


class ActivityRegularization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AdditiveAttention(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AlphaDropout(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Attention(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AugMix(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AutoContrast(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AveragePooling1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AveragePooling2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AveragePooling3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AvgPool1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AvgPool2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class AvgPool3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Bidirectional(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class CategoryEncoding(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class CenterCrop(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv1DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv2DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Conv3DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ConvLSTM1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ConvLSTM2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ConvLSTM3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution1DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution2DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Convolution3DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Cropping1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Cropping2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Cropping3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class CutMix(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class DepthwiseConv1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class DepthwiseConv2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Discretization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ELU(Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, **kwargs):
        from zero_keras import activations

        return activations.elu(inputs, alpha=self.alpha)


class EinsumDense(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Embedding(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Equalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class FlaxLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GRU(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GRUCell(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GaussianDropout(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GaussianNoise(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAveragePooling1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAveragePooling2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAveragePooling3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAvgPool1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAvgPool2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalAvgPool3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPool1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPool2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPool3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPooling1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPooling2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GlobalMaxPooling3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GroupNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class GroupQueryAttention(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class HashedCrossing(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Hashing(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Identity(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class InputLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class InputSpec(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class IntegerLookup(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class JaxLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class LSTM(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class LSTMCell(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.3, **kwargs):
        super().__init__(**kwargs)
        self.negative_slope = negative_slope

    def call(self, inputs, **kwargs):
        from zero_keras import activations

        return activations.leaky_relu(inputs, negative_slope=self.negative_slope)


class MaxNumBoundingBoxes(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPool1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPool2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPool3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPooling1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPooling2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MaxPooling3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MelSpectrogram(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MixUp(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MultiHeadAttention(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Normalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class PReLU(Layer):
    def __init__(
        self,
        alpha_initializer="zeros",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha_initializer = alpha_initializer
        from zero_keras.initializers import get

        self._alpha_init_fn = get(alpha_initializer)

    def build(self, input_shape):
        print("BUILD CALLED")
        if input_shape is not None:
            shape = input_shape[1:]
        else:
            shape = ()
        # Actually in zero_keras we might not have variables, just return a tensor
        self.alpha = self._alpha_init_fn(shape=shape)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        from ml_switcheroo_compiler import ops as backend_ops
        from zero_keras.activations import _to_tensor, _wrap

        inputs_t = _to_tensor(inputs)
        alpha_t = getattr(self.alpha, "data", None) if hasattr(self, "alpha") else None
        if alpha_t is None:
            alpha_t = 0.0
        alpha_t = _to_tensor(alpha_t)
        zero_t = backend_ops.asarray(0.0, dtype=getattr(inputs_t, "dtype", "float32"))
        print("alpha_t:", alpha_t.numpy() if hasattr(alpha_t, "numpy") else alpha_t)
        pos = backend_ops.maximum(inputs_t, zero_t)
        min_t = backend_ops.minimum(inputs_t, zero_t)
        neg = backend_ops.multiply(alpha_t, min_t)
        print("min_t:", min_t)
        print("neg:", neg)
        return _wrap(backend_ops.add(pos, neg))


class Pipeline(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RMSNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RNN(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandAugment(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomBrightness(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomColorDegeneration(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomColorJitter(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomContrast(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomCrop(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomElasticTransform(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomErasing(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomFlip(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomGaussianBlur(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomGrayscale(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomHue(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomInvert(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomPerspective(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomPosterization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomRotation(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomSaturation(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomSharpness(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomShear(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomTranslation(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RandomZoom(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ReLU(Layer):
    def __init__(self, max_value=None, negative_slope=0.0, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold

    def call(self, inputs, **kwargs):
        from zero_keras import activations

        return activations.relu(
            inputs,
            max_value=self.max_value,
            negative_slope=self.negative_slope,
            threshold=self.threshold,
        )


class Rescaling(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Resizing(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class STFTSpectrogram(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SeparableConv1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SeparableConv2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SeparableConvolution1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SeparableConvolution2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SimpleRNN(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SimpleRNNCell(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        from zero_keras import activations

        return activations.softmax(inputs, axis=self.axis)


class Solarization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SpatialDropout1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SpatialDropout2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SpatialDropout3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class SpectralNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class StackedRNNCells(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class StringLookup(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class TFSMLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class TextVectorization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class TimeDistributed(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class TorchModuleWrapper(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class UnitNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class UpSampling1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class UpSampling2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class UpSampling3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Wrapper(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ZeroPadding1D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ZeroPadding2D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ZeroPadding3D(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)
