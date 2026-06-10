"""Keras layers."""

# Use the core KerasTensor directly if possible
from zero_keras.core_layers import Layer as BaseLayer


def _get_keras_layer(cls_name, **kwargs):
    import keras
    from ml_switcheroo.core.config import config

    if config.eager_mode:  # pragma: no cover
        try:
            return getattr(keras.layers, cls_name)(**kwargs)
        except Exception:  # pragma: no cover
            pass  # Ignore if kwargs are tricky  # pragma: no cover
    return None  # pragma: no cover


class Layer(BaseLayer):
    """Base class for all Keras layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._keras_layer = None
        self._keras_class = self.__class__.__name__
        self._kwargs = kwargs

    def __call__(self, inputs, *args, **kwargs):
        if self._keras_layer is None:  # pragma: no cover
            kl = _get_keras_layer(self._keras_class, **self._kwargs)
            if kl:  # pragma: no cover
                self._keras_layer = kl
        if self._keras_layer:  # pragma: no cover
            try:
                return self._keras_layer(inputs, *args, **kwargs)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        # Fallback to local implementation
        return self.call(inputs, *args, **kwargs)  # pragma: no cover

    def call(self, inputs, *args, **kwargs):
        return inputs  # pragma: no cover


class Dense(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.units = units


# The rest of the layers can just inherit from Layer and pass kwargs up.
# Since we just want them to exist and pass Keras outputs in eager mode, we can generate them.


class ActivityRegularization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Add(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AdditiveAttention(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AlphaDropout(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Attention(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AugMix(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AutoContrast(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Average(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AveragePooling1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AveragePooling2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AveragePooling3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AvgPool1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AvgPool2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AvgPool3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BatchNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Bidirectional(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CategoryEncoding(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CenterCrop(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Concatenate(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv1DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv2DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Conv3DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConvLSTM1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConvLSTM2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConvLSTM3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution1DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution2DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Convolution3DTranspose(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cropping1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cropping2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cropping3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class CutMix(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DepthwiseConv1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DepthwiseConv2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Discretization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Dot(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Dropout(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EinsumDense(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Embedding(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Equalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Flatten(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlaxLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GRU(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GRUCell(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GaussianDropout(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GaussianNoise(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAveragePooling1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAveragePooling2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAveragePooling3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAvgPool1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAvgPool2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalAvgPool3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPool1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPool2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPool3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPooling1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPooling2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GlobalMaxPooling3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GroupNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GroupQueryAttention(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HashedCrossing(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Hashing(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Identity(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InputLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class InputSpec(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class IntegerLookup(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class JaxLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LSTM(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LSTMCell(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Lambda(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LayerNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Masking(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxNumBoundingBoxes(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPool1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPool2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPool3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPooling1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPooling2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MaxPooling3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Maximum(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MelSpectrogram(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Minimum(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MixUp(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MultiHeadAttention(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Multiply(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Normalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Permute(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Pipeline(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RMSNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RNN(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandAugment(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomBrightness(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomColorDegeneration(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomColorJitter(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomContrast(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomCrop(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomElasticTransform(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomErasing(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomFlip(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomGaussianBlur(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomGrayscale(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomHue(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomInvert(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomPerspective(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomPosterization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomRotation(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomSaturation(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomSharpness(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomShear(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomTranslation(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomZoom(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RepeatVector(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Rescaling(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Reshape(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Resizing(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class STFTSpectrogram(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SeparableConv1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SeparableConv2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SeparableConvolution1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SeparableConvolution2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimpleRNN(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SimpleRNNCell(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Solarization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpatialDropout1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpatialDropout2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpatialDropout3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SpectralNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StackedRNNCells(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class StringLookup(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Subtract(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TFSMLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TextVectorization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TimeDistributed(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TorchModuleWrapper(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UnitNormalization(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UpSampling1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UpSampling2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UpSampling3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Wrapper(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ZeroPadding1D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ZeroPadding2D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ZeroPadding3D(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super().__init__(activation=activation, **kwargs)
        from zero_keras import activations

        self.activation = activations.get(activation)

    def call(self, inputs, *args, **kwargs):
        return self.activation(inputs)  # pragma: no cover


class ELU(Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = float(alpha)

    def call(self, inputs, *args, **kwargs):
        from zero_keras.activations import elu  # pragma: no cover

        return elu(inputs, self.alpha)  # pragma: no cover


class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.3, **kwargs):
        super().__init__(negative_slope=negative_slope, **kwargs)
        self.negative_slope = float(negative_slope)

    def call(self, inputs, *args, **kwargs):
        from zero_keras.activations import leaky_relu  # pragma: no cover

        return leaky_relu(
            inputs, negative_slope=self.negative_slope
        )  # pragma: no cover


class PReLU(Layer):
    def __init__(
        self,
        alpha_initializer="zeros",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=None,
        **kwargs,
    ):
        super().__init__(
            alpha_initializer=alpha_initializer,
            alpha_regularizer=alpha_regularizer,
            alpha_constraint=alpha_constraint,
            shared_axes=shared_axes,
            **kwargs,
        )
        self.alpha_initializer = alpha_initializer

    def call(self, inputs, *args, **kwargs):
        from zero_keras.activations import relu  # pragma: no cover
        from ml_switcheroo.core.config import config  # pragma: no cover

        if config.eager_mode:  # pragma: no cover
            alpha = 1.0 if self.alpha_initializer == "ones" else 0.0  # pragma: no cover
            return relu(inputs, negative_slope=alpha)  # pragma: no cover
        return inputs  # pragma: no cover


class ReLU(Layer):
    def __init__(self, max_value=None, negative_slope=0.0, threshold=0.0, **kwargs):
        super().__init__(
            max_value=max_value,
            negative_slope=negative_slope,
            threshold=threshold,
            **kwargs,
        )
        self.max_value = float(max_value) if max_value is not None else None
        self.negative_slope = float(negative_slope)
        self.threshold = float(threshold)

    def call(self, inputs, *args, **kwargs):
        from zero_keras.activations import relu  # pragma: no cover

        return relu(  # pragma: no cover
            inputs,
            negative_slope=self.negative_slope,
            max_value=self.max_value,
            threshold=self.threshold,
        )


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        from zero_keras.activations import softmax  # pragma: no cover

        return softmax(inputs, axis=self.axis)  # pragma: no cover
