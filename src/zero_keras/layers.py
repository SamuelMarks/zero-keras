"""Keras layers."""

from zero_keras.core_layers import Layer as BaseLayer
import ml_switcheroo.nn.layers as layers_impl
from .activations import _wrap


class Layer(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._impl = (
            getattr(layers_impl, self.__class__.__name__)(**kwargs)
            if self.__class__.__name__ != "Layer"
            else layers_impl.Layer(**kwargs)
        )

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        return _wrap(self._impl(*args, **kwargs))


class Activation(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ActivityRegularization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Add(Layer):
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


class Average(Layer):
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


class BatchNormalization(Layer):
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


class Concatenate(Layer):
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


class Dense(Layer):
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


class Dot(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Dropout(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class ELU(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


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


class Flatten(Layer):
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


class Lambda(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class LayerNormalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class LeakyReLU(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Masking(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


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


class Maximum(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class MelSpectrogram(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Minimum(Layer):
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


class Multiply(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Normalization(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class PReLU(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Permute(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


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
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class RepeatVector(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Rescaling(Layer):
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


class Reshape(Layer):
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
    def __init__(self, *args, **kwargs):
        self.units = kwargs.get("units", args[0] if args else None)
        super().__init__(**kwargs)


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


class Subtract(Layer):
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
