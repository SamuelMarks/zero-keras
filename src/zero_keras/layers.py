"""Keras layers module."""

from typing import Any, Optional
from zero_keras.core_layers import Layer

__all__ = [
    "Layer",
    "Activation",
    "ActivityRegularization",
    "Add",
    "AdditiveAttention",
    "AlphaDropout",
    "Attention",
    "AugMix",
    "AutoContrast",
    "Average",
    "AveragePooling1D",
    "AveragePooling2D",
    "AveragePooling3D",
    "AvgPool1D",
    "AvgPool2D",
    "AvgPool3D",
    "BatchNormalization",
    "Bidirectional",
    "CategoryEncoding",
    "CenterCrop",
    "Concatenate",
    "Conv1D",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DTranspose",
    "ConvLSTM1D",
    "ConvLSTM2D",
    "ConvLSTM3D",
    "Convolution1D",
    "Convolution1DTranspose",
    "Convolution2D",
    "Convolution2DTranspose",
    "Convolution3D",
    "Convolution3DTranspose",
    "Cropping1D",
    "Cropping2D",
    "Cropping3D",
    "CutMix",
    "Dense",
    "DepthwiseConv1D",
    "DepthwiseConv2D",
    "Discretization",
    "Dot",
    "Dropout",
    "ELU",
    "EinsumDense",
    "Embedding",
    "Equalization",
    "Flatten",
    "FlaxLayer",
    "GRU",
    "GRUCell",
    "GaussianDropout",
    "GaussianNoise",
    "GlobalAveragePooling1D",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling3D",
    "GlobalAvgPool1D",
    "GlobalAvgPool2D",
    "GlobalAvgPool3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "GlobalMaxPooling1D",
    "GlobalMaxPooling2D",
    "GlobalMaxPooling3D",
    "GroupNormalization",
    "GroupQueryAttention",
    "HashedCrossing",
    "Hashing",
    "Identity",
    "InputLayer",
    "InputSpec",
    "IntegerLookup",
    "JaxLayer",
    "LSTM",
    "LSTMCell",
    "Lambda",
    "LayerNormalization",
    "LeakyReLU",
    "Masking",
    "MaxNumBoundingBoxes",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "MaxPooling1D",
    "MaxPooling2D",
    "MaxPooling3D",
    "Maximum",
    "MelSpectrogram",
    "Minimum",
    "MixUp",
    "MultiHeadAttention",
    "Multiply",
    "Normalization",
    "PReLU",
    "Permute",
    "Pipeline",
    "RMSNormalization",
    "RNN",
    "RandAugment",
    "RandomBrightness",
    "RandomColorDegeneration",
    "RandomColorJitter",
    "RandomContrast",
    "RandomCrop",
    "RandomElasticTransform",
    "RandomErasing",
    "RandomFlip",
    "RandomGaussianBlur",
    "RandomGrayscale",
    "RandomHue",
    "RandomInvert",
    "RandomPerspective",
    "RandomPosterization",
    "RandomRotation",
    "RandomSaturation",
    "RandomSharpness",
    "RandomShear",
    "RandomTranslation",
    "RandomZoom",
    "ReLU",
    "RepeatVector",
    "Rescaling",
    "Reshape",
    "Resizing",
    "STFTSpectrogram",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConvolution1D",
    "SeparableConvolution2D",
    "SimpleRNN",
    "SimpleRNNCell",
    "Softmax",
    "Solarization",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "SpectralNormalization",
    "StackedRNNCells",
    "StringLookup",
    "Subtract",
    "TFSMLayer",
    "TextVectorization",
    "TimeDistributed",
    "TorchModuleWrapper",
    "UnitNormalization",
    "UpSampling1D",
    "UpSampling2D",
    "UpSampling3D",
    "Wrapper",
    "ZeroPadding1D",
    "ZeroPadding2D",
    "ZeroPadding3D",
]


class Dense(Layer):
    """docstring"""

    def __init__(
        self,
        units: int,
        activation: Optional[Any] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        activity_regularizer: Optional[Any] = None,
        kernel_constraint: Optional[Any] = None,
        bias_constraint: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.units = units


class Activation(Layer):
    pass


class ActivityRegularization(Layer):
    pass


class Add(Layer):
    pass


class AdditiveAttention(Layer):
    pass


class AlphaDropout(Layer):
    pass


class Attention(Layer):
    pass


class AugMix(Layer):
    pass


class AutoContrast(Layer):
    pass


class Average(Layer):
    pass


class AveragePooling1D(Layer):
    pass


class AveragePooling2D(Layer):
    pass


class AveragePooling3D(Layer):
    pass


class AvgPool1D(Layer):
    pass


class AvgPool2D(Layer):
    pass


class AvgPool3D(Layer):
    pass


class BatchNormalization(Layer):
    pass


class Bidirectional(Layer):
    pass


class CategoryEncoding(Layer):
    pass


class CenterCrop(Layer):
    pass


class Concatenate(Layer):
    pass


class Conv1D(Layer):
    pass


class Conv1DTranspose(Layer):
    pass


class Conv2D(Layer):
    pass


class Conv2DTranspose(Layer):
    pass


class Conv3D(Layer):
    pass


class Conv3DTranspose(Layer):
    pass


class ConvLSTM1D(Layer):
    pass


class ConvLSTM2D(Layer):
    pass


class ConvLSTM3D(Layer):
    pass


class Convolution1D(Layer):
    pass


class Convolution1DTranspose(Layer):
    pass


class Convolution2D(Layer):
    pass


class Convolution2DTranspose(Layer):
    pass


class Convolution3D(Layer):
    pass


class Convolution3DTranspose(Layer):
    pass


class Cropping1D(Layer):
    pass


class Cropping2D(Layer):
    pass


class Cropping3D(Layer):
    pass


class CutMix(Layer):
    pass


class DepthwiseConv1D(Layer):
    pass


class DepthwiseConv2D(Layer):
    pass


class Discretization(Layer):
    pass


class Dot(Layer):
    pass


class Dropout(Layer):
    pass


class ELU(Layer):
    pass


class EinsumDense(Layer):
    pass


class Embedding(Layer):
    pass


class Equalization(Layer):
    pass


class Flatten(Layer):
    pass


class FlaxLayer(Layer):
    pass


class GRU(Layer):
    pass


class GRUCell(Layer):
    pass


class GaussianDropout(Layer):
    pass


class GaussianNoise(Layer):
    pass


class GlobalAveragePooling1D(Layer):
    pass


class GlobalAveragePooling2D(Layer):
    pass


class GlobalAveragePooling3D(Layer):
    pass


class GlobalAvgPool1D(Layer):
    pass


class GlobalAvgPool2D(Layer):
    pass


class GlobalAvgPool3D(Layer):
    pass


class GlobalMaxPool1D(Layer):
    pass


class GlobalMaxPool2D(Layer):
    pass


class GlobalMaxPool3D(Layer):
    pass


class GlobalMaxPooling1D(Layer):
    pass


class GlobalMaxPooling2D(Layer):
    pass


class GlobalMaxPooling3D(Layer):
    pass


class GroupNormalization(Layer):
    pass


class GroupQueryAttention(Layer):
    pass


class HashedCrossing(Layer):
    pass


class Hashing(Layer):
    pass


class Identity(Layer):
    pass


class InputLayer(Layer):
    pass


class InputSpec(Layer):
    pass


class IntegerLookup(Layer):
    pass


class JaxLayer(Layer):
    pass


class LSTM(Layer):
    pass


class LSTMCell(Layer):
    pass


class Lambda(Layer):
    pass


class LayerNormalization(Layer):
    pass


class LeakyReLU(Layer):
    pass


class Masking(Layer):
    pass


class MaxNumBoundingBoxes(Layer):
    pass


class MaxPool1D(Layer):
    pass


class MaxPool2D(Layer):
    pass


class MaxPool3D(Layer):
    pass


class MaxPooling1D(Layer):
    pass


class MaxPooling2D(Layer):
    pass


class MaxPooling3D(Layer):
    pass


class Maximum(Layer):
    pass


class MelSpectrogram(Layer):
    pass


class Minimum(Layer):
    pass


class MixUp(Layer):
    pass


class MultiHeadAttention(Layer):
    pass


class Multiply(Layer):
    pass


class Normalization(Layer):
    pass


class PReLU(Layer):
    pass


class Permute(Layer):
    pass


class Pipeline(Layer):
    pass


class RMSNormalization(Layer):
    pass


class RNN(Layer):
    pass


class RandAugment(Layer):
    pass


class RandomBrightness(Layer):
    pass


class RandomColorDegeneration(Layer):
    pass


class RandomColorJitter(Layer):
    pass


class RandomContrast(Layer):
    pass


class RandomCrop(Layer):
    pass


class RandomElasticTransform(Layer):
    pass


class RandomErasing(Layer):
    pass


class RandomFlip(Layer):
    pass


class RandomGaussianBlur(Layer):
    pass


class RandomGrayscale(Layer):
    pass


class RandomHue(Layer):
    pass


class RandomInvert(Layer):
    pass


class RandomPerspective(Layer):
    pass


class RandomPosterization(Layer):
    pass


class RandomRotation(Layer):
    pass


class RandomSaturation(Layer):
    pass


class RandomSharpness(Layer):
    pass


class RandomShear(Layer):
    pass


class RandomTranslation(Layer):
    pass


class RandomZoom(Layer):
    pass


class ReLU(Layer):
    pass


class RepeatVector(Layer):
    pass


class Rescaling(Layer):
    pass


class Reshape(Layer):
    pass


class Resizing(Layer):
    pass


class STFTSpectrogram(Layer):
    pass


class SeparableConv1D(Layer):
    pass


class SeparableConv2D(Layer):
    pass


class SeparableConvolution1D(Layer):
    pass


class SeparableConvolution2D(Layer):
    pass


class SimpleRNN(Layer):
    pass


class SimpleRNNCell(Layer):
    pass


class Softmax(Layer):
    pass


class Solarization(Layer):
    pass


class SpatialDropout1D(Layer):
    pass


class SpatialDropout2D(Layer):
    pass


class SpatialDropout3D(Layer):
    pass


class SpectralNormalization(Layer):
    pass


class StackedRNNCells(Layer):
    pass


class StringLookup(Layer):
    pass


class Subtract(Layer):
    pass


class TFSMLayer(Layer):
    pass


class TextVectorization(Layer):
    pass


class TimeDistributed(Layer):
    pass


class TorchModuleWrapper(Layer):
    pass


class UnitNormalization(Layer):
    pass


class UpSampling1D(Layer):
    pass


class UpSampling2D(Layer):
    pass


class UpSampling3D(Layer):
    pass


class Wrapper(Layer):
    pass


class ZeroPadding1D(Layer):
    pass


class ZeroPadding2D(Layer):
    pass


class ZeroPadding3D(Layer):
    pass
