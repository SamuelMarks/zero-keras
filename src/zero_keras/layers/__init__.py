# ruff: noqa
"""Keras layers."""

from zero_keras.core_layers import Layer, Wrapper
from .core import *
from .normalization import *
from .merging import *
from .activations import *
from .convolution import *
from .pooling import *
from .regularization import *
from .attention import *
from .rnn import *
from .preprocessing import *
from .misc import *

from zero_keras.core_layers import Input as Input

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
Convolution1DTranspose = Conv1DTranspose
Convolution2DTranspose = Conv2DTranspose
Convolution3DTranspose = Conv3DTranspose

AvgPool1D = AveragePooling1D
AvgPool2D = AveragePooling2D
AvgPool3D = AveragePooling3D
GlobalAvgPool1D = GlobalAveragePooling1D
GlobalAvgPool2D = GlobalAveragePooling2D
GlobalAvgPool3D = GlobalAveragePooling3D
GlobalMaxPool1D = GlobalMaxPooling1D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool3D = GlobalMaxPooling3D
MaxPool1D = MaxPooling1D
MaxPool2D = MaxPooling2D
MaxPool3D = MaxPooling3D
SeparableConvolution1D = SeparableConv1D
SeparableConvolution2D = SeparableConv2D
