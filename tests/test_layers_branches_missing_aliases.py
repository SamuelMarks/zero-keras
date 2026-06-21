import numpy as np
import zero_keras.layers as layers


def test_conv_aliases_activation_none():
    for cls, in_shape in [
        (layers.Convolution1D, (2, 5, 3)),
        (layers.Convolution1DTranspose, (2, 5, 3)),
        (layers.Convolution2D, (2, 5, 5, 3)),
        (layers.Convolution2DTranspose, (2, 5, 5, 3)),
        (layers.Convolution3D, (2, 5, 5, 5, 3)),
        (layers.Convolution3DTranspose, (2, 5, 5, 5, 3)),
    ]:
        l = cls(2, 2, activation=None)
        l(np.random.rand(*in_shape))


def test_conv_aliases_preset_weights():
    # 1D
    l = layers.Convolution1D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 3))

    # 1DT
    l = layers.Convolution1DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 3))

    # 2D
    l = layers.Convolution2D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 3))

    # 2DT
    l = layers.Convolution2DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 3))

    # 3D
    l = layers.Convolution3D(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 3))

    # 3DT
    l = layers.Convolution3DTranspose(2, 2)
    l.kernel = 1
    l.bias = 1
    l.build((None, 5, 5, 5, 3))
