from unittest.mock import patch
from zero_keras import layers
from ml_switcheroo_compiler.ops import asarray


def test_convolution_aliases_missing_branches():
    # test pre-set weights, use_bias=False, activation=None for aliases
    with (
        patch("zero_keras.layers.ops") as mock_ops,
        patch(
            "ml_switcheroo_compiler.ops.linalg.conv.conv_general_dilated"
        ) as mock_cgd,
        patch("ml_switcheroo_compiler.ops.linalg.conv_general_dilated") as mock_cgd2,
    ):
        mock_ops.conv1d.return_value = asarray([[1.0]])
        mock_ops.conv2d.return_value = asarray([[1.0]])
        mock_ops.conv3d.return_value = asarray([[1.0]])
        mock_ops.conv1d_transpose.return_value = asarray([[1.0]])
        mock_ops.conv2d_transpose.return_value = asarray([[1.0]])
        mock_ops.conv3d_transpose.return_value = asarray([[1.0]])
        mock_cgd.return_value = asarray([[1.0]])
        mock_cgd2.return_value = asarray([[1.0]])
        mock_cgd2.return_value = asarray([[1.0]])

        for layer_cls in [
            layers.Convolution1D,
            layers.Convolution2D,
            layers.Convolution3D,
            layers.Convolution1DTranspose,
            layers.Convolution2DTranspose,
            layers.Convolution3DTranspose,
        ]:
            # First hit pre-set weights and use_bias=True
            kwargs = {"use_bias": True, "filters": 4, "kernel_size": 2}
            layer = layer_cls(**kwargs)
            layer.kernel = asarray([1.0])
            layer.bias = asarray([1.0])
            input_shape = (2, 4, 4, 4, 4)
            if "1D" in layer_cls.__name__:
                input_shape = (2, 4, 4)
            elif "2D" in layer_cls.__name__:
                input_shape = (2, 4, 4, 4)
            layer.build(input_shape)

            # Then hit use_bias=False and activation=None
            kwargs = {
                "use_bias": False,
                "activation": None,
                "filters": 4,
                "kernel_size": 2,
            }
            layer2 = layer_cls(**kwargs)
            layer2.built = True
            layer2.kernel = asarray([1.0])

            if "1D" in layer_cls.__name__:
                layer2(asarray([[[1.0] * 4] * 4] * 2))
            elif "2D" in layer_cls.__name__:
                layer2(asarray([[[[1.0] * 4] * 4] * 4] * 2))
            elif "3D" in layer_cls.__name__:
                layer2(asarray([[[[[1.0] * 4] * 4] * 4] * 4] * 2))


def test_activation_none_dense_conv():
    # specifically trigger `activation is not None` evaluates to False in Dense/Conv/EinsumDense
    with patch("zero_keras.layers.ops") as mock_ops:
        mock_ops.matmul.return_value = asarray([[1.0]])
        mock_ops.add.return_value = asarray([[1.0]])

        for layer_cls in [layers.Dense]:
            layer = layer_cls(units=4, activation=None)
            layer.built = True
            layer.kernel = asarray([[1.0]])
            layer.bias = asarray([1.0])
            layer(asarray([[1.0]]))

    with patch("zero_keras.layers.ops") as mock_ops:
        mock_ops.einsum.return_value = asarray([[1.0]])
        mock_ops.add.return_value = asarray([[1.0]])
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(4,), activation=None)
        layer.built = True
        layer.kernel = asarray([[1.0]])
        layer.bias = asarray([1.0])
        layer(asarray([[1.0]]))

        # also EinsumDense build with preset kernel and bias
        layer = layers.EinsumDense("ab,bc->ac", output_shape=(4,), bias_axes="c")
        layer.kernel = asarray([[1.0]])
        layer.bias = asarray([1.0])
        layer.build((2, 4))


def test_rnn_cells_missing_branches():
    with patch("zero_keras.layers.ops") as mock_ops:
        mock_ops.dot.return_value = asarray([[1.0]])
        mock_ops.add.return_value = asarray([[1.0]])
        mock_ops.split.return_value = [asarray([[1.0]])] * 4
        mock_ops.multiply.return_value = asarray([[1.0]])
        mock_ops.gru_cell.return_value = (asarray([[1.0]]), asarray([[1.0]]))
        mock_ops.lstm_cell.return_value = (
            asarray([[1.0]]),
            [asarray([[1.0]]), asarray([[1.0]])],
        )
        mock_ops.rnn_cell.return_value = (asarray([[1.0]]), asarray([[1.0]]))

        for layer_cls in [layers.SimpleRNNCell, layers.GRUCell, layers.LSTMCell]:
            # preset weights + use_bias=True
            kwargs = {"units": 4, "use_bias": True}
            layer = layer_cls(**kwargs)
            layer.kernel = asarray([1.0])
            layer.recurrent_kernel = asarray([1.0])
            layer.bias = asarray([1.0])
            layer.build((2, 4))

            # use_bias=False + activation=None
            kwargs = {"units": 4, "use_bias": False, "activation": None}
            layer2 = layer_cls(**kwargs)
            layer2.built = True
            layer2.kernel = asarray([1.0])
            layer2.recurrent_kernel = asarray([1.0])
            states = (
                [asarray([[1.0]])]
                if layer_cls != layers.LSTMCell
                else [asarray([[1.0]]), asarray([[1.0]])]
            )
            layer2(asarray([[1.0]]), states)


def test_conv_lstm_cell_use_bias_false():
    with (
        patch("zero_keras.layers.ops") as mock_ops,
        patch(
            "ml_switcheroo_compiler.ops.linalg.conv.conv_general_dilated"
        ) as mock_cgd,
        patch("ml_switcheroo_compiler.ops.linalg.conv_general_dilated") as mock_cgd2,
    ):
        mock_ops.add.return_value = asarray([[1.0]])
        mock_ops.split.return_value = [asarray([[1.0]])] * 4
        mock_ops.multiply.return_value = asarray([[1.0]])
        mock_cgd.return_value = asarray([[1.0]])
        mock_cgd2.return_value = asarray([[1.0]])

        layer = layers.ConvLSTMCell(filters=4, kernel_size=2, use_bias=False)
        layer.built = True
        layer.kernel = asarray([1.0])
        layer.recurrent_kernel = asarray([1.0])
        states = [asarray([[[[1.0]]]]), asarray([[[[1.0]]]])]
        layer(asarray([[[[1.0]]]]), states)


def test_wrapper_already_built():
    inner = layers.Dense(4)
    inner.build((2, 4))
    wrapper = layers.Wrapper(inner)
    wrapper.build((2, 4))


def test_spectral_norm_no_kernel():
    inner = layers.Activation("relu")
    wrapper = layers.SpectralNormalization(inner)
    wrapper.build((2, 4))


def test_spectral_norm_no_weights_attr():
    inner = layers.Dense(4)
    wrapper = layers.SpectralNormalization(inner)
    # mock lack of _weights
    if hasattr(wrapper, "_weights"):
        del wrapper._weights
    wrapper.build((2, 4))


def test_stacked_rnn_cells_built():
    cell = layers.SimpleRNNCell(4)
    cell.build((2, 4))
    stacked = layers.StackedRNNCells([cell])
    stacked.build((2, 4))
