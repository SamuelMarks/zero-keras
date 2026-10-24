"""Exhaustive unit and branch coverage tests for ``zero_keras.ops``."""

from __future__ import annotations

from unittest import mock

from zero_keras import ops
from zero_keras.core_layers import KerasTensor, Variable
from zero_keras.types import TensorLike

_ORIG_CONV1D_T = ops.conv1d_transpose
_ORIG_CONV2D_T = ops.conv2d_transpose
_ORIG_CONV3D_T = ops.conv3d_transpose


def test_dispatch_and_wrap_op_branches() -> None:
    """Test ``_dispatch_op`` and ``_wrap_op`` across eager, symbolic, and list-output branches."""
    kt = KerasTensor((2, 4), dtype="float32")

    # Symbolic positional and list args, plus scalar and list kwargs
    res_single = ops._dispatch_op(
        "add",
        lambda *a, **k: a[0],
        ([kt], 1),
        {"axis": 0, "extra": [kt]},
    )
    assert isinstance(res_single, KerasTensor)

    # Symbolic multi-output list return from compute_shape_propagation
    with mock.patch(
        "zero_keras.ops.compute_shape_propagation",
        return_value=[(2, 2), (2, 2)],
    ):
        res_multi = ops._dispatch_op("split", lambda *a, **k: a[0], (kt, 2), {})
        assert isinstance(res_multi, list) and len(res_multi) == 2

    # _wrap_op wrapper invocation
    wrapped = ops._wrap_op("identity", lambda x: x)
    assert wrapped(42) == 42


def test_all_top_level_and_image_ops_symbolic() -> None:
    """Invoke every operation in ``zero_keras.ops`` and ``zero_keras.ops.image`` symbolically."""
    kt = KerasTensor((2, 4), dtype="float32")
    kt_mat = KerasTensor((2, 2), dtype="float32")

    skip_special = {
        "_dispatch_op",
        "_wrap_op",
        "cdist",
        "celu",
        "cholesky_inverse",
        "copy",
        "depth_to_space",
        "dtype",
        "elu",
        "extract_sequences",
        "gelu",
        "get_item",
        "glu",
        "hard_shrink",
        "hard_sigmoid",
        "hard_tanh",
        "is_tensor",
        "jvp",
        "layer_normalization",
        "leaky_relu",
        "mel_filterbank",
        "multi_hot",
        "ndim",
        "psnr",
        "ptp",
        "ravel",
        "rearrange",
        "relu",
        "relu6",
        "repeat",
        "reverse",
        "rnn",
        "saturate_cast",
        "selu",
        "shape",
        "silu",
        "size",
        "soft_shrink",
        "sort",
        "space_to_depth",
        "sparse_top_k_categorical_accuracy",
        "sparsemax",
        "split",
        "swish",
        "tanh_shrink",
        "threshold",
        "top_k_categorical_accuracy",
        "transpose",
        "tril",
        "triu",
        "view",
    }

    with mock.patch("zero_keras.ops.compute_shape_propagation", return_value=(2, 4)):
        for name, fn in list(ops.__dict__.items()):
            if not callable(fn) or isinstance(fn, type) or name in skip_special:
                continue
            if fn.__module__ != "zero_keras.ops":
                continue
            out = fn(kt, (2, 4))
            assert isinstance(out, (KerasTensor, list))

        # Also directly invoke real ops.conv1d_transpose, conv2d_transpose, conv3d_transpose bypassed by conftest monkeypatch
        assert _ORIG_CONV1D_T(kt, kt) is not None
        assert _ORIG_CONV2D_T(kt, kt) is not None
        assert _ORIG_CONV3D_T(kt, kt) is not None

        for name, fn in list(ops.image.__dict__.items()):
            if name.startswith("_") or name in (
                "resize",
                "scale_and_translate",
                "sobel_edges",
                "ssim",
            ):
                continue
            static_fn = ops.image.__dict__[name].__func__
            out_img = static_fn(kt, (2, 2))
            assert isinstance(out_img, (KerasTensor, list))


def test_custom_composed_ops_branches() -> None:
    """Cover every branch in custom composed operations in ``zero_keras.ops``."""
    x = ops.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    y = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    kt = KerasTensor((2, 4), dtype="float32")

    # cdist (positional and kwargs)
    assert ops.cdist(x, y) is not None
    assert ops.cdist(x=x, y=y) is not None

    # cholesky_inverse (upper=False and upper=True, positional and kwargs)
    assert ops.cholesky_inverse(y, False) is not None
    assert ops.cholesky_inverse(x=y, upper=True) is not None

    # depth_to_space & space_to_depth & copy (positional and kwargs)
    assert ops.copy(x) is not None
    assert ops.copy(x=x) is not None
    assert ops.copy(kt) is not None
    assert ops.depth_to_space(x, 2) is not None
    assert ops.depth_to_space(x=x, block_size=2) is not None
    assert ops.space_to_depth(x, 2) is not None
    assert ops.space_to_depth(x=x, block_size=2) is not None

    # dtype (Tensor, KerasTensor, raw python list)
    assert ops.dtype(x) == "float32"
    assert ops.dtype(kt) == "float32"
    assert ops.dtype([1.0, 2.0]) in ("float32", "float64")

    # extract_sequences
    seq = ops.convert_to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
    assert ops.extract_sequences(seq, 2, 1).shape == (3, 2)

    # get_item (Tensor, list, raw scalar/nested)
    assert ops.get_item(x, 0) is not None
    assert ops.get_item([10, 20], 1) == 20
    assert ops.get_item(ops.array([5, 6]).data, 0) is not None

    # is_tensor
    v = Variable("zeros", shape=(2,))
    assert ops.is_tensor(x) is True
    assert ops.is_tensor(kt) is True
    assert ops.is_tensor(v) is True
    assert ops.is_tensor(123) is False

    # jvp (has_aux=False, has_aux=True, empty tangents)
    p_out, t_out = ops.jvp(lambda a: a, [x], [y], has_aux=False)
    assert p_out is not None and t_out is not None
    p_out2, t_out2, aux = ops.jvp(lambda a: (a, "aux_val"), [x], [y], has_aux=True)
    assert aux == "aux_val"
    p_out3, t_out3 = ops.jvp(lambda a: a, [x], [], has_aux=False)
    assert p_out3 is not None and t_out3 is not None

    # layer_normalization (with and without gamma/beta)
    assert ops.layer_normalization(x) is not None
    gamma = ops.ones((2,), dtype="float32")
    beta = ops.zeros((2,), dtype="float32")
    assert ops.layer_normalization(x, gamma=gamma, beta=beta) is not None

    # mel_filterbank & multi_hot
    assert ops.mel_filterbank(num_mel_bins=8, num_spectrogram_bins=16).shape == (16, 8)
    idx_t = ops.convert_to_tensor([[0, 2], [1, 1]], dtype="int32")
    assert ops.multi_hot(idx_t, num_classes=4).shape == (2, 4)

    # ndim (Tensor, KerasTensor, raw list)
    assert ops.ndim(x) == 2
    assert ops.ndim(kt) == 2
    assert ops.ndim([[1, 2], [3, 4]]) == 2

    # psnr, ptp, ravel, rearrange, saturate_cast
    assert ops.psnr(x, y, 255.0) is not None
    assert ops.ptp(x, axis=-1) is not None
    assert ops.ravel(x).shape == (4,)
    assert ops.rearrange(x, "h w -> h w") is not None
    assert ops.dtype(ops.saturate_cast(x, "int32")) == "int32"

    # repeat (with and without dim)
    assert ops.repeat(x, 2, axis=0) is not None
    assert ops.repeat(x, 2, dim=0) is not None

    # reverse (dims, axes, no kw)
    assert ops.reverse(x, dims=0) is not None
    assert ops.reverse(x, axes=0) is not None
    assert ops.reverse(x, 0) is not None

    # rnn
    def dummy_cell(inp: TensorLike, st: TensorLike) -> tuple[TensorLike, TensorLike]:
        """Dummy RNN cell step.

        Args:
            inp: Step input.
            st: Step state.

        Returns:
            tuple[TensorLike, TensorLike]: ``(inp, st)``.
        """
        return inp, st

    assert ops.rnn(kt, [kt], dummy_cell) is not None

    # shape & size (Tensor, KerasTensor, raw list)
    assert ops.shape(x) == (2, 2)
    assert len(ops.shape(kt)) == 2
    assert ops.shape([[1, 2, 3]]) == (1, 3)

    assert ops.size(x) == 4
    assert ops.size(KerasTensor((None, 3), dtype="float32")) == 3

    # sort & split (with axis and with dim)
    assert ops.sort(x, axis=-1) is not None
    assert len(ops.split(x, 2, axis=0)) == 2  # type: ignore[arg-type]
    assert len(ops.split(x, 2, dim=0)) == 2  # type: ignore[arg-type]

    # sparse_top_k_categorical_accuracy & top_k_categorical_accuracy
    y_pred_logits = ops.convert_to_tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]], dtype="float32")
    y_true_1d = ops.convert_to_tensor([1, 0], dtype="int32")
    y_true_2d = ops.convert_to_tensor([[1], [0]], dtype="int32")
    y_true_oh = ops.convert_to_tensor([[0, 1, 0], [1, 0, 0]], dtype="float32")
    assert ops.sparse_top_k_categorical_accuracy(y_true_1d, y_pred_logits, k=1) is not None
    assert ops.sparse_top_k_categorical_accuracy(y_true_2d, y_pred_logits, k=1) is not None
    assert ops.top_k_categorical_accuracy(y_true_oh, y_pred_logits, k=1) is not None

    # transpose (axes kwarg, positional axes, default)
    assert ops.transpose(x).shape == (2, 2)
    assert ops.transpose(x, (1, 0)).shape == (2, 2)
    assert ops.transpose(x, axes=(1, 0)).shape == (2, 2)

    # tril, triu, view
    assert ops.tril(x) is not None
    assert ops.triu(x) is not None
    assert ops.view(x, None) is not None
    assert ops.view(x, "int32") is not None


def test_activation_wrappers_and_image_ops_in_ops() -> None:
    """Cover all activation wrappers and ``ops.image`` methods in ``zero_keras.ops``."""
    x = ops.convert_to_tensor([[-1.0, 0.5, 2.0, -0.5]], dtype="float32")
    for act_name in [
        "celu",
        "elu",
        "gelu",
        "glu",
        "hard_shrink",
        "hard_sigmoid",
        "hard_tanh",
        "leaky_relu",
        "relu",
        "relu6",
        "selu",
        "silu",
        "soft_shrink",
        "sparsemax",
        "swish",
        "tanh_shrink",
    ]:
        fn = ops.__dict__[act_name]
        assert fn(x) is not None
    assert ops.threshold(x, 0.0, 0.0) is not None

    # ops.image.resize across 4D, 3D, 2D in both symbolic and eager modes + channels_first/last + scalar size
    for rank, shp in [(4, (2, 3, 8, 8)), (3, (3, 8, 8)), (2, (8, 8))]:
        kt_img = KerasTensor(shp, dtype="float32")
        eager_img = ops.zeros(shp, dtype="float32")
        assert ops.image.resize(kt_img, (4, 4), data_format="channels_last") is not None
        assert ops.image.resize(kt_img, (4, 4), data_format="channels_first") is not None
        assert ops.image.resize(eager_img, (4, 4), data_format="channels_last") is not None
        assert ops.image.resize(eager_img, (4, 4), data_format="channels_first") is not None
    assert ops.image.resize(images=ops.zeros((4, 4)), size=2).shape == (2, 2)

    # scale_and_translate, sobel_edges, ssim
    img4 = ops.zeros((1, 4, 4, 3), dtype="float32")
    assert ops.image.scale_and_translate(img4, (1, 2, 2, 3), (1.0, 1.0), (0.0, 0.0)).shape == (
        1,
        2,
        2,
        3,
    )
    assert ops.image.scale_and_translate(images=img4, output_shape=(2, 2)).shape == (1, 2, 2, 3)
    assert ops.image.scale_and_translate(images=img4, output_shape=1) is not None
    assert ops.image.sobel_edges(img4) is not None
    assert ops.image.sobel_edges(images=img4) is not None
    assert ops.image.ssim(img4, img4) is not None
    assert ops.image.ssim(image1=img4, image2=img4) is not None
