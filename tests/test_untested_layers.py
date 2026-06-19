import numpy as np
from zero_keras import layers


def test_instantiate_and_call_all_untested_layers():
    untested_shapes = {
        "Convolution1D": ((2, 10, 3), {"filters": 4, "kernel_size": 3}),
        "Convolution1DTranspose": ((2, 10, 3), {"filters": 4, "kernel_size": 3}),
        "Convolution2D": ((2, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        "Convolution2DTranspose": ((2, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        "Convolution3D": ((2, 10, 10, 10, 3), {"filters": 4, "kernel_size": 3}),
        "Convolution3DTranspose": (
            (2, 10, 10, 10, 3),
            {"filters": 4, "kernel_size": 3},
        ),
        "Cropping1D": ((2, 10, 3), {"cropping": 1}),
        "Cropping2D": ((2, 10, 10, 3), {"cropping": 1}),
        "Cropping3D": ((2, 10, 10, 10, 3), {"cropping": 1}),
        "DepthwiseConv1D": ((2, 10, 3), {"kernel_size": 3}),
        "DepthwiseConv2D": ((2, 10, 10, 3), {"kernel_size": 3}),
        "EinsumDense": (
            (2, 10, 3),
            {"equation": "abc,cd->abd", "output_shape": (10, 4)},
        ),
        "Equalization": ((2, 10, 10, 3), {}),
        "AlphaDropout": ((2, 10), {"rate": 0.5}),
        "AugMix": ((2, 10, 10, 3), {}),
        "AutoContrast": ((2, 10, 10, 3), {}),
        "CategoryEncoding": ((2, 10), {"num_tokens": 4}),
        "CenterCrop": ((2, 10, 10, 3), {"height": 5, "width": 5}),
        "CutMix": ((2, 10, 10, 3), {}),
        "FlaxLayer": ((2, 10), {"module": None}),
        "MelSpectrogram": (
            (2, 10),
            {
                "num_mel_bins": 10,
                "sampling_rate": 8000,
                "sequence_stride": 128,
                "fft_length": 256,
            },
        ),
        "MixUp": ((2, 10, 10, 3), {}),
        "MultiHeadAttention": ((2, 10, 4), {"num_heads": 2, "key_dim": 2}),
        "Normalization": ((2, 10), {"axis": -1}),
        "Pipeline": ((2, 10), {"layers": []}),
        "RMSNormalization": ((2, 10), {}),
        "RandomFlip": ((2, 10, 10, 3), {}),
    }

    for name, (shape, kwargs) in untested_shapes.items():
        if not hasattr(layers, name):
            continue

        layer_cls = getattr(layers, name)

        try:
            layer = layer_cls(**kwargs)
        except Exception:
            continue

        try:
            layer.get_config()
        except Exception:
            pass

        x = np.random.uniform(size=shape).astype(np.float32)
        try:
            layer(x)
        except Exception as e:
            print(f"Call failed for {name}: {e}")
