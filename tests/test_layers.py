"""Tests for zero_keras layers."""

from zero_keras import layers


def test_layers():
    try:
        layers.Activation()
    except TypeError:
        pass
    try:
        layers.ActivityRegularization()
    except TypeError:
        pass
    try:
        layers.Add()
    except TypeError:
        pass
    try:
        layers.AdditiveAttention()
    except TypeError:
        pass
    try:
        layers.AlphaDropout()
    except TypeError:
        pass
    try:
        layers.Attention()
    except TypeError:
        pass
    try:
        layers.AugMix()
    except TypeError:
        pass
    try:
        layers.AutoContrast()
    except TypeError:
        pass
    try:
        layers.Average()
    except TypeError:
        pass
    try:
        layers.AveragePooling1D()
    except TypeError:
        pass
    try:
        layers.AveragePooling2D()
    except TypeError:
        pass
    try:
        layers.AveragePooling3D()
    except TypeError:
        pass
    try:
        layers.AvgPool1D()
    except TypeError:
        pass
    try:
        layers.AvgPool2D()
    except TypeError:
        pass
    try:
        layers.AvgPool3D()
    except TypeError:
        pass
    try:
        layers.BatchNormalization()
    except TypeError:
        pass
    try:
        layers.Bidirectional()
    except TypeError:
        pass
    try:
        layers.CategoryEncoding()
    except TypeError:
        pass
    try:
        layers.CenterCrop()
    except TypeError:
        pass
    try:
        layers.Concatenate()
    except TypeError:
        pass
    try:
        layers.Conv1D()
    except TypeError:
        pass
    try:
        layers.Conv1DTranspose()
    except TypeError:
        pass
    try:
        layers.Conv2D()
    except TypeError:
        pass
    try:
        layers.Conv2DTranspose()
    except TypeError:
        pass
    try:
        layers.Conv3D()
    except TypeError:
        pass
    try:
        layers.Conv3DTranspose()
    except TypeError:
        pass
    try:
        layers.ConvLSTM1D()
    except TypeError:
        pass
    try:
        layers.ConvLSTM2D()
    except TypeError:
        pass
    try:
        layers.ConvLSTM3D()
    except TypeError:
        pass
    try:
        layers.Convolution1D()
    except TypeError:
        pass
    try:
        layers.Convolution1DTranspose()
    except TypeError:
        pass
    try:
        layers.Convolution2D()
    except TypeError:
        pass
    try:
        layers.Convolution2DTranspose()
    except TypeError:
        pass
    try:
        layers.Convolution3D()
    except TypeError:
        pass
    try:
        layers.Convolution3DTranspose()
    except TypeError:
        pass
    try:
        layers.Cropping1D()
    except TypeError:
        pass
    try:
        layers.Cropping2D()
    except TypeError:
        pass
    try:
        layers.Cropping3D()
    except TypeError:
        pass
    try:
        layers.CutMix()
    except TypeError:
        pass
    try:
        layers.Dense()
    except TypeError:
        pass
    try:
        layers.DepthwiseConv1D()
    except TypeError:
        pass
    try:
        layers.DepthwiseConv2D()
    except TypeError:
        pass
    try:
        layers.Discretization()
    except TypeError:
        pass
    try:
        layers.Dot()
    except TypeError:
        pass
    try:
        layers.Dropout()
    except TypeError:
        pass
    try:
        layers.ELU()
    except TypeError:
        pass
    try:
        layers.EinsumDense()
    except TypeError:
        pass
    try:
        layers.Embedding()
    except TypeError:
        pass
    try:
        layers.Equalization()
    except TypeError:
        pass
    try:
        layers.Flatten()
    except TypeError:
        pass
    try:
        layers.FlaxLayer()
    except TypeError:
        pass
    try:
        layers.GRU()
    except TypeError:
        pass
    try:
        layers.GRUCell()
    except TypeError:
        pass
    try:
        layers.GaussianDropout()
    except TypeError:
        pass
    try:
        layers.GaussianNoise()
    except TypeError:
        pass
    try:
        layers.GlobalAveragePooling1D()
    except TypeError:
        pass
    try:
        layers.GlobalAveragePooling2D()
    except TypeError:
        pass
    try:
        layers.GlobalAveragePooling3D()
    except TypeError:
        pass
    try:
        layers.GlobalAvgPool1D()
    except TypeError:
        pass
    try:
        layers.GlobalAvgPool2D()
    except TypeError:
        pass
    try:
        layers.GlobalAvgPool3D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPool1D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPool2D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPool3D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPooling1D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPooling2D()
    except TypeError:
        pass
    try:
        layers.GlobalMaxPooling3D()
    except TypeError:
        pass
    try:
        layers.GroupNormalization()
    except TypeError:
        pass
    try:
        layers.GroupQueryAttention()
    except TypeError:
        pass
    try:
        layers.HashedCrossing()
    except TypeError:
        pass
    try:
        layers.Hashing()
    except TypeError:
        pass
    try:
        layers.Identity()
    except TypeError:
        pass
    try:
        layers.InputLayer()
    except TypeError:
        pass
    try:
        layers.InputSpec()
    except TypeError:
        pass
    try:
        layers.IntegerLookup()
    except TypeError:
        pass
    try:
        layers.JaxLayer()
    except TypeError:
        pass
    try:
        layers.LSTM()
    except TypeError:
        pass
    try:
        layers.LSTMCell()
    except TypeError:
        pass
    try:
        layers.Lambda()
    except TypeError:
        pass
    try:
        layers.LayerNormalization()
    except TypeError:
        pass
    try:
        layers.LeakyReLU()
    except TypeError:
        pass
    try:
        layers.Masking()
    except TypeError:
        pass
    try:
        layers.MaxNumBoundingBoxes()
    except TypeError:
        pass
    try:
        layers.MaxPool1D()
    except TypeError:
        pass
    try:
        layers.MaxPool2D()
    except TypeError:
        pass
    try:
        layers.MaxPool3D()
    except TypeError:
        pass
    try:
        layers.MaxPooling1D()
    except TypeError:
        pass
    try:
        layers.MaxPooling2D()
    except TypeError:
        pass
    try:
        layers.MaxPooling3D()
    except TypeError:
        pass
    try:
        layers.Maximum()
    except TypeError:
        pass
    try:
        layers.MelSpectrogram()
    except TypeError:
        pass
    try:
        layers.Minimum()
    except TypeError:
        pass
    try:
        layers.MixUp()
    except TypeError:
        pass
    try:
        layers.MultiHeadAttention()
    except TypeError:
        pass
    try:
        layers.Multiply()
    except TypeError:
        pass
    try:
        layers.Normalization()
    except TypeError:
        pass
    try:
        layers.PReLU()
    except TypeError:
        pass
    try:
        layers.Permute()
    except TypeError:
        pass
    try:
        layers.Pipeline()
    except TypeError:
        pass
    try:
        layers.RMSNormalization()
    except TypeError:
        pass
    try:
        layers.RNN()
    except TypeError:
        pass
    try:
        layers.RandAugment()
    except TypeError:
        pass
    try:
        layers.RandomBrightness()
    except TypeError:
        pass
    try:
        layers.RandomColorDegeneration()
    except TypeError:
        pass
    try:
        layers.RandomColorJitter()
    except TypeError:
        pass
    try:
        layers.RandomContrast()
    except TypeError:
        pass
    try:
        layers.RandomCrop()
    except TypeError:
        pass
    try:
        layers.RandomElasticTransform()
    except TypeError:
        pass
    try:
        layers.RandomErasing()
    except TypeError:
        pass
    try:
        layers.RandomFlip()
    except TypeError:
        pass
    try:
        layers.RandomGaussianBlur()
    except TypeError:
        pass
    try:
        layers.RandomGrayscale()
    except TypeError:
        pass
    try:
        layers.RandomHue()
    except TypeError:
        pass
    try:
        layers.RandomInvert()
    except TypeError:
        pass
    try:
        layers.RandomPerspective()
    except TypeError:
        pass
    try:
        layers.RandomPosterization()
    except TypeError:
        pass
    try:
        layers.RandomRotation()
    except TypeError:
        pass
    try:
        layers.RandomSaturation()
    except TypeError:
        pass
    try:
        layers.RandomSharpness()
    except TypeError:
        pass
    try:
        layers.RandomShear()
    except TypeError:
        pass
    try:
        layers.RandomTranslation()
    except TypeError:
        pass
    try:
        layers.RandomZoom()
    except TypeError:
        pass
    try:
        layers.ReLU()
    except TypeError:
        pass
    try:
        layers.RepeatVector()
    except TypeError:
        pass
    try:
        layers.Rescaling()
    except TypeError:
        pass
    try:
        layers.Reshape()
    except TypeError:
        pass
    try:
        layers.Resizing()
    except TypeError:
        pass
    try:
        layers.STFTSpectrogram()
    except TypeError:
        pass
    try:
        layers.SeparableConv1D()
    except TypeError:
        pass
    try:
        layers.SeparableConv2D()
    except TypeError:
        pass
    try:
        layers.SeparableConvolution1D()
    except TypeError:
        pass
    try:
        layers.SeparableConvolution2D()
    except TypeError:
        pass
    try:
        layers.SimpleRNN()
    except TypeError:
        pass
    try:
        layers.SimpleRNNCell()
    except TypeError:
        pass
    try:
        layers.Softmax()
    except TypeError:
        pass
    try:
        layers.Solarization()
    except TypeError:
        pass
    try:
        layers.SpatialDropout1D()
    except TypeError:
        pass
    try:
        layers.SpatialDropout2D()
    except TypeError:
        pass
    try:
        layers.SpatialDropout3D()
    except TypeError:
        pass
    try:
        layers.SpectralNormalization()
    except TypeError:
        pass
    try:
        layers.StackedRNNCells()
    except TypeError:
        pass
    try:
        layers.StringLookup()
    except TypeError:
        pass
    try:
        layers.Subtract()
    except TypeError:
        pass
    try:
        layers.TFSMLayer()
    except TypeError:
        pass
    try:
        layers.TextVectorization()
    except TypeError:
        pass
    try:
        layers.TimeDistributed()
    except TypeError:
        pass
    try:
        layers.TorchModuleWrapper()
    except TypeError:
        pass
    try:
        layers.UnitNormalization()
    except TypeError:
        pass
    try:
        layers.UpSampling1D()
    except TypeError:
        pass
    try:
        layers.UpSampling2D()
    except TypeError:
        pass
    try:
        layers.UpSampling3D()
    except TypeError:
        pass
    try:
        layers.Wrapper()
    except TypeError:
        pass
    try:
        layers.ZeroPadding1D()
    except TypeError:
        pass
    try:
        layers.ZeroPadding2D()
    except TypeError:
        pass
    try:
        layers.ZeroPadding3D()
    except TypeError:
        pass


def test_layers_args():
    import inspect

    for name in dir(layers):
        if name.startswith("_"):
            continue
        obj = getattr(layers, name)
        if (
            type(obj) is type
            and issubclass(obj, layers.Layer)
            and obj is not layers.Layer
        ):
            try:
                sig = inspect.signature(obj.__init__)
                kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    if param.default == inspect.Parameter.empty and param.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        kwargs[param_name] = None
                obj(**kwargs)
            except Exception:
                pass
