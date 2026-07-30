"""Ops wrapper module."""

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation


def _wrap_op(name, msc_op):
    def wrapper(*args, **kwargs):
        from zero_keras.core_layers import KerasTensor

        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
        for a in kwargs.values():
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)

        has_kt = False
        for x in flat_args:
            if isinstance(x, KerasTensor):
                has_kt = True
                break

        if has_kt:
            kt = next(x for x in flat_args if isinstance(x, KerasTensor))
            shape_or_shapes = compute_shape_propagation(name, kt.shape, args, kwargs)
            if isinstance(shape_or_shapes, list):
                return [KerasTensor(s, dtype=kt.dtype) for s in shape_or_shapes]
            return KerasTensor(shape_or_shapes, dtype=kt.dtype)
        return msc_op(*args, **kwargs)

    return wrapper


def AdjustBrightness(*args, **kwargs):
    return _wrap_op("AdjustBrightness", msc_ops.AdjustBrightness)(*args, **kwargs)


def AdjustContrast(*args, **kwargs):
    return _wrap_op("AdjustContrast", msc_ops.AdjustContrast)(*args, **kwargs)


def AdjustHue(*args, **kwargs):
    return _wrap_op("AdjustHue", msc_ops.AdjustHue)(*args, **kwargs)


def AdjustSaturation(*args, **kwargs):
    return _wrap_op("AdjustSaturation", msc_ops.AdjustSaturation)(*args, **kwargs)


def AffineConfig(*args, **kwargs):
    return _wrap_op("AffineConfig", msc_ops.AffineConfig)(*args, **kwargs)


def AffineGenerator(*args, **kwargs):
    return _wrap_op("AffineGenerator", msc_ops.AffineGenerator)(*args, **kwargs)


def AffineGrid(*args, **kwargs):
    return _wrap_op("AffineGrid", msc_ops.AffineGrid)(*args, **kwargs)


def AffineTransform(*args, **kwargs):
    return _wrap_op("AffineTransform", msc_ops.AffineTransform)(*args, **kwargs)


def Any(*args, **kwargs):
    return _wrap_op("Any", msc_ops.Any)(*args, **kwargs)


def Append(*args, **kwargs):
    return _wrap_op("Append", msc_ops.Append)(*args, **kwargs)


def AsStringConfig(*args, **kwargs):
    return _wrap_op("AsStringConfig", msc_ops.AsStringConfig)(*args, **kwargs)


def AssertOp(*args, **kwargs):
    return _wrap_op("AssertOp", msc_ops.AssertOp)(*args, **kwargs)


def AssignVariable(*args, **kwargs):
    return _wrap_op("AssignVariable", msc_ops.AssignVariable)(*args, **kwargs)


def AssociativeScan(*args, **kwargs):
    return _wrap_op("AssociativeScan", msc_ops.AssociativeScan)(*args, **kwargs)


def Atleast1d(*args, **kwargs):
    return _wrap_op("Atleast1d", msc_ops.Atleast1d)(*args, **kwargs)


def Atleast2d(*args, **kwargs):
    return _wrap_op("Atleast2d", msc_ops.Atleast2d)(*args, **kwargs)


def Atleast3d(*args, **kwargs):
    return _wrap_op("Atleast3d", msc_ops.Atleast3d)(*args, **kwargs)


def AugMix(*args, **kwargs):
    return _wrap_op("AugMix", msc_ops.AugMix)(*args, **kwargs)


def AutoContrast(*args, **kwargs):
    return _wrap_op("AutoContrast", msc_ops.AutoContrast)(*args, **kwargs)


def Block(*args, **kwargs):
    return _wrap_op("Block", msc_ops.Block)(*args, **kwargs)


def BlurConfig(*args, **kwargs):
    return _wrap_op("BlurConfig", msc_ops.BlurConfig)(*args, **kwargs)


def BooleanMask(*args, **kwargs):
    return _wrap_op("BooleanMask", msc_ops.BooleanMask)(*args, **kwargs)


def Callable(*args, **kwargs):
    return _wrap_op("Callable", msc_ops.Callable)(*args, **kwargs)


def ColumnStack(*args, **kwargs):
    return _wrap_op("ColumnStack", msc_ops.ColumnStack)(*args, **kwargs)


def ConvGeneralDilated(*args, **kwargs):
    return _wrap_op("ConvGeneralDilated", msc_ops.ConvGeneralDilated)(*args, **kwargs)


def ConvGeneralDilatedLocal(*args, **kwargs):
    return _wrap_op("ConvGeneralDilatedLocal", msc_ops.ConvGeneralDilatedLocal)(
        *args, **kwargs
    )


def ConvGeneralDilatedPatches(*args, **kwargs):
    return _wrap_op("ConvGeneralDilatedPatches", msc_ops.ConvGeneralDilatedPatches)(
        *args, **kwargs
    )


def ConvTranspose(*args, **kwargs):
    return _wrap_op("ConvTranspose", msc_ops.ConvTranspose)(*args, **kwargs)


def ConvWithGeneralPadding(*args, **kwargs):
    return _wrap_op("ConvWithGeneralPadding", msc_ops.ConvWithGeneralPadding)(
        *args, **kwargs
    )


def CustomLinearSolve(*args, **kwargs):
    return _wrap_op("CustomLinearSolve", msc_ops.CustomLinearSolve)(*args, **kwargs)


def CustomRoot(*args, **kwargs):
    return _wrap_op("CustomRoot", msc_ops.CustomRoot)(*args, **kwargs)


def DType(*args, **kwargs):
    return _wrap_op("DType", msc_ops.DType)(*args, **kwargs)


def DebugInfs(*args, **kwargs):
    return _wrap_op("DebugInfs", msc_ops.DebugInfs)(*args, **kwargs)


def DebugNans(*args, **kwargs):
    return _wrap_op("DebugNans", msc_ops.DebugNans)(*args, **kwargs)


def Delete(*args, **kwargs):
    return _wrap_op("Delete", msc_ops.Delete)(*args, **kwargs)


def DiagIndices(*args, **kwargs):
    return _wrap_op("DiagIndices", msc_ops.DiagIndices)(*args, **kwargs)


def DiagIndicesFrom(*args, **kwargs):
    return _wrap_op("DiagIndicesFrom", msc_ops.DiagIndicesFrom)(*args, **kwargs)


def Diagflat(*args, **kwargs):
    return _wrap_op("Diagflat", msc_ops.Diagflat)(*args, **kwargs)


def Dot(*args, **kwargs):
    return _wrap_op("Dot", msc_ops.Dot)(*args, **kwargs)


def DotGeneral(*args, **kwargs):
    return _wrap_op("DotGeneral", msc_ops.DotGeneral)(*args, **kwargs)


def Dsplit(*args, **kwargs):
    return _wrap_op("Dsplit", msc_ops.Dsplit)(*args, **kwargs)


def Dstack(*args, **kwargs):
    return _wrap_op("Dstack", msc_ops.Dstack)(*args, **kwargs)


def Einsum(*args, **kwargs):
    return _wrap_op("Einsum", msc_ops.Einsum)(*args, **kwargs)


def ElasticConfig(*args, **kwargs):
    return _wrap_op("ElasticConfig", msc_ops.ElasticConfig)(*args, **kwargs)


def ElasticTransform(*args, **kwargs):
    return _wrap_op("ElasticTransform", msc_ops.ElasticTransform)(*args, **kwargs)


def Equalization(*args, **kwargs):
    return _wrap_op("Equalization", msc_ops.Equalization)(*args, **kwargs)


def ExpandDims(*args, **kwargs):
    return _wrap_op("ExpandDims", msc_ops.ExpandDims)(*args, **kwargs)


def ExtractBoundingBoxes(*args, **kwargs):
    return _wrap_op("ExtractBoundingBoxes", msc_ops.ExtractBoundingBoxes)(
        *args, **kwargs
    )


def ExtractPatchesOptions(*args, **kwargs):
    return _wrap_op("ExtractPatchesOptions", msc_ops.ExtractPatchesOptions)(
        *args, **kwargs
    )


def FillDiagonal(*args, **kwargs):
    return _wrap_op("FillDiagonal", msc_ops.FillDiagonal)(*args, **kwargs)


def Flip(*args, **kwargs):
    return _wrap_op("Flip", msc_ops.Flip)(*args, **kwargs)


def Fliplr(*args, **kwargs):
    return _wrap_op("Fliplr", msc_ops.Fliplr)(*args, **kwargs)


def Flipud(*args, **kwargs):
    return _wrap_op("Flipud", msc_ops.Flipud)(*args, **kwargs)


def Hsplit(*args, **kwargs):
    return _wrap_op("Hsplit", msc_ops.Hsplit)(*args, **kwargs)


def Hstack(*args, **kwargs):
    return _wrap_op("Hstack", msc_ops.Hstack)(*args, **kwargs)


def Insert(*args, **kwargs):
    return _wrap_op("Insert", msc_ops.Insert)(*args, **kwargs)


def Invert(*args, **kwargs):
    return _wrap_op("Invert", msc_ops.Invert)(*args, **kwargs)


def Iou(*args, **kwargs):
    return _wrap_op("Iou", msc_ops.Iou)(*args, **kwargs)


def LinearOperator(*args, **kwargs):
    return _wrap_op("LinearOperator", msc_ops.LinearOperator)(*args, **kwargs)


def LinearOperatorAdjoint(*args, **kwargs):
    return _wrap_op("LinearOperatorAdjoint", msc_ops.LinearOperatorAdjoint)(
        *args, **kwargs
    )


def LinearOperatorBlockDiag(*args, **kwargs):
    return _wrap_op("LinearOperatorBlockDiag", msc_ops.LinearOperatorBlockDiag)(
        *args, **kwargs
    )


def LinearOperatorBlockLowerTriangular(*args, **kwargs):
    return _wrap_op(
        "LinearOperatorBlockLowerTriangular",
        msc_ops.LinearOperatorBlockLowerTriangular,
    )(*args, **kwargs)


def LinearOperatorCirculant(*args, **kwargs):
    return _wrap_op("LinearOperatorCirculant", msc_ops.LinearOperatorCirculant)(
        *args, **kwargs
    )


def LinearOperatorCirculant2D(*args, **kwargs):
    return _wrap_op("LinearOperatorCirculant2D", msc_ops.LinearOperatorCirculant2D)(
        *args, **kwargs
    )


def LinearOperatorCirculant3D(*args, **kwargs):
    return _wrap_op("LinearOperatorCirculant3D", msc_ops.LinearOperatorCirculant3D)(
        *args, **kwargs
    )


def LinearOperatorComposition(*args, **kwargs):
    return _wrap_op("LinearOperatorComposition", msc_ops.LinearOperatorComposition)(
        *args, **kwargs
    )


def LinearOperatorDiag(*args, **kwargs):
    return _wrap_op("LinearOperatorDiag", msc_ops.LinearOperatorDiag)(*args, **kwargs)


def LinearOperatorFullMatrix(*args, **kwargs):
    return _wrap_op("LinearOperatorFullMatrix", msc_ops.LinearOperatorFullMatrix)(
        *args, **kwargs
    )


def LinearOperatorHouseholder(*args, **kwargs):
    return _wrap_op("LinearOperatorHouseholder", msc_ops.LinearOperatorHouseholder)(
        *args, **kwargs
    )


def LinearOperatorIdentity(*args, **kwargs):
    return _wrap_op("LinearOperatorIdentity", msc_ops.LinearOperatorIdentity)(
        *args, **kwargs
    )


def LinearOperatorInversion(*args, **kwargs):
    return _wrap_op("LinearOperatorInversion", msc_ops.LinearOperatorInversion)(
        *args, **kwargs
    )


def LinearOperatorKronecker(*args, **kwargs):
    return _wrap_op("LinearOperatorKronecker", msc_ops.LinearOperatorKronecker)(
        *args, **kwargs
    )


def LinearOperatorLowRankUpdate(*args, **kwargs):
    return _wrap_op("LinearOperatorLowRankUpdate", msc_ops.LinearOperatorLowRankUpdate)(
        *args, **kwargs
    )


def LinearOperatorLowerTriangular(*args, **kwargs):
    return _wrap_op(
        "LinearOperatorLowerTriangular",
        msc_ops.LinearOperatorLowerTriangular,
    )(*args, **kwargs)


def LinearOperatorPermutation(*args, **kwargs):
    return _wrap_op("LinearOperatorPermutation", msc_ops.LinearOperatorPermutation)(
        *args, **kwargs
    )


def LinearOperatorScaledIdentity(*args, **kwargs):
    return _wrap_op(
        "LinearOperatorScaledIdentity", msc_ops.LinearOperatorScaledIdentity
    )(*args, **kwargs)


def LinearOperatorToeplitz(*args, **kwargs):
    return _wrap_op("LinearOperatorToeplitz", msc_ops.LinearOperatorToeplitz)(
        *args, **kwargs
    )


def LinearOperatorTridiag(*args, **kwargs):
    return _wrap_op("LinearOperatorTridiag", msc_ops.LinearOperatorTridiag)(
        *args, **kwargs
    )


def LinearOperatorZeros(*args, **kwargs):
    return _wrap_op("LinearOperatorZeros", msc_ops.LinearOperatorZeros)(*args, **kwargs)


def MapFlatValues(*args, **kwargs):
    return _wrap_op("MapFlatValues", msc_ops.MapFlatValues)(*args, **kwargs)


def Matmul(*args, **kwargs):
    return _wrap_op("Matmul", msc_ops.Matmul)(*args, **kwargs)


def MatrixRank(*args, **kwargs):
    return _wrap_op("MatrixRank", msc_ops.MatrixRank)(*args, **kwargs)


def MatrixTranspose(*args, **kwargs):
    return _wrap_op("MatrixTranspose", msc_ops.MatrixTranspose)(*args, **kwargs)


def Moveaxis(*args, **kwargs):
    return _wrap_op("Moveaxis", msc_ops.Moveaxis)(*args, **kwargs)


def Nms(*args, **kwargs):
    return _wrap_op("Nms", msc_ops.Nms)(*args, **kwargs)


def NormConfig(*args, **kwargs):
    return _wrap_op("NormConfig", msc_ops.NormConfig)(*args, **kwargs)


def OpDef(*args, **kwargs):
    return _wrap_op("OpDef", msc_ops.OpDef)(*args, **kwargs)


def Pdot(*args, **kwargs):
    return _wrap_op("Pdot", msc_ops.Pdot)(*args, **kwargs)


def Permute(*args, **kwargs):
    return _wrap_op("Permute", msc_ops.Permute)(*args, **kwargs)


def PerspectiveConfig(*args, **kwargs):
    return _wrap_op("PerspectiveConfig", msc_ops.PerspectiveConfig)(*args, **kwargs)


def PerspectiveTransform(*args, **kwargs):
    return _wrap_op("PerspectiveTransform", msc_ops.PerspectiveTransform)(
        *args, **kwargs
    )


def Pmean(*args, **kwargs):
    return _wrap_op("Pmean", msc_ops.Pmean)(*args, **kwargs)


def Posterize(*args, **kwargs):
    return _wrap_op("Posterize", msc_ops.Posterize)(*args, **kwargs)


def Psum(*args, **kwargs):
    return _wrap_op("Psum", msc_ops.Psum)(*args, **kwargs)


def RNNConfig(*args, **kwargs):
    return _wrap_op("RNNConfig", msc_ops.RNNConfig)(*args, **kwargs)


def RaggedAdd(*args, **kwargs):
    return _wrap_op("RaggedAdd", msc_ops.RaggedAdd)(*args, **kwargs)


def RaggedConstant(*args, **kwargs):
    return _wrap_op("RaggedConstant", msc_ops.RaggedConstant)(*args, **kwargs)


def RaggedCrossHashed(*args, **kwargs):
    return _wrap_op("RaggedCrossHashed", msc_ops.RaggedCrossHashed)(*args, **kwargs)


def RaggedDot(*args, **kwargs):
    return _wrap_op("RaggedDot", msc_ops.RaggedDot)(*args, **kwargs)


def RaggedDynamicBroadcast(*args, **kwargs):
    return _wrap_op("RaggedDynamicBroadcast", msc_ops.RaggedDynamicBroadcast)(
        *args, **kwargs
    )


def RaggedGather(*args, **kwargs):
    return _wrap_op("RaggedGather", msc_ops.RaggedGather)(*args, **kwargs)


def RaggedMatMul(*args, **kwargs):
    return _wrap_op("RaggedMatMul", msc_ops.RaggedMatMul)(*args, **kwargs)


def RaggedRange(*args, **kwargs):
    return _wrap_op("RaggedRange", msc_ops.RaggedRange)(*args, **kwargs)


def RaggedRowSplitsToSegmentIds(*args, **kwargs):
    return _wrap_op("RaggedRowSplitsToSegmentIds", msc_ops.RaggedRowSplitsToSegmentIds)(
        *args, **kwargs
    )


def RaggedSegmentIdsToRowSplits(*args, **kwargs):
    return _wrap_op("RaggedSegmentIdsToRowSplits", msc_ops.RaggedSegmentIdsToRowSplits)(
        *args, **kwargs
    )


def RaggedStack(*args, **kwargs):
    return _wrap_op("RaggedStack", msc_ops.RaggedStack)(*args, **kwargs)


def RaggedStackDynamicPartitions(*args, **kwargs):
    return _wrap_op(
        "RaggedStackDynamicPartitions", msc_ops.RaggedStackDynamicPartitions
    )(*args, **kwargs)


def RaggedTensorToDense(*args, **kwargs):
    return _wrap_op("RaggedTensorToDense", msc_ops.RaggedTensorToDense)(*args, **kwargs)


def RawConv2D(*args, **kwargs):
    return _wrap_op("RawConv2D", msc_ops.RawConv2D)(*args, **kwargs)


def RawMatMul(*args, **kwargs):
    return _wrap_op("RawMatMul", msc_ops.RawMatMul)(*args, **kwargs)


def RawMerge(*args, **kwargs):
    return _wrap_op("RawMerge", msc_ops.RawMerge)(*args, **kwargs)


def RawOp(*args, **kwargs):
    return _wrap_op("RawOp", msc_ops.RawOp)(*args, **kwargs)


def RawSwitch(*args, **kwargs):
    return _wrap_op("RawSwitch", msc_ops.RawSwitch)(*args, **kwargs)


def ReadVariable(*args, **kwargs):
    return _wrap_op("ReadVariable", msc_ops.ReadVariable)(*args, **kwargs)


def Resize(*args, **kwargs):
    return _wrap_op("Resize", msc_ops.Resize)(*args, **kwargs)


def RgbToGrayscale(*args, **kwargs):
    return _wrap_op("RgbToGrayscale", msc_ops.RgbToGrayscale)(*args, **kwargs)


def Roll(*args, **kwargs):
    return _wrap_op("Roll", msc_ops.Roll)(*args, **kwargs)


def ScanOp(*args, **kwargs):
    return _wrap_op("ScanOp", msc_ops.ScanOp)(*args, **kwargs)


def SobolSample(*args, **kwargs):
    return _wrap_op("SobolSample", msc_ops.SobolSample)(*args, **kwargs)


def Solarize(*args, **kwargs):
    return _wrap_op("Solarize", msc_ops.Solarize)(*args, **kwargs)


def SparseConcat(*args, **kwargs):
    return _wrap_op("SparseConcat", msc_ops.SparseConcat)(*args, **kwargs)


def SparseDenseMatMul(*args, **kwargs):
    return _wrap_op("SparseDenseMatMul", msc_ops.SparseDenseMatMul)(*args, **kwargs)


def SparseMapValues(*args, **kwargs):
    return _wrap_op("SparseMapValues", msc_ops.SparseMapValues)(*args, **kwargs)


def SparseReduceMax(*args, **kwargs):
    return _wrap_op("SparseReduceMax", msc_ops.SparseReduceMax)(*args, **kwargs)


def SparseReshape(*args, **kwargs):
    return _wrap_op("SparseReshape", msc_ops.SparseReshape)(*args, **kwargs)


def SparseSampledAdd(*args, **kwargs):
    return _wrap_op("SparseSampledAdd", msc_ops.SparseSampledAdd)(*args, **kwargs)


def SparseSegmentSum(*args, **kwargs):
    return _wrap_op("SparseSegmentSum", msc_ops.SparseSegmentSum)(*args, **kwargs)


def SparseSplit(*args, **kwargs):
    return _wrap_op("SparseSplit", msc_ops.SparseSplit)(*args, **kwargs)


def SparseToDense(*args, **kwargs):
    return _wrap_op("SparseToDense", msc_ops.SparseToDense)(*args, **kwargs)


def SparseTranspose(*args, **kwargs):
    return _wrap_op("SparseTranspose", msc_ops.SparseTranspose)(*args, **kwargs)


def Sqrtm(*args, **kwargs):
    return _wrap_op("Sqrtm", msc_ops.Sqrtm)(*args, **kwargs)


def Squeeze(*args, **kwargs):
    return _wrap_op("Squeeze", msc_ops.Squeeze)(*args, **kwargs)


def Swapaxes(*args, **kwargs):
    return _wrap_op("Swapaxes", msc_ops.Swapaxes)(*args, **kwargs)


def SwitchOp(*args, **kwargs):
    return _wrap_op("SwitchOp", msc_ops.SwitchOp)(*args, **kwargs)


def TFRecordOptions(*args, **kwargs):
    return _wrap_op("TFRecordOptions", msc_ops.TFRecordOptions)(*args, **kwargs)


def TFRecordWriter(*args, **kwargs):
    return _wrap_op("TFRecordWriter", msc_ops.TFRecordWriter)(*args, **kwargs)


def Tensor(*args, **kwargs):
    return _wrap_op("Tensor", msc_ops.Tensor)(*args, **kwargs)


def TensorArrayRead(*args, **kwargs):
    return _wrap_op("TensorArrayRead", msc_ops.TensorArrayRead)(*args, **kwargs)


def TensorArrayStack(*args, **kwargs):
    return _wrap_op("TensorArrayStack", msc_ops.TensorArrayStack)(*args, **kwargs)


def TensorArrayWrite(*args, **kwargs):
    return _wrap_op("TensorArrayWrite", msc_ops.TensorArrayWrite)(*args, **kwargs)


def TensorConfig(*args, **kwargs):
    return _wrap_op("TensorConfig", msc_ops.TensorConfig)(*args, **kwargs)


def Trace(*args, **kwargs):
    return _wrap_op("Trace", msc_ops.Trace)(*args, **kwargs)


def Vecdot(*args, **kwargs):
    return _wrap_op("Vecdot", msc_ops.Vecdot)(*args, **kwargs)


def Vsplit(*args, **kwargs):
    return _wrap_op("Vsplit", msc_ops.Vsplit)(*args, **kwargs)


def Vstack(*args, **kwargs):
    return _wrap_op("Vstack", msc_ops.Vstack)(*args, **kwargs)


def abs(*args, **kwargs):
    return _wrap_op("abs", msc_ops.abs)(*args, **kwargs)


def absolute(*args, **kwargs):
    raise NotImplementedError(
        "ops.absolute is not yet implemented in ml-switcheroo-compiler"
    )


def accumulate_n(*args, **kwargs):
    return _wrap_op("accumulate_n", msc_ops.accumulate_n)(*args, **kwargs)


def acos(*args, **kwargs):
    return _wrap_op("acos", msc_ops.acos)(*args, **kwargs)


def acosh(*args, **kwargs):
    return _wrap_op("acosh", msc_ops.acosh)(*args, **kwargs)


activations = msc_ops.activations


def adaptive_avg_pool2d(*args, **kwargs):
    return _wrap_op("adaptive_avg_pool2d", msc_ops.adaptive_avg_pool2d)(*args, **kwargs)


def adaptive_max_pool2d(*args, **kwargs):
    return _wrap_op("adaptive_max_pool2d", msc_ops.adaptive_max_pool2d)(*args, **kwargs)


def add(*args, **kwargs):
    return _wrap_op("add", msc_ops.add)(*args, **kwargs)


def add_n(*args, **kwargs):
    return _wrap_op("add_n", msc_ops.add_n)(*args, **kwargs)


def addmm(*args, **kwargs):
    return _wrap_op("addmm", msc_ops.addmm)(*args, **kwargs)


def adjoint(*args, **kwargs):
    return _wrap_op("adjoint", msc_ops.adjoint)(*args, **kwargs)


def adjust_brightness(*args, **kwargs):
    return _wrap_op("adjust_brightness", msc_ops.adjust_brightness)(*args, **kwargs)


def adjust_contrast(*args, **kwargs):
    return _wrap_op("adjust_contrast", msc_ops.adjust_contrast)(*args, **kwargs)


def adjust_hue(*args, **kwargs):
    return _wrap_op("adjust_hue", msc_ops.adjust_hue)(*args, **kwargs)


def adjust_saturation(*args, **kwargs):
    return _wrap_op("adjust_saturation", msc_ops.adjust_saturation)(*args, **kwargs)


def affine_generator(*args, **kwargs):
    return _wrap_op("affine_generator", msc_ops.affine_generator)(*args, **kwargs)


def affine_grid(*args, **kwargs):
    return _wrap_op("affine_grid", msc_ops.affine_grid)(*args, **kwargs)


def affine_transform(*args, **kwargs):
    return _wrap_op("affine_transform", msc_ops.affine_transform)(*args, **kwargs)


aggregations = msc_ops.aggregations


def all(*args, **kwargs):
    return _wrap_op("all", msc_ops.all)(*args, **kwargs)


def allclose(*args, **kwargs):
    return _wrap_op("allclose", msc_ops.allclose)(*args, **kwargs)


def amax(*args, **kwargs):
    raise NotImplementedError(
        "ops.amax is not yet implemented in ml-switcheroo-compiler"
    )


def amin(*args, **kwargs):
    raise NotImplementedError(
        "ops.amin is not yet implemented in ml-switcheroo-compiler"
    )


def angle(*args, **kwargs):
    return _wrap_op("angle", msc_ops.angle)(*args, **kwargs)


annotations = msc_ops.annotations


def any(*args, **kwargs):
    return _wrap_op("any", msc_ops.any)(*args, **kwargs)


def append(*args, **kwargs):
    return _wrap_op("append", msc_ops.append)(*args, **kwargs)


def approx_max_k(*args, **kwargs):
    return _wrap_op("approx_max_k", msc_ops.approx_max_k)(*args, **kwargs)


def approx_min_k(*args, **kwargs):
    return _wrap_op("approx_min_k", msc_ops.approx_min_k)(*args, **kwargs)


def arange(*args, **kwargs):
    return _wrap_op("arange", msc_ops.arange)(*args, **kwargs)


def arccos(*args, **kwargs):
    raise NotImplementedError(
        "ops.arccos is not yet implemented in ml-switcheroo-compiler"
    )


def arccosh(*args, **kwargs):
    raise NotImplementedError(
        "ops.arccosh is not yet implemented in ml-switcheroo-compiler"
    )


def arcsin(*args, **kwargs):
    raise NotImplementedError(
        "ops.arcsin is not yet implemented in ml-switcheroo-compiler"
    )


def arcsinh(*args, **kwargs):
    raise NotImplementedError(
        "ops.arcsinh is not yet implemented in ml-switcheroo-compiler"
    )


def arctan(*args, **kwargs):
    raise NotImplementedError(
        "ops.arctan is not yet implemented in ml-switcheroo-compiler"
    )


def arctan2(*args, **kwargs):
    raise NotImplementedError(
        "ops.arctan2 is not yet implemented in ml-switcheroo-compiler"
    )


def arctanh(*args, **kwargs):
    raise NotImplementedError(
        "ops.arctanh is not yet implemented in ml-switcheroo-compiler"
    )


def argmax(*args, **kwargs):
    return _wrap_op("argmax", msc_ops.argmax)(*args, **kwargs)


def argmin(*args, **kwargs):
    return _wrap_op("argmin", msc_ops.argmin)(*args, **kwargs)


def argpartition(*args, **kwargs):
    raise NotImplementedError(
        "ops.argpartition is not yet implemented in ml-switcheroo-compiler"
    )


def argsort(*args, **kwargs):
    raise NotImplementedError(
        "ops.argsort is not yet implemented in ml-switcheroo-compiler"
    )


arithmetic = msc_ops.arithmetic


def array(*args, **kwargs):
    return _wrap_op("array", msc_ops.array)(*args, **kwargs)


def array_split(*args, **kwargs):
    return _wrap_op("array_split", msc_ops.array_split)(*args, **kwargs)


def as_string(*args, **kwargs):
    return _wrap_op("as_string", msc_ops.as_string)(*args, **kwargs)


def asarray(*args, **kwargs):
    return _wrap_op("asarray", msc_ops.asarray)(*args, **kwargs)


def asin(*args, **kwargs):
    return _wrap_op("asin", msc_ops.asin)(*args, **kwargs)


def asinh(*args, **kwargs):
    return _wrap_op("asinh", msc_ops.asinh)(*args, **kwargs)


def assert_value(*args, **kwargs):
    return _wrap_op("assert_value", msc_ops.assert_value)(*args, **kwargs)


def assert_value_eager(*args, **kwargs):
    return _wrap_op("assert_value_eager", msc_ops.assert_value_eager)(*args, **kwargs)


def assert_value_tracing(*args, **kwargs):
    return _wrap_op("assert_value_tracing", msc_ops.assert_value_tracing)(
        *args, **kwargs
    )


def associative_scan(*args, **kwargs):
    return _wrap_op("associative_scan", msc_ops.associative_scan)(*args, **kwargs)


def atan(*args, **kwargs):
    return _wrap_op("atan", msc_ops.atan)(*args, **kwargs)


def atan2(*args, **kwargs):
    return _wrap_op("atan2", msc_ops.atan2)(*args, **kwargs)


def atanh(*args, **kwargs):
    return _wrap_op("atanh", msc_ops.atanh)(*args, **kwargs)


def atleast_1d(*args, **kwargs):
    return _wrap_op("atleast_1d", msc_ops.atleast_1d)(*args, **kwargs)


def atleast_2d(*args, **kwargs):
    return _wrap_op("atleast_2d", msc_ops.atleast_2d)(*args, **kwargs)


def atleast_3d(*args, **kwargs):
    return _wrap_op("atleast_3d", msc_ops.atleast_3d)(*args, **kwargs)


def augmix(*args, **kwargs):
    return _wrap_op("augmix", msc_ops.augmix)(*args, **kwargs)


def auto_contrast(*args, **kwargs):
    return _wrap_op("auto_contrast", msc_ops.auto_contrast)(*args, **kwargs)


def average(*args, **kwargs):
    raise NotImplementedError(
        "ops.average is not yet implemented in ml-switcheroo-compiler"
    )


def average_pool(*args, **kwargs):
    raise NotImplementedError(
        "ops.average_pool is not yet implemented in ml-switcheroo-compiler"
    )


def avg_pool(*args, **kwargs):
    return _wrap_op("avg_pool", msc_ops.avg_pool)(*args, **kwargs)


def avg_pool1d(*args, **kwargs):
    return _wrap_op("avg_pool1d", msc_ops.avg_pool1d)(*args, **kwargs)


def avg_pool2d(*args, **kwargs):
    return _wrap_op("avg_pool2d", msc_ops.avg_pool2d)(*args, **kwargs)


def avg_pool3d(*args, **kwargs):
    return _wrap_op("avg_pool3d", msc_ops.avg_pool3d)(*args, **kwargs)


def band_part(*args, **kwargs):
    return _wrap_op("band_part", msc_ops.band_part)(*args, **kwargs)


def banded_triangular_solve(*args, **kwargs):
    return _wrap_op("banded_triangular_solve", msc_ops.banded_triangular_solve)(
        *args, **kwargs
    )


def bartlett(*args, **kwargs):
    raise NotImplementedError(
        "ops.bartlett is not yet implemented in ml-switcheroo-compiler"
    )


base = msc_ops.base


def batch_normalization(*args, **kwargs):
    raise NotImplementedError(
        "ops.batch_normalization is not yet implemented in ml-switcheroo-compiler"
    )


def bessel_i0(*args, **kwargs):
    return _wrap_op("bessel_i0", msc_ops.bessel_i0)(*args, **kwargs)


def bessel_i0e(*args, **kwargs):
    return _wrap_op("bessel_i0e", msc_ops.bessel_i0e)(*args, **kwargs)


def bessel_i1(*args, **kwargs):
    return _wrap_op("bessel_i1", msc_ops.bessel_i1)(*args, **kwargs)


def bessel_i1e(*args, **kwargs):
    return _wrap_op("bessel_i1e", msc_ops.bessel_i1e)(*args, **kwargs)


def betainc(*args, **kwargs):
    return _wrap_op("betainc", msc_ops.betainc)(*args, **kwargs)


binary = msc_ops.binary


def binary_crossentropy(*args, **kwargs):
    raise NotImplementedError(
        "ops.binary_crossentropy is not yet implemented in ml-switcheroo-compiler"
    )


def bincount(*args, **kwargs):
    raise NotImplementedError(
        "ops.bincount is not yet implemented in ml-switcheroo-compiler"
    )


def bitcast(*args, **kwargs):
    return _wrap_op("bitcast", msc_ops.bitcast)(*args, **kwargs)


def bitwise_and(*args, **kwargs):
    return _wrap_op("bitwise_and", msc_ops.bitwise_and)(*args, **kwargs)


def bitwise_count(*args, **kwargs):
    return _wrap_op("bitwise_count", msc_ops.bitwise_count)(*args, **kwargs)


def bitwise_invert(*args, **kwargs):
    raise NotImplementedError(
        "ops.bitwise_invert is not yet implemented in ml-switcheroo-compiler"
    )


def bitwise_left_shift(*args, **kwargs):
    raise NotImplementedError(
        "ops.bitwise_left_shift is not yet implemented in ml-switcheroo-compiler"
    )


def bitwise_not(*args, **kwargs):
    return _wrap_op("bitwise_not", msc_ops.bitwise_not)(*args, **kwargs)


def bitwise_or(*args, **kwargs):
    return _wrap_op("bitwise_or", msc_ops.bitwise_or)(*args, **kwargs)


def bitwise_right_shift(*args, **kwargs):
    raise NotImplementedError(
        "ops.bitwise_right_shift is not yet implemented in ml-switcheroo-compiler"
    )


def bitwise_xor(*args, **kwargs):
    return _wrap_op("bitwise_xor", msc_ops.bitwise_xor)(*args, **kwargs)


def blackman(*args, **kwargs):
    raise NotImplementedError(
        "ops.blackman is not yet implemented in ml-switcheroo-compiler"
    )


def block(*args, **kwargs):
    return _wrap_op("block", msc_ops.block)(*args, **kwargs)


def block_masked_mm(*args, **kwargs):
    return _wrap_op("block_masked_mm", msc_ops.block_masked_mm)(*args, **kwargs)


boolean = msc_ops.boolean


def boolean_mask(*args, **kwargs):
    return _wrap_op("boolean_mask", msc_ops.boolean_mask)(*args, **kwargs)


def broadcast_arrays(*args, **kwargs):
    return _wrap_op("broadcast_arrays", msc_ops.broadcast_arrays)(*args, **kwargs)


def broadcast_to(*args, **kwargs):
    return _wrap_op("broadcast_to", msc_ops.broadcast_to)(*args, **kwargs)


def case(*args, **kwargs):
    return _wrap_op("case", msc_ops.case)(*args, **kwargs)


def cast(*args, **kwargs):
    return _wrap_op("cast", msc_ops.cast)(*args, **kwargs)


def categorical_crossentropy(*args, **kwargs):
    raise NotImplementedError(
        "ops.categorical_crossentropy is not yet implemented in ml-switcheroo-compiler"
    )


def cbrt(*args, **kwargs):
    return _wrap_op("cbrt", msc_ops.cbrt)(*args, **kwargs)


def ceil(*args, **kwargs):
    return _wrap_op("ceil", msc_ops.ceil)(*args, **kwargs)


def celu(*args, **kwargs):
    raise NotImplementedError(
        "ops.celu is not yet implemented in ml-switcheroo-compiler"
    )


def chebyshev_polynomial_t(*args, **kwargs):
    return _wrap_op("chebyshev_polynomial_t", msc_ops.chebyshev_polynomial_t)(
        *args, **kwargs
    )


def chebyshev_polynomial_u(*args, **kwargs):
    return _wrap_op("chebyshev_polynomial_u", msc_ops.chebyshev_polynomial_u)(
        *args, **kwargs
    )


def cholesky(*args, **kwargs):
    raise NotImplementedError(
        "ops.cholesky is not yet implemented in ml-switcheroo-compiler"
    )


def cholesky_solve(*args, **kwargs):
    return _wrap_op("cholesky_solve", msc_ops.cholesky_solve)(*args, **kwargs)


def clip(*args, **kwargs):
    return _wrap_op("clip", msc_ops.clip)(*args, **kwargs)


def column_stack(*args, **kwargs):
    return _wrap_op("column_stack", msc_ops.column_stack)(*args, **kwargs)


complex_types = msc_ops.complex_types

concat = msc_ops.concat


def concatenate(*args, **kwargs):
    return _wrap_op("concatenate", msc_ops.concatenate)(*args, **kwargs)


def cond(*args, **kwargs):
    return _wrap_op("cond", msc_ops.cond)(*args, **kwargs)


def cond_eager(*args, **kwargs):
    return _wrap_op("cond_eager", msc_ops.cond_eager)(*args, **kwargs)


def cond_tracing(*args, **kwargs):
    return _wrap_op("cond_tracing", msc_ops.cond_tracing)(*args, **kwargs)


config = msc_ops.config

configs = msc_ops.configs


def confusion_matrix(*args, **kwargs):
    return _wrap_op("confusion_matrix", msc_ops.confusion_matrix)(*args, **kwargs)


def conj(*args, **kwargs):
    return _wrap_op("conj", msc_ops.conj)(*args, **kwargs)


def conjugate(*args, **kwargs):
    raise NotImplementedError(
        "ops.conjugate is not yet implemented in ml-switcheroo-compiler"
    )


def conjugate_gradient(*args, **kwargs):
    return _wrap_op("conjugate_gradient", msc_ops.conjugate_gradient)(*args, **kwargs)


control_flow = msc_ops.control_flow

control_flow_utils = msc_ops.control_flow_utils


def conv(*args, **kwargs):
    return _wrap_op("conv", msc_ops.conv.conv)(*args, **kwargs)


from ml_switcheroo_compiler.ops.nn.conv1d import conv1d as msc_conv1d


def conv1d(*args, **kwargs):
    return _wrap_op("conv1d", msc_conv1d)(*args, **kwargs)


def conv1d_lstm_cell(*args, **kwargs):
    return _wrap_op("conv1d_lstm_cell", msc_ops.conv1d_lstm_cell)(*args, **kwargs)


from ml_switcheroo_compiler.ops.nn.conv_nd import conv_transpose as msc_conv_transpose


def conv1d_transpose(*args, **kwargs):
    return _wrap_op("conv1d_transpose", msc_conv_transpose)(*args, **kwargs)


from ml_switcheroo_compiler.ops.nn.conv2d import conv2d as msc_conv2d


def conv2d(*args, **kwargs):
    return _wrap_op("conv2d", msc_conv2d)(*args, **kwargs)


def conv2d_lstm_cell(*args, **kwargs):
    return _wrap_op("conv2d_lstm_cell", msc_ops.conv2d_lstm_cell)(*args, **kwargs)


def conv2d_transpose(*args, **kwargs):
    return _wrap_op("conv2d_transpose", msc_conv_transpose)(*args, **kwargs)


from ml_switcheroo_compiler.ops.nn.conv3d import conv3d as msc_conv3d


def conv3d(*args, **kwargs):
    return _wrap_op("conv3d", msc_conv3d)(*args, **kwargs)


def conv3d_lstm_cell(*args, **kwargs):
    return _wrap_op("conv3d_lstm_cell", msc_ops.conv3d_lstm_cell)(*args, **kwargs)


def conv3d_transpose(*args, **kwargs):
    return _wrap_op("conv3d_transpose", msc_conv_transpose)(*args, **kwargs)


def conv_general_dilated(*args, **kwargs):
    return _wrap_op("conv_general_dilated", msc_ops.conv_general_dilated)(
        *args, **kwargs
    )


def conv_general_dilated_local(*args, **kwargs):
    return _wrap_op("conv_general_dilated_local", msc_ops.conv_general_dilated_local)(
        *args, **kwargs
    )


def conv_general_dilated_patches(*args, **kwargs):
    return _wrap_op(
        "conv_general_dilated_patches", msc_ops.conv_general_dilated_patches
    )(*args, **kwargs)


conv_lstm = msc_ops.conv_lstm

conv_nd = msc_ops.conv_nd


def conv_transpose(*args, **kwargs):
    raise NotImplementedError(
        "ops.conv_transpose is not yet implemented in ml-switcheroo-compiler"
    )


conv_utils = msc_ops.conv_utils


def conv_with_general_padding(*args, **kwargs):
    return _wrap_op("conv_with_general_padding", msc_ops.conv_with_general_padding)(
        *args, **kwargs
    )


def convert_to_numpy(*args, **kwargs):
    return _wrap_op("convert_to_numpy", msc_ops.convert_to_numpy)(*args, **kwargs)


def convert_to_tensor(*args, **kwargs):
    return _wrap_op("convert_to_tensor", msc_ops.convert_to_tensor)(*args, **kwargs)


def convolve(*args, **kwargs):
    return _wrap_op("convolve", msc_ops.convolve)(*args, **kwargs)


def convolve2d(*args, **kwargs):
    return _wrap_op("convolve2d", msc_ops.convolve2d)(*args, **kwargs)


def copy(*args, **kwargs):
    raise NotImplementedError(
        "ops.copy is not yet implemented in ml-switcheroo-compiler"
    )


def copysign(*args, **kwargs):
    return _wrap_op("copysign", msc_ops.copysign)(*args, **kwargs)


core = msc_ops.core


def corrcoef(*args, **kwargs):
    return _wrap_op("corrcoef", msc_ops.corrcoef)(*args, **kwargs)


def correlate(*args, **kwargs):
    return _wrap_op("correlate", msc_ops.correlate)(*args, **kwargs)


def cos(*args, **kwargs):
    return _wrap_op("cos", msc_ops.cos)(*args, **kwargs)


def cosh(*args, **kwargs):
    return _wrap_op("cosh", msc_ops.cosh)(*args, **kwargs)


def count_nonzero(*args, **kwargs):
    return _wrap_op("count_nonzero", msc_ops.count_nonzero)(*args, **kwargs)


def cov(*args, **kwargs):
    return _wrap_op("cov", msc_ops.cov)(*args, **kwargs)


creation = msc_ops.creation


def crop(*args, **kwargs):
    return _wrap_op("crop", msc_ops.crop)(*args, **kwargs)


def crop_and_resize(*args, **kwargs):
    return _wrap_op("crop_and_resize", msc_ops.crop_and_resize)(*args, **kwargs)


def crop_images(*args, **kwargs):
    return _wrap_op("crop_images", msc_ops.crop_images)(*args, **kwargs)


def cross(*args, **kwargs):
    return _wrap_op("cross", msc_ops.cross)(*args, **kwargs)


def ctc_decode(*args, **kwargs):
    raise NotImplementedError(
        "ops.ctc_decode is not yet implemented in ml-switcheroo-compiler"
    )


def ctc_loss(*args, **kwargs):
    raise NotImplementedError(
        "ops.ctc_loss is not yet implemented in ml-switcheroo-compiler"
    )


def cummax(*args, **kwargs):
    return _wrap_op("cummax", msc_ops.cummax)(*args, **kwargs)


def cummin(*args, **kwargs):
    return _wrap_op("cummin", msc_ops.cummin)(*args, **kwargs)


def cumprod(*args, **kwargs):
    return _wrap_op("cumprod", msc_ops.cumprod)(*args, **kwargs)


def cumsum(*args, **kwargs):
    return _wrap_op("cumsum", msc_ops.cumsum)(*args, **kwargs)


def cumulative_logsumexp(*args, **kwargs):
    return _wrap_op("cumulative_logsumexp", msc_ops.cumulative_logsumexp)(
        *args, **kwargs
    )


def custom_gradient(*args, **kwargs):
    return _wrap_op("custom_gradient", msc_ops.custom_gradient)(*args, **kwargs)


def cutmix(*args, **kwargs):
    return _wrap_op("cutmix", msc_ops.cutmix)(*args, **kwargs)


def dataclass(*args, **kwargs):
    return _wrap_op("dataclass", msc_ops.dataclass)(*args, **kwargs)


def debug_infs(*args, **kwargs):
    return _wrap_op("debug_infs", msc_ops.debug_infs)(*args, **kwargs)


def debug_nans(*args, **kwargs):
    return _wrap_op("debug_nans", msc_ops.debug_nans)(*args, **kwargs)


def decode_base64(*args, **kwargs):
    return _wrap_op("decode_base64", msc_ops.decode_base64)(*args, **kwargs)


def decode_bmp(*args, **kwargs):
    return _wrap_op("decode_bmp", msc_ops.decode_bmp)(*args, **kwargs)


def decode_csv(*args, **kwargs):
    return _wrap_op("decode_csv", msc_ops.decode_csv)(*args, **kwargs)


def decode_gif(*args, **kwargs):
    return _wrap_op("decode_gif", msc_ops.decode_gif)(*args, **kwargs)


def decode_image(*args, **kwargs):
    return _wrap_op("decode_image", msc_ops.decode_image)(*args, **kwargs)


def decode_jpeg(*args, **kwargs):
    return _wrap_op("decode_jpeg", msc_ops.decode_jpeg)(*args, **kwargs)


def decode_png(*args, **kwargs):
    return _wrap_op("decode_png", msc_ops.decode_png)(*args, **kwargs)


def deg2rad(*args, **kwargs):
    return _wrap_op("deg2rad", msc_ops.deg2rad)(*args, **kwargs)


def degeneration(*args, **kwargs):
    return _wrap_op("degeneration", msc_ops.degeneration)(*args, **kwargs)


def degrees(*args, **kwargs):
    return _wrap_op("degrees", msc_ops.degrees)(*args, **kwargs)


def delete(*args, **kwargs):
    return _wrap_op("delete", msc_ops.delete)(*args, **kwargs)


def depthwise_conv(*args, **kwargs):
    raise NotImplementedError(
        "ops.depthwise_conv is not yet implemented in ml-switcheroo-compiler"
    )


def descriptive(*args, **kwargs):
    return _wrap_op("descriptive", msc_ops.descriptive.descriptive)(*args, **kwargs)


def det(*args, **kwargs):
    raise NotImplementedError(
        "ops.det is not yet implemented in ml-switcheroo-compiler"
    )


device = msc_ops.device


def diag(*args, **kwargs):
    return _wrap_op("diag", msc_ops.diag)(*args, **kwargs)


def diag_indices(*args, **kwargs):
    return _wrap_op("diag_indices", msc_ops.diag_indices)(*args, **kwargs)


def diag_indices_from(*args, **kwargs):
    return _wrap_op("diag_indices_from", msc_ops.diag_indices_from)(*args, **kwargs)


def diag_part(*args, **kwargs):
    return _wrap_op("diag_part", msc_ops.diag_part)(*args, **kwargs)


def diagflat(*args, **kwargs):
    return _wrap_op("diagflat", msc_ops.diagflat)(*args, **kwargs)


def diagonal(*args, **kwargs):
    return _wrap_op("diagonal", msc_ops.diagonal)(*args, **kwargs)


def diff(*args, **kwargs):
    raise NotImplementedError(
        "ops.diff is not yet implemented in ml-switcheroo-compiler"
    )


def digamma(*args, **kwargs):
    return _wrap_op("digamma", msc_ops.digamma)(*args, **kwargs)


def digitize(*args, **kwargs):
    raise NotImplementedError(
        "ops.digitize is not yet implemented in ml-switcheroo-compiler"
    )


def dispatch_eager(*args, **kwargs):
    return _wrap_op("dispatch_eager", msc_ops.dispatch_eager)(*args, **kwargs)


dispatcher = msc_ops.dispatcher

distributed = msc_ops.distributed

distributed_ops = msc_ops.distributed_ops

distributions = msc_ops.distributions


def divide(*args, **kwargs):
    return _wrap_op("divide", msc_ops.divide)(*args, **kwargs)


def divide_no_nan(*args, **kwargs):
    return _wrap_op("divide_no_nan", msc_ops.divide_no_nan)(*args, **kwargs)


def divmod(*args, **kwargs):
    return _wrap_op("divmod", msc_ops.divmod)(*args, **kwargs)


def dot(*args, **kwargs):
    return _wrap_op("dot", msc_ops.dot)(*args, **kwargs)


def dot_general(*args, **kwargs):
    return _wrap_op("dot_general", msc_ops.dot_general)(*args, **kwargs)


def dot_product_attention(*args, **kwargs):
    raise NotImplementedError(
        "ops.dot_product_attention is not yet implemented in ml-switcheroo-compiler"
    )


def draw_bounding_boxes(*args, **kwargs):
    return _wrap_op("draw_bounding_boxes", msc_ops.draw_bounding_boxes)(*args, **kwargs)


from ml_switcheroo_compiler.ops.nn.dropout import dropout as msc_dropout


def dropout(*args, **kwargs):
    return _wrap_op("dropout", msc_dropout)(*args, **kwargs)


def dsplit(*args, **kwargs):
    return _wrap_op("dsplit", msc_ops.dsplit)(*args, **kwargs)


def dstack(*args, **kwargs):
    return _wrap_op("dstack", msc_ops.dstack)(*args, **kwargs)


def dtype(*args, **kwargs):
    raise NotImplementedError(
        "ops.dtype is not yet implemented in ml-switcheroo-compiler"
    )


dynamic_slicing = msc_ops.dynamic_slicing

from ml_switcheroo_compiler.ops.shape.dynamic_slicing import (
    dynamic_update_slice as msc_dynamic_update_slice,
)


def dynamic_update_slice(*args, **kwargs):
    return _wrap_op("dynamic_update_slice", msc_dynamic_update_slice)(*args, **kwargs)


eager = msc_ops.eager

eager_evaluator = msc_ops.eager_evaluator


def ediff1d(*args, **kwargs):
    return _wrap_op("ediff1d", msc_ops.ediff1d)(*args, **kwargs)


def edit_distance(*args, **kwargs):
    return _wrap_op("edit_distance", msc_ops.edit_distance)(*args, **kwargs)


def eig(*args, **kwargs):
    raise NotImplementedError(
        "ops.eig is not yet implemented in ml-switcheroo-compiler"
    )


def eigh(*args, **kwargs):
    raise NotImplementedError(
        "ops.eigh is not yet implemented in ml-switcheroo-compiler"
    )


def eigh_tridiagonal(*args, **kwargs):
    return _wrap_op("eigh_tridiagonal", msc_ops.eigh_tridiagonal)(*args, **kwargs)


def einsum(*args, **kwargs):
    return _wrap_op("einsum", msc_ops.einsum)(*args, **kwargs)


def elastic_transform(*args, **kwargs):
    return _wrap_op("elastic_transform", msc_ops.elastic_transform)(*args, **kwargs)


def elu(*args, **kwargs):
    raise NotImplementedError(
        "ops.elu is not yet implemented in ml-switcheroo-compiler"
    )


def emit_ir_node(*args, **kwargs):
    return _wrap_op("emit_ir_node", msc_ops.emit_ir_node)(*args, **kwargs)


def empty(*args, **kwargs):
    return _wrap_op("empty", msc_ops.empty)(*args, **kwargs)


def empty_like(*args, **kwargs):
    return _wrap_op("empty_like", msc_ops.empty_like)(*args, **kwargs)


def encode_base64(*args, **kwargs):
    return _wrap_op("encode_base64", msc_ops.encode_base64)(*args, **kwargs)


def equal(*args, **kwargs):
    return _wrap_op("equal", msc_ops.equal)(*args, **kwargs)


def equalization(*args, **kwargs):
    return _wrap_op("equalization", msc_ops.equalization)(*args, **kwargs)


def erf(*args, **kwargs):
    return _wrap_op("erf", msc_ops.erf)(*args, **kwargs)


def erf_inv(*args, **kwargs):
    return _wrap_op("erf_inv", msc_ops.erf_inv)(*args, **kwargs)


def erfc(*args, **kwargs):
    return _wrap_op("erfc", msc_ops.erfc)(*args, **kwargs)


def erfcinv(*args, **kwargs):
    return _wrap_op("erfcinv", msc_ops.erfcinv)(*args, **kwargs)


def erfinv(*args, **kwargs):
    return _wrap_op("erfinv", msc_ops.erfinv)(*args, **kwargs)


def eval(*args, **kwargs):
    return _wrap_op("eval", msc_ops.eval)(*args, **kwargs)


def exp(*args, **kwargs):
    return _wrap_op("exp", msc_ops.exp)(*args, **kwargs)


def exp2(*args, **kwargs):
    return _wrap_op("exp2", msc_ops.exp2)(*args, **kwargs)


def expand_dims(*args, **kwargs):
    return _wrap_op("expand_dims", msc_ops.expand_dims)(*args, **kwargs)


def expm(*args, **kwargs):
    return _wrap_op("expm", msc_ops.expm)(*args, **kwargs)


def expm1(*args, **kwargs):
    return _wrap_op("expm1", msc_ops.expm1)(*args, **kwargs)


exponential = msc_ops.exponential


def extract_bounding_boxes(*args, **kwargs):
    return _wrap_op("extract_bounding_boxes", msc_ops.extract_bounding_boxes)(
        *args, **kwargs
    )


def extract_patches(*args, **kwargs):
    return _wrap_op("extract_patches", msc_ops.extract_patches)(*args, **kwargs)


def extract_sequences(*args, **kwargs):
    raise NotImplementedError(
        "ops.extract_sequences is not yet implemented in ml-switcheroo-compiler"
    )


def eye(*args, **kwargs):
    return _wrap_op("eye", msc_ops.eye)(*args, **kwargs)


def fft(*args, **kwargs):
    return _wrap_op("fft", msc_ops.fft)(*args, **kwargs)


def fft2(*args, **kwargs):
    raise NotImplementedError(
        "ops.fft2 is not yet implemented in ml-switcheroo-compiler"
    )


def fftconvolve(*args, **kwargs):
    return _wrap_op("fftconvolve", msc_ops.fftconvolve)(*args, **kwargs)


def fftn(*args, **kwargs):
    return _wrap_op("fftn", msc_ops.fftn)(*args, **kwargs)


def fftnd(*args, **kwargs):
    return _wrap_op("fftnd", msc_ops.fftnd)(*args, **kwargs)


def fftshift(*args, **kwargs):
    return _wrap_op("fftshift", msc_ops.fftshift)(*args, **kwargs)


def fill_diagonal(*args, **kwargs):
    return _wrap_op("fill_diagonal", msc_ops.fill_diagonal)(*args, **kwargs)


def fix(*args, **kwargs):
    return _wrap_op("fix", msc_ops.fix)(*args, **kwargs)


def flip(*args, **kwargs):
    return _wrap_op("flip", msc_ops.flip)(*args, **kwargs)


def flip_left_right(*args, **kwargs):
    return _wrap_op("flip_left_right", msc_ops.flip_left_right)(*args, **kwargs)


def flip_up_down(*args, **kwargs):
    return _wrap_op("flip_up_down", msc_ops.flip_up_down)(*args, **kwargs)


def fliplr(*args, **kwargs):
    return _wrap_op("fliplr", msc_ops.fliplr)(*args, **kwargs)


def flipud(*args, **kwargs):
    return _wrap_op("flipud", msc_ops.flipud)(*args, **kwargs)


def float_power(*args, **kwargs):
    return _wrap_op("float_power", msc_ops.float_power)(*args, **kwargs)


def floor(*args, **kwargs):
    return _wrap_op("floor", msc_ops.floor)(*args, **kwargs)


def floor_divide(*args, **kwargs):
    return _wrap_op("floor_divide", msc_ops.floor_divide)(*args, **kwargs)


def fmax(*args, **kwargs):
    return _wrap_op("fmax", msc_ops.fmax)(*args, **kwargs)


def fmin(*args, **kwargs):
    return _wrap_op("fmin", msc_ops.fmin)(*args, **kwargs)


def fmod(*args, **kwargs):
    return _wrap_op("fmod", msc_ops.fmod)(*args, **kwargs)


def fori_loop(*args, **kwargs):
    return _wrap_op("fori_loop", msc_ops.fori_loop)(*args, **kwargs)


def frexp(*args, **kwargs):
    return _wrap_op("frexp", msc_ops.frexp)(*args, **kwargs)


def frombuffer(*args, **kwargs):
    return _wrap_op("frombuffer", msc_ops.frombuffer)(*args, **kwargs)


frontend = msc_ops.frontend

frontend_pool = msc_ops.frontend_pool

frontend_segment = msc_ops.frontend_segment

frontend_stats = msc_ops.frontend_stats

frontend_utils = msc_ops.frontend_utils


def full(*args, **kwargs):
    return _wrap_op("full", msc_ops.full)(*args, **kwargs)


def full_like(*args, **kwargs):
    return _wrap_op("full_like", msc_ops.full_like)(*args, **kwargs)


def gather_mm(*args, **kwargs):
    return _wrap_op("gather_mm", msc_ops.gather_mm)(*args, **kwargs)


def gaussian_blur(*args, **kwargs):
    return _wrap_op("gaussian_blur", msc_ops.gaussian_blur)(*args, **kwargs)


def gcd(*args, **kwargs):
    return _wrap_op("gcd", msc_ops.gcd)(*args, **kwargs)


def gelu(*args, **kwargs):
    raise NotImplementedError(
        "ops.gelu is not yet implemented in ml-switcheroo-compiler"
    )


def get_active_backend(*args, **kwargs):
    return _wrap_op("get_active_backend", msc_ops.get_active_backend)(*args, **kwargs)


def get_item(*args, **kwargs):
    raise NotImplementedError(
        "ops.get_item is not yet implemented in ml-switcheroo-compiler"
    )


def get_op(*args, **kwargs):
    return _wrap_op("get_op", msc_ops.get_op)(*args, **kwargs)


def get_peak_memory(*args, **kwargs):
    return _wrap_op("get_peak_memory", msc_ops.get_peak_memory)(*args, **kwargs)


def gfile_copy(*args, **kwargs):
    return _wrap_op("gfile_copy", msc_ops.gfile_copy)(*args, **kwargs)


def gfile_glob(*args, **kwargs):
    return _wrap_op("gfile_glob", msc_ops.gfile_glob)(*args, **kwargs)


def gfile_makedirs(*args, **kwargs):
    return _wrap_op("gfile_makedirs", msc_ops.gfile_makedirs)(*args, **kwargs)


def gfile_stat(*args, **kwargs):
    return _wrap_op("gfile_stat", msc_ops.gfile_stat)(*args, **kwargs)


global_config = msc_ops.global_config


def global_norm(*args, **kwargs):
    return _wrap_op("global_norm", msc_ops.global_norm)(*args, **kwargs)


def glu(*args, **kwargs):
    raise NotImplementedError(
        "ops.glu is not yet implemented in ml-switcheroo-compiler"
    )


def greater(*args, **kwargs):
    return _wrap_op("greater", msc_ops.greater)(*args, **kwargs)


def greater_equal(*args, **kwargs):
    return _wrap_op("greater_equal", msc_ops.greater_equal)(*args, **kwargs)


def grid_sample(*args, **kwargs):
    return _wrap_op("grid_sample", msc_ops.grid_sample)(*args, **kwargs)


def group_variance(*args, **kwargs):
    return _wrap_op("group_variance", msc_ops.group_variance)(*args, **kwargs)


def gru(*args, **kwargs):
    return _wrap_op("gru", msc_ops.gru)(*args, **kwargs)


def gru_cell(*args, **kwargs):
    return _wrap_op("gru_cell", msc_ops.gru_cell)(*args, **kwargs)


def hadamard_transform(*args, **kwargs):
    return _wrap_op("hadamard_transform", msc_ops.hadamard_transform)(*args, **kwargs)


def hamming(*args, **kwargs):
    raise NotImplementedError(
        "ops.hamming is not yet implemented in ml-switcheroo-compiler"
    )


def hard_shrink(*args, **kwargs):
    raise NotImplementedError(
        "ops.hard_shrink is not yet implemented in ml-switcheroo-compiler"
    )


def hard_sigmoid(*args, **kwargs):
    raise NotImplementedError(
        "ops.hard_sigmoid is not yet implemented in ml-switcheroo-compiler"
    )


def hard_silu(*args, **kwargs):
    return _wrap_op("hard_silu", msc_ops.activations.hard_silu)(*args, **kwargs)


def hard_swish(*args, **kwargs):
    return _wrap_op("hard_swish", msc_ops.activations.hard_swish)(*args, **kwargs)


def hard_tanh(*args, **kwargs):
    raise NotImplementedError(
        "ops.hard_tanh is not yet implemented in ml-switcheroo-compiler"
    )


def heaviside(*args, **kwargs):
    return _wrap_op("heaviside", msc_ops.heaviside)(*args, **kwargs)


def hermite_polynomial_h(*args, **kwargs):
    return _wrap_op("hermite_polynomial_h", msc_ops.hermite_polynomial_h)(
        *args, **kwargs
    )


def hermite_polynomial_he(*args, **kwargs):
    return _wrap_op("hermite_polynomial_he", msc_ops.hermite_polynomial_he)(
        *args, **kwargs
    )


def hfft(*args, **kwargs):
    return _wrap_op("hfft", msc_ops.hfft)(*args, **kwargs)


def histogram(*args, **kwargs):
    raise NotImplementedError(
        "ops.histogram is not yet implemented in ml-switcheroo-compiler"
    )


def hsplit(*args, **kwargs):
    return _wrap_op("hsplit", msc_ops.hsplit)(*args, **kwargs)


def hstack(*args, **kwargs):
    return _wrap_op("hstack", msc_ops.hstack)(*args, **kwargs)


def hsv_to_rgb(*args, **kwargs):
    return _wrap_op("hsv_to_rgb", msc_ops.hsv_to_rgb)(*args, **kwargs)


hyperbolic = msc_ops.hyperbolic


def hypot(*args, **kwargs):
    return _wrap_op("hypot", msc_ops.hypot)(*args, **kwargs)


def identity(*args, **kwargs):
    return _wrap_op("identity", msc_ops.identity)(*args, **kwargs)


def ifft(*args, **kwargs):
    return _wrap_op("ifft", msc_ops.ifft)(*args, **kwargs)


def ifft2(*args, **kwargs):
    return _wrap_op("ifft2", msc_ops.ifft2)(*args, **kwargs)


def ifftn(*args, **kwargs):
    return _wrap_op("ifftn", msc_ops.ifftn)(*args, **kwargs)


def ifftnd(*args, **kwargs):
    return _wrap_op("ifftnd", msc_ops.ifftnd)(*args, **kwargs)


def ifftshift(*args, **kwargs):
    return _wrap_op("ifftshift", msc_ops.ifftshift)(*args, **kwargs)


def igamma(*args, **kwargs):
    return _wrap_op("igamma", msc_ops.igamma)(*args, **kwargs)


def igammac(*args, **kwargs):
    return _wrap_op("igammac", msc_ops.igammac)(*args, **kwargs)


def imag(*args, **kwargs):
    return _wrap_op("imag", msc_ops.imag)(*args, **kwargs)


class image:
    @staticmethod
    def resize(*args, **kwargs):
        return _wrap_op("resize", msc_ops.resize)(*args, **kwargs)

    @staticmethod
    def crop_images(*args, **kwargs):
        return _wrap_op("crop_images", msc_ops.crop_images)(*args, **kwargs)

    @staticmethod
    def extract_patches(*args, **kwargs):
        return _wrap_op("extract_patches", msc_ops.extract_patches)(*args, **kwargs)

    @staticmethod
    def affine_transform(*args, **kwargs):
        return _wrap_op("affine_transform", msc_ops.affine_transform)(*args, **kwargs)

    @staticmethod
    def elastic_transform(*args, **kwargs):
        return _wrap_op("elastic_transform", msc_ops.elastic_transform)(*args, **kwargs)

    @staticmethod
    def gaussian_blur(*args, **kwargs):
        return _wrap_op("gaussian_blur", msc_ops.gaussian_blur)(*args, **kwargs)

    @staticmethod
    def hsv_to_rgb(*args, **kwargs):
        return _wrap_op("hsv_to_rgb", msc_ops.hsv_to_rgb)(*args, **kwargs)

    @staticmethod
    def map_coordinates(*args, **kwargs):
        return _wrap_op("map_coordinates", msc_ops.map_coordinates)(*args, **kwargs)

    @staticmethod
    def pad_images(*args, **kwargs):
        return _wrap_op("pad_images", msc_ops.pad_images)(*args, **kwargs)

    @staticmethod
    def perspective_transform(*args, **kwargs):
        return _wrap_op("perspective_transform", msc_ops.perspective_transform)(
            *args, **kwargs
        )

    @staticmethod
    def rgb_to_grayscale(*args, **kwargs):
        return _wrap_op("rgb_to_grayscale", msc_ops.rgb_to_grayscale)(*args, **kwargs)

    @staticmethod
    def rgb_to_hsv(*args, **kwargs):
        return _wrap_op("rgb_to_hsv", msc_ops.rgb_to_hsv)(*args, **kwargs)

    @staticmethod
    def random_flip(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.affine import random_flip as rf

        return _wrap_op("random_flip", rf)(*args, **kwargs)

    @staticmethod
    def random_rotation(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.affine import random_rotation as rr

        return _wrap_op("random_rotation", rr)(*args, **kwargs)

    @staticmethod
    def adjust_contrast(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import adjust_contrast as ac

        return _wrap_op("adjust_contrast", ac)(*args, **kwargs)

    @staticmethod
    def auto_contrast(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import auto_contrast as ac

        return _wrap_op("auto_contrast", ac)(*args, **kwargs)

    @staticmethod
    def equalization(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.color import equalization as eq

        return _wrap_op("equalization", eq)(*args, **kwargs)

    @staticmethod
    def resize_bilinear(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import (
            resize_bilinear as rb,
        )

        return _wrap_op("resize_bilinear", rb)(*args, **kwargs)

    @staticmethod
    def resize_nearest(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import resize_nearest as rn

        return _wrap_op("resize_nearest", rn)(*args, **kwargs)

    @staticmethod
    def resize_bicubic(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import (
            resize_bicubic as rbc,
        )

        return _wrap_op("resize_bicubic", rbc)(*args, **kwargs)

    @staticmethod
    def resize_lanczos3(*args, **kwargs):
        from ml_switcheroo_compiler.ops.vision.interpolation import (
            resize_lanczos3 as rl3,
        )

        return _wrap_op("resize_lanczos3", rl3)(*args, **kwargs)


def in_top_k(*args, **kwargs):
    raise NotImplementedError(
        "ops.in_top_k is not yet implemented in ml-switcheroo-compiler"
    )


indexing = msc_ops.indexing

indexing_advanced = msc_ops.indexing_advanced


def inner(*args, **kwargs):
    return _wrap_op("inner", msc_ops.inner)(*args, **kwargs)


def insert(*args, **kwargs):
    return _wrap_op("insert", msc_ops.insert)(*args, **kwargs)


def inv(*args, **kwargs):
    raise NotImplementedError(
        "ops.inv is not yet implemented in ml-switcheroo-compiler"
    )


def invert(*args, **kwargs):
    return _wrap_op("invert", msc_ops.invert)(*args, **kwargs)


io = msc_ops.io


def iou(*args, **kwargs):
    return _wrap_op("iou", msc_ops.iou)(*args, **kwargs)


def irfft(*args, **kwargs):
    raise NotImplementedError(
        "ops.irfft is not yet implemented in ml-switcheroo-compiler"
    )


def irfft2(*args, **kwargs):
    return _wrap_op("irfft2", msc_ops.irfft2)(*args, **kwargs)


def irfftn(*args, **kwargs):
    return _wrap_op("irfftn", msc_ops.irfftn)(*args, **kwargs)


def irfftnd(*args, **kwargs):
    return _wrap_op("irfftnd", msc_ops.irfftnd)(*args, **kwargs)


def is_non_decreasing(*args, **kwargs):
    return _wrap_op("is_non_decreasing", msc_ops.is_non_decreasing)(*args, **kwargs)


def is_strictly_increasing(*args, **kwargs):
    return _wrap_op("is_strictly_increasing", msc_ops.is_strictly_increasing)(
        *args, **kwargs
    )


def is_tensor(*args, **kwargs):
    raise NotImplementedError(
        "ops.is_tensor is not yet implemented in ml-switcheroo-compiler"
    )


def isclose(*args, **kwargs):
    return _wrap_op("isclose", msc_ops.isclose)(*args, **kwargs)


def iscomplex(*args, **kwargs):
    return _wrap_op("iscomplex", msc_ops.iscomplex)(*args, **kwargs)


def iscomplexobj(*args, **kwargs):
    return _wrap_op("iscomplexobj", msc_ops.iscomplexobj)(*args, **kwargs)


def isfinite(*args, **kwargs):
    return _wrap_op("isfinite", msc_ops.isfinite)(*args, **kwargs)


def isin(*args, **kwargs):
    return _wrap_op("isin", msc_ops.isin)(*args, **kwargs)


def isinf(*args, **kwargs):
    return _wrap_op("isinf", msc_ops.isinf)(*args, **kwargs)


def isnan(*args, **kwargs):
    return _wrap_op("isnan", msc_ops.isnan)(*args, **kwargs)


def isneginf(*args, **kwargs):
    return _wrap_op("isneginf", msc_ops.isneginf)(*args, **kwargs)


def isposinf(*args, **kwargs):
    return _wrap_op("isposinf", msc_ops.isposinf)(*args, **kwargs)


def isreal(*args, **kwargs):
    return _wrap_op("isreal", msc_ops.isreal)(*args, **kwargs)


def isrealobj(*args, **kwargs):
    return _wrap_op("isrealobj", msc_ops.isrealobj)(*args, **kwargs)


def issubdtype(*args, **kwargs):
    return _wrap_op("issubdtype", msc_ops.issubdtype)(*args, **kwargs)


def istft(*args, **kwargs):
    return _wrap_op("istft", msc_ops.istft)(*args, **kwargs)


joining = msc_ops.joining


def laguerre_polynomial_l(*args, **kwargs):
    return _wrap_op("laguerre_polynomial_l", msc_ops.laguerre_polynomial_l)(
        *args, **kwargs
    )


def lbeta(*args, **kwargs):
    return _wrap_op("lbeta", msc_ops.lbeta)(*args, **kwargs)


def lcm(*args, **kwargs):
    return _wrap_op("lcm", msc_ops.lcm)(*args, **kwargs)


def ldexp(*args, **kwargs):
    return _wrap_op("ldexp", msc_ops.ldexp)(*args, **kwargs)


def leaky_relu(*args, **kwargs):
    raise NotImplementedError(
        "ops.leaky_relu is not yet implemented in ml-switcheroo-compiler"
    )


def left_shift(*args, **kwargs):
    return _wrap_op("left_shift", msc_ops.left_shift)(*args, **kwargs)


def legendre_polynomial_p(*args, **kwargs):
    return _wrap_op("legendre_polynomial_p", msc_ops.legendre_polynomial_p)(
        *args, **kwargs
    )


def less(*args, **kwargs):
    return _wrap_op("less", msc_ops.less)(*args, **kwargs)


def less_equal(*args, **kwargs):
    return _wrap_op("less_equal", msc_ops.less_equal)(*args, **kwargs)


def lgamma(*args, **kwargs):
    return _wrap_op("lgamma", msc_ops.lgamma)(*args, **kwargs)


linalg = msc_ops.linalg


def linspace(*args, **kwargs):
    return _wrap_op("linspace", msc_ops.linspace)(*args, **kwargs)


def load(*args, **kwargs):
    return _wrap_op("load", msc_ops.load)(*args, **kwargs)


def log(*args, **kwargs):
    return _wrap_op("log", msc_ops.log)(*args, **kwargs)


def log10(*args, **kwargs):
    return _wrap_op("log10", msc_ops.log10)(*args, **kwargs)


def log1p(*args, **kwargs):
    return _wrap_op("log1p", msc_ops.log1p)(*args, **kwargs)


def log2(*args, **kwargs):
    return _wrap_op("log2", msc_ops.log2)(*args, **kwargs)


def log_sigmoid(*args, **kwargs):
    raise NotImplementedError(
        "ops.log_sigmoid is not yet implemented in ml-switcheroo-compiler"
    )


def log_softmax(*args, **kwargs):
    return _wrap_op("log_softmax", msc_ops.activations.log_softmax)(*args, **kwargs)


def logaddexp(*args, **kwargs):
    return _wrap_op("logaddexp", msc_ops.logaddexp)(*args, **kwargs)


def logaddexp2(*args, **kwargs):
    return _wrap_op("logaddexp2", msc_ops.logaddexp2)(*args, **kwargs)


def logcumsumexp(*args, **kwargs):
    return _wrap_op("logcumsumexp", msc_ops.logcumsumexp)(*args, **kwargs)


def logdet(*args, **kwargs):
    return _wrap_op("logdet", msc_ops.logdet)(*args, **kwargs)


logical = msc_ops.logical


def logical_and(*args, **kwargs):
    return _wrap_op("logical_and", msc_ops.logical_and)(*args, **kwargs)


def logical_not(*args, **kwargs):
    return _wrap_op("logical_not", msc_ops.logical_not)(*args, **kwargs)


def logical_or(*args, **kwargs):
    return _wrap_op("logical_or", msc_ops.logical_or)(*args, **kwargs)


def logical_xor(*args, **kwargs):
    return _wrap_op("logical_xor", msc_ops.logical_xor)(*args, **kwargs)


def logit(*args, **kwargs):
    return _wrap_op("logit", msc_ops.logit)(*args, **kwargs)


def logm(*args, **kwargs):
    return _wrap_op("logm", msc_ops.logm)(*args, **kwargs)


def logspace(*args, **kwargs):
    raise NotImplementedError(
        "ops.logspace is not yet implemented in ml-switcheroo-compiler"
    )


def logsumexp(*args, **kwargs):
    return _wrap_op("logsumexp", msc_ops.logsumexp)(*args, **kwargs)


def lookup(*args, **kwargs):
    return _wrap_op("lookup", msc_ops.lookup)(*args, **kwargs)


loss = msc_ops.loss

lstm = msc_ops.lstm


def lstm_cell(*args, **kwargs):
    return _wrap_op("lstm_cell", msc_ops.lstm_cell)(*args, **kwargs)


def lstsq(*args, **kwargs):
    return _wrap_op("lstsq", msc_ops.lstsq)(*args, **kwargs)


def lu(*args, **kwargs):
    return _wrap_op("lu", msc_ops.lu)(*args, **kwargs)


def lu_factor(*args, **kwargs):
    raise NotImplementedError(
        "ops.lu_factor is not yet implemented in ml-switcheroo-compiler"
    )


def lu_matrix_inverse(*args, **kwargs):
    return _wrap_op("lu_matrix_inverse", msc_ops.lu_matrix_inverse)(*args, **kwargs)


def lu_reconstruct(*args, **kwargs):
    return _wrap_op("lu_reconstruct", msc_ops.lu_reconstruct)(*args, **kwargs)


def lu_solve(*args, **kwargs):
    return _wrap_op("lu_solve", msc_ops.lu_solve)(*args, **kwargs)


def manual_seed(*args, **kwargs):
    return _wrap_op("manual_seed", msc_ops.manual_seed)(*args, **kwargs)


def map(*args, **kwargs):
    return _wrap_op("map", msc_ops.map)(*args, **kwargs)


def map_coordinates(*args, **kwargs):
    return _wrap_op("map_coordinates", msc_ops.map_coordinates)(*args, **kwargs)


def map_flat_values(*args, **kwargs):
    return _wrap_op("map_flat_values", msc_ops.map_flat_values)(*args, **kwargs)


def map_fn(*args, **kwargs):
    return _wrap_op("map_fn", msc_ops.map_fn)(*args, **kwargs)


def map_fn_eager(*args, **kwargs):
    return _wrap_op("map_fn_eager", msc_ops.map_fn_eager)(*args, **kwargs)


def map_fn_tracing(*args, **kwargs):
    return _wrap_op("map_fn_tracing", msc_ops.map_fn_tracing)(*args, **kwargs)


math = msc_ops.math


def matmul(*args, **kwargs):
    return _wrap_op("matmul", msc_ops.matmul)(*args, **kwargs)


def matrix_norm(*args, **kwargs):
    return _wrap_op("matrix_norm", msc_ops.matrix_norm)(*args, **kwargs)


def matrix_rank(*args, **kwargs):
    return _wrap_op("matrix_rank", msc_ops.matrix_rank)(*args, **kwargs)


def matrix_transpose(*args, **kwargs):
    return _wrap_op("matrix_transpose", msc_ops.matrix_transpose)(*args, **kwargs)


def matvec(*args, **kwargs):
    return _wrap_op("matvec", msc_ops.matvec)(*args, **kwargs)


def max(*args, **kwargs):
    return _wrap_op("max", msc_ops.max)(*args, **kwargs)


def max_pool(*args, **kwargs):
    return _wrap_op("max_pool", msc_ops.max_pool)(*args, **kwargs)


def max_pool1d(*args, **kwargs):
    return _wrap_op("max_pool1d", msc_ops.max_pool1d)(*args, **kwargs)


def max_pool2d(*args, **kwargs):
    return _wrap_op("max_pool2d", msc_ops.max_pool2d)(*args, **kwargs)


def max_pool3d(*args, **kwargs):
    return _wrap_op("max_pool3d", msc_ops.max_pool3d)(*args, **kwargs)


def maximum(*args, **kwargs):
    return _wrap_op("maximum", msc_ops.maximum)(*args, **kwargs)


def mean(*args, **kwargs):
    return _wrap_op("mean", msc_ops.mean)(*args, **kwargs)


def median(*args, **kwargs):
    raise NotImplementedError(
        "ops.median is not yet implemented in ml-switcheroo-compiler"
    )


def median_filter(*args, **kwargs):
    return _wrap_op("median_filter", msc_ops.median_filter)(*args, **kwargs)


def meshgrid(*args, **kwargs):
    raise NotImplementedError(
        "ops.meshgrid is not yet implemented in ml-switcheroo-compiler"
    )


def min(*args, **kwargs):
    return _wrap_op("min", msc_ops.min)(*args, **kwargs)


def minimum(*args, **kwargs):
    return _wrap_op("minimum", msc_ops.minimum)(*args, **kwargs)


misc = msc_ops.misc


def mixup(*args, **kwargs):
    return _wrap_op("mixup", msc_ops.mixup)(*args, **kwargs)


def mod(*args, **kwargs):
    return _wrap_op("mod", msc_ops.mod)(*args, **kwargs)


def modified_bessel_i0(*args, **kwargs):
    return _wrap_op("modified_bessel_i0", msc_ops.modified_bessel_i0)(*args, **kwargs)


def modified_bessel_i1(*args, **kwargs):
    return _wrap_op("modified_bessel_i1", msc_ops.modified_bessel_i1)(*args, **kwargs)


def modified_bessel_k0(*args, **kwargs):
    return _wrap_op("modified_bessel_k0", msc_ops.modified_bessel_k0)(*args, **kwargs)


def modified_bessel_k1(*args, **kwargs):
    return _wrap_op("modified_bessel_k1", msc_ops.modified_bessel_k1)(*args, **kwargs)


def moments(*args, **kwargs):
    raise NotImplementedError(
        "ops.moments is not yet implemented in ml-switcheroo-compiler"
    )


def moveaxis(*args, **kwargs):
    return _wrap_op("moveaxis", msc_ops.moveaxis)(*args, **kwargs)


def multi_dot(*args, **kwargs):
    return _wrap_op("multi_dot", msc_ops.multi_dot)(*args, **kwargs)


def multi_hot(*args, **kwargs):
    raise NotImplementedError(
        "ops.multi_hot is not yet implemented in ml-switcheroo-compiler"
    )


def multiply(*args, **kwargs):
    return _wrap_op("multiply", msc_ops.multiply)(*args, **kwargs)


def multiply_no_nan(*args, **kwargs):
    return _wrap_op("multiply_no_nan", msc_ops.multiply_no_nan)(*args, **kwargs)


def mvlgamma(*args, **kwargs):
    return _wrap_op("mvlgamma", msc_ops.mvlgamma)(*args, **kwargs)


nan = msc_ops.nan


def nan_to_num(*args, **kwargs):
    return _wrap_op("nan_to_num", msc_ops.nan_to_num)(*args, **kwargs)


def nanargmax(*args, **kwargs):
    return _wrap_op("nanargmax", msc_ops.nanargmax)(*args, **kwargs)


def nanargmin(*args, **kwargs):
    return _wrap_op("nanargmin", msc_ops.nanargmin)(*args, **kwargs)


def nancumprod(*args, **kwargs):
    return _wrap_op("nancumprod", msc_ops.nancumprod)(*args, **kwargs)


def nancumsum(*args, **kwargs):
    return _wrap_op("nancumsum", msc_ops.nancumsum)(*args, **kwargs)


def nanmax(*args, **kwargs):
    return _wrap_op("nanmax", msc_ops.nanmax)(*args, **kwargs)


def nanmean(*args, **kwargs):
    return _wrap_op("nanmean", msc_ops.nanmean)(*args, **kwargs)


def nanmedian(*args, **kwargs):
    return _wrap_op("nanmedian", msc_ops.nanmedian)(*args, **kwargs)


def nanmin(*args, **kwargs):
    return _wrap_op("nanmin", msc_ops.nanmin)(*args, **kwargs)


def nanpercentile(*args, **kwargs):
    return _wrap_op("nanpercentile", msc_ops.nanpercentile)(*args, **kwargs)


def nanprod(*args, **kwargs):
    return _wrap_op("nanprod", msc_ops.nanprod)(*args, **kwargs)


def nanquantile(*args, **kwargs):
    return _wrap_op("nanquantile", msc_ops.nanquantile)(*args, **kwargs)


def nanstd(*args, **kwargs):
    return _wrap_op("nanstd", msc_ops.nanstd)(*args, **kwargs)


def nansum(*args, **kwargs):
    return _wrap_op("nansum", msc_ops.nansum)(*args, **kwargs)


def nanvar(*args, **kwargs):
    return _wrap_op("nanvar", msc_ops.nanvar)(*args, **kwargs)


def ndim(*args, **kwargs):
    raise NotImplementedError(
        "ops.ndim is not yet implemented in ml-switcheroo-compiler"
    )


def ndtri(*args, **kwargs):
    return _wrap_op("ndtri", msc_ops.ndtri)(*args, **kwargs)


def negative(*args, **kwargs):
    return _wrap_op("negative", msc_ops.negative)(*args, **kwargs)


newaxis = None


def nextafter(*args, **kwargs):
    return _wrap_op("nextafter", msc_ops.nextafter)(*args, **kwargs)


nlp = msc_ops.nlp

nn = msc_ops.nn


def non_max_suppression(*args, **kwargs):
    return _wrap_op("non_max_suppression", msc_ops.non_max_suppression)(*args, **kwargs)


def nonzero(*args, **kwargs):
    raise NotImplementedError(
        "ops.nonzero is not yet implemented in ml-switcheroo-compiler"
    )


def norm(*args, **kwargs):
    return _wrap_op("norm", msc_ops.norm)(*args, **kwargs)


normalization = msc_ops.normalization


def normalize(*args, **kwargs):
    return _wrap_op("normalize", msc_ops.normalize)(*args, **kwargs)


def not_equal(*args, **kwargs):
    return _wrap_op("not_equal", msc_ops.not_equal)(*args, **kwargs)


def numpy(*args, **kwargs):
    raise NotImplementedError(
        "ops.numpy is not yet implemented in ml-switcheroo-compiler"
    )


def one_hot(*args, **kwargs):
    return _wrap_op("one_hot", msc_ops.activations.one_hot)(*args, **kwargs)


def ones(*args, **kwargs):
    return _wrap_op("ones", msc_ops.ones)(*args, **kwargs)


def ones_like(*args, **kwargs):
    return _wrap_op("ones_like", msc_ops.ones_like)(*args, **kwargs)


def outer(*args, **kwargs):
    return _wrap_op("outer", msc_ops.outer)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.misc import pad as msc_pad


def pad(*args, **kwargs):
    return _wrap_op("pad", msc_pad)(*args, **kwargs)


def pad_images(*args, **kwargs):
    return _wrap_op("pad_images", msc_ops.pad_images)(*args, **kwargs)


def pad_to_bounding_box(*args, **kwargs):
    return _wrap_op("pad_to_bounding_box", msc_ops.pad_to_bounding_box)(*args, **kwargs)


def parse_example(*args, **kwargs):
    return _wrap_op("parse_example", msc_ops.parse_example)(*args, **kwargs)


def parse_sequence_example(*args, **kwargs):
    return _wrap_op("parse_sequence_example", msc_ops.parse_sequence_example)(
        *args, **kwargs
    )


def parse_tensor(*args, **kwargs):
    return _wrap_op("parse_tensor", msc_ops.parse_tensor)(*args, **kwargs)


def permute(*args, **kwargs):
    return _wrap_op("permute", msc_ops.permute)(*args, **kwargs)


def perspective_transform(*args, **kwargs):
    return _wrap_op("perspective_transform", msc_ops.perspective_transform)(
        *args, **kwargs
    )


def pmap(*args, **kwargs):
    return _wrap_op("pmap", msc_ops.pmap)(*args, **kwargs)


def pmap_eager(*args, **kwargs):
    return _wrap_op("pmap_eager", msc_ops.pmap_eager)(*args, **kwargs)


def pmap_tracing(*args, **kwargs):
    return _wrap_op("pmap_tracing", msc_ops.pmap_tracing)(*args, **kwargs)


def pmean(*args, **kwargs):
    return _wrap_op("pmean", msc_ops.pmean)(*args, **kwargs)


def polar(*args, **kwargs):
    return _wrap_op("polar", msc_ops.polar)(*args, **kwargs)


def polygamma(*args, **kwargs):
    return _wrap_op("polygamma", msc_ops.polygamma)(*args, **kwargs)


pooling = msc_ops.pooling


def positive(*args, **kwargs):
    return _wrap_op("positive", msc_ops.positive)(*args, **kwargs)


def posterize(*args, **kwargs):
    return _wrap_op("posterize", msc_ops.posterize)(*args, **kwargs)


def power(*args, **kwargs):
    return _wrap_op("power", msc_ops.power)(*args, **kwargs)


def power_iteration(*args, **kwargs):
    return _wrap_op("power_iteration", msc_ops.power_iteration)(*args, **kwargs)


def prod(*args, **kwargs):
    return _wrap_op("prod", msc_ops.prod)(*args, **kwargs)


def psnr(*args, **kwargs):
    raise NotImplementedError(
        "ops.psnr is not yet implemented in ml-switcheroo-compiler"
    )


def psum(*args, **kwargs):
    return _wrap_op("psum", msc_ops.psum)(*args, **kwargs)


from ml_switcheroo_compiler.ops.linalg.decompositions.qr import qr as msc_qr


def qr(*args, **kwargs):
    return _wrap_op("qr", msc_qr)(*args, **kwargs)


def quantile(*args, **kwargs):
    raise NotImplementedError(
        "ops.quantile is not yet implemented in ml-switcheroo-compiler"
    )


def rad2deg(*args, **kwargs):
    return _wrap_op("rad2deg", msc_ops.rad2deg)(*args, **kwargs)


def radians(*args, **kwargs):
    return _wrap_op("radians", msc_ops.radians)(*args, **kwargs)


ragged = msc_ops.ragged


def ragged_tensor_to_dense(*args, **kwargs):
    return _wrap_op("ragged_tensor_to_dense", msc_ops.ragged_tensor_to_dense)(
        *args, **kwargs
    )


def rand(*args, **kwargs):
    return _wrap_op("rand", msc_ops.rand)(*args, **kwargs)


def rand_augment(*args, **kwargs):
    return _wrap_op("rand_augment", msc_ops.rand_augment)(*args, **kwargs)


def randint(*args, **kwargs):
    return _wrap_op("randint", msc_ops.randint)(*args, **kwargs)


def randn(*args, **kwargs):
    return _wrap_op("randn", msc_ops.randn)(*args, **kwargs)


def random_color_jitter(*args, **kwargs):
    return _wrap_op("random_color_jitter", msc_ops.random_color_jitter)(*args, **kwargs)


def random_crop(*args, **kwargs):
    return _wrap_op("random_crop", msc_ops.random_crop)(*args, **kwargs)


def random_elastic_transform(*args, **kwargs):
    return _wrap_op("random_elastic_transform", msc_ops.random_elastic_transform)(
        *args, **kwargs
    )


def random_erasing(*args, **kwargs):
    return _wrap_op("random_erasing", msc_ops.random_erasing)(*args, **kwargs)


def random_flip(*args, **kwargs):
    return _wrap_op("random_flip", msc_ops.random_flip)(*args, **kwargs)


def random_gaussian_blur(*args, **kwargs):
    return _wrap_op("random_gaussian_blur", msc_ops.random_gaussian_blur)(
        *args, **kwargs
    )


random_ops = msc_ops.random_ops


def random_perspective(*args, **kwargs):
    return _wrap_op("random_perspective", msc_ops.random_perspective)(*args, **kwargs)


def random_rotation(*args, **kwargs):
    return _wrap_op("random_rotation", msc_ops.random_rotation)(*args, **kwargs)


def random_sharpness(*args, **kwargs):
    return _wrap_op("random_sharpness", msc_ops.random_sharpness)(*args, **kwargs)


def random_shear(*args, **kwargs):
    return _wrap_op("random_shear", msc_ops.random_shear)(*args, **kwargs)


def random_translation(*args, **kwargs):
    return _wrap_op("random_translation", msc_ops.random_translation)(*args, **kwargs)


def random_zoom(*args, **kwargs):
    return _wrap_op("random_zoom", msc_ops.random_zoom)(*args, **kwargs)


def ravel(*args, **kwargs):
    raise NotImplementedError(
        "ops.ravel is not yet implemented in ml-switcheroo-compiler"
    )


raw_ops = msc_ops.raw_ops


def read_file(*args, **kwargs):
    return _wrap_op("read_file", msc_ops.read_file)(*args, **kwargs)


def real(*args, **kwargs):
    return _wrap_op("real", msc_ops.real)(*args, **kwargs)


def rearrange(*args, **kwargs):
    raise NotImplementedError(
        "ops.rearrange is not yet implemented in ml-switcheroo-compiler"
    )


def reciprocal(*args, **kwargs):
    return _wrap_op("reciprocal", msc_ops.reciprocal)(*args, **kwargs)


def reciprocal_no_nan(*args, **kwargs):
    return _wrap_op("reciprocal_no_nan", msc_ops.reciprocal_no_nan)(*args, **kwargs)


def reduce_euclidean_norm(*args, **kwargs):
    return _wrap_op("reduce_euclidean_norm", msc_ops.reduce_euclidean_norm)(
        *args, **kwargs
    )


def reduce_logsumexp(*args, **kwargs):
    return _wrap_op("reduce_logsumexp", msc_ops.reduce_logsumexp)(*args, **kwargs)


def reduce_window(*args, **kwargs):
    return _wrap_op("reduce_window", msc_ops.reduce_window)(*args, **kwargs)


reductions = msc_ops.reductions


def regex_full_match(*args, **kwargs):
    return _wrap_op("regex_full_match", msc_ops.regex_full_match)(*args, **kwargs)


def regex_replace(*args, **kwargs):
    return _wrap_op("regex_replace", msc_ops.regex_replace)(*args, **kwargs)


def register_op(*args, **kwargs):
    return _wrap_op("register_op", msc_ops.register_op)(*args, **kwargs)


registry = msc_ops.registry


def relu(*args, **kwargs):
    raise NotImplementedError(
        "ops.relu is not yet implemented in ml-switcheroo-compiler"
    )


def relu6(*args, **kwargs):
    raise NotImplementedError(
        "ops.relu6 is not yet implemented in ml-switcheroo-compiler"
    )


def rem(*args, **kwargs):
    return _wrap_op("rem", msc_ops.rem)(*args, **kwargs)


def remainder(*args, **kwargs):
    return _wrap_op("remainder", msc_ops.remainder)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.misc import repeat as msc_repeat


def repeat(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("repeat", msc_repeat)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.frontend import reshape as msc_reshape


def reshape(*args, **kwargs):
    return _wrap_op("reshape", msc_reshape)(*args, **kwargs)


def resize(*args, **kwargs):
    return _wrap_op("resize", msc_ops.resize)(*args, **kwargs)


def resize_bicubic(*args, **kwargs):
    return _wrap_op("resize_bicubic", msc_ops.resize_bicubic)(*args, **kwargs)


def resize_bilinear(*args, **kwargs):
    return _wrap_op("resize_bilinear", msc_ops.resize_bilinear)(*args, **kwargs)


def resize_lanczos3(*args, **kwargs):
    return _wrap_op("resize_lanczos3", msc_ops.resize_lanczos3)(*args, **kwargs)


def resize_nearest(*args, **kwargs):
    return _wrap_op("resize_nearest", msc_ops.resize_nearest)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.frontend import flip as msc_flip


def reverse(*args, **kwargs):
    if "dims" in kwargs:
        kwargs["axis"] = kwargs.pop("dims")
    elif "axes" in kwargs:
        kwargs["axis"] = kwargs.pop("axes")
    return _wrap_op("reverse", msc_flip)(*args, **kwargs)


def rfft(*args, **kwargs):
    raise NotImplementedError(
        "ops.rfft is not yet implemented in ml-switcheroo-compiler"
    )


def rfft2(*args, **kwargs):
    return _wrap_op("rfft2", msc_ops.rfft2)(*args, **kwargs)


def rfftfreq(*args, **kwargs):
    return _wrap_op("rfftfreq", msc_ops.rfftfreq)(*args, **kwargs)


def rfftn(*args, **kwargs):
    return _wrap_op("rfftn", msc_ops.rfftn)(*args, **kwargs)


def rfftnd(*args, **kwargs):
    return _wrap_op("rfftnd", msc_ops.rfftnd)(*args, **kwargs)


def rgb_to_grayscale(*args, **kwargs):
    return _wrap_op("rgb_to_grayscale", msc_ops.rgb_to_grayscale)(*args, **kwargs)


def rgb_to_hsv(*args, **kwargs):
    return _wrap_op("rgb_to_hsv", msc_ops.rgb_to_hsv)(*args, **kwargs)


def rgb_to_yiq(*args, **kwargs):
    return _wrap_op("rgb_to_yiq", msc_ops.rgb_to_yiq)(*args, **kwargs)


def rgb_to_yuv(*args, **kwargs):
    return _wrap_op("rgb_to_yuv", msc_ops.rgb_to_yuv)(*args, **kwargs)


def right_shift(*args, **kwargs):
    return _wrap_op("right_shift", msc_ops.right_shift)(*args, **kwargs)


def rint(*args, **kwargs):
    return _wrap_op("rint", msc_ops.rint)(*args, **kwargs)


def rms_normalization(*args, **kwargs):
    raise NotImplementedError(
        "ops.rms_normalization is not yet implemented in ml-switcheroo-compiler"
    )


from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    RNNConfig as MSC_RNNConfig,
)
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    rnn as msc_rnn,
)


def rnn(
    inputs,
    initial_state,
    cell_fn,
    go_backwards=False,
    mask=None,
    unroll=False,
    input_length=None,
    time_major=False,
    return_all_outputs=True,
    zero_output_for_mask=False,
    **kwargs,
):
    config = MSC_RNNConfig(
        time_major=time_major,
        go_backwards=go_backwards,
        unroll=unroll,
        return_all_outputs=return_all_outputs,
    )
    return _wrap_op("rnn", msc_rnn)(inputs, initial_state, cell_fn, config)


rnn_cell = msc_ops.rnn_cell

rnn_utils = msc_ops.rnn_utils


def roll(*args, **kwargs):
    return _wrap_op("roll", msc_ops.roll)(*args, **kwargs)


def rot90(*args, **kwargs):
    raise NotImplementedError(
        "ops.rot90 is not yet implemented in ml-switcheroo-compiler"
    )


def round(*args, **kwargs):
    return _wrap_op("round", msc_ops.round)(*args, **kwargs)


def rsqrt(*args, **kwargs):
    return _wrap_op("rsqrt", msc_ops.rsqrt)(*args, **kwargs)


def saturate_cast(*args, **kwargs):
    raise NotImplementedError(
        "ops.saturate_cast is not yet implemented in ml-switcheroo-compiler"
    )


def save(*args, **kwargs):
    return _wrap_op("save", msc_ops.save)(*args, **kwargs)


def save_gguf(*args, **kwargs):
    return _wrap_op("save_gguf", msc_ops.save_gguf)(*args, **kwargs)


def save_safetensors(*args, **kwargs):
    return _wrap_op("save_safetensors", msc_ops.save_safetensors)(*args, **kwargs)


def savez(*args, **kwargs):
    return _wrap_op("savez", msc_ops.savez)(*args, **kwargs)


def savez_compressed(*args, **kwargs):
    return _wrap_op("savez_compressed", msc_ops.savez_compressed)(*args, **kwargs)


def scalar_mul(*args, **kwargs):
    raise NotImplementedError(
        "ops.scalar_mul is not yet implemented in ml-switcheroo-compiler"
    )


def scan(*args, **kwargs):
    return _wrap_op("scan", msc_ops.scan)(*args, **kwargs)


def scan_bind(*args, **kwargs):
    return _wrap_op("scan_bind", msc_ops.scan_bind)(*args, **kwargs)


def scan_eager(*args, **kwargs):
    return _wrap_op("scan_eager", msc_ops.scan_eager)(*args, **kwargs)


def scan_tracing(*args, **kwargs):
    return _wrap_op("scan_tracing", msc_ops.scan_tracing)(*args, **kwargs)


def scatter(*args, **kwargs):
    return _wrap_op("scatter", msc_ops.scatter.scatter)(*args, **kwargs)


def scatter_update(*args, **kwargs):
    raise NotImplementedError(
        "ops.scatter_update is not yet implemented in ml-switcheroo-compiler"
    )


from ml_switcheroo_compiler.ops.shape.indexing import searchsorted as msc_searchsorted


def searchsorted(*args, **kwargs):
    return _wrap_op("searchsorted", msc_searchsorted)(*args, **kwargs)


def segment_max(*args, **kwargs):
    return _wrap_op("segment_max", msc_ops.segment_max)(*args, **kwargs)


def segment_mean(*args, **kwargs):
    return _wrap_op("segment_mean", msc_ops.segment_mean)(*args, **kwargs)


def segment_min(*args, **kwargs):
    return _wrap_op("segment_min", msc_ops.segment_min)(*args, **kwargs)


def segment_prod(*args, **kwargs):
    return _wrap_op("segment_prod", msc_ops.segment_prod)(*args, **kwargs)


def segment_sum(*args, **kwargs):
    return _wrap_op("segment_sum", msc_ops.segment_sum)(*args, **kwargs)


def segmented_mm(*args, **kwargs):
    return _wrap_op("segmented_mm", msc_ops.segmented_mm)(*args, **kwargs)


def select(*args, **kwargs):
    raise NotImplementedError(
        "ops.select is not yet implemented in ml-switcheroo-compiler"
    )


def selu(*args, **kwargs):
    raise NotImplementedError(
        "ops.selu is not yet implemented in ml-switcheroo-compiler"
    )


def separable_conv(*args, **kwargs):
    raise NotImplementedError(
        "ops.separable_conv is not yet implemented in ml-switcheroo-compiler"
    )


def serialize_tensor(*args, **kwargs):
    return _wrap_op("serialize_tensor", msc_ops.serialize_tensor)(*args, **kwargs)


def set_default_stream(*args, **kwargs):
    return _wrap_op("set_default_stream", msc_ops.set_default_stream)(*args, **kwargs)


def set_diag(*args, **kwargs):
    return _wrap_op("set_diag", msc_ops.set_diag)(*args, **kwargs)


def set_memory_limit(*args, **kwargs):
    return _wrap_op("set_memory_limit", msc_ops.set_memory_limit)(*args, **kwargs)


def set_wired_limit(*args, **kwargs):
    return _wrap_op("set_wired_limit", msc_ops.set_wired_limit)(*args, **kwargs)


sets = msc_ops.sets

shape = msc_ops.shape

shape_inference = msc_ops.shape_inference


def sharpen(*args, **kwargs):
    return _wrap_op("sharpen", msc_ops.sharpen)(*args, **kwargs)


def shifted_chebyshev_polynomial_t(*args, **kwargs):
    return _wrap_op(
        "shifted_chebyshev_polynomial_t",
        msc_ops.shifted_chebyshev_polynomial_t,
    )(*args, **kwargs)


def shifted_chebyshev_polynomial_u(*args, **kwargs):
    return _wrap_op(
        "shifted_chebyshev_polynomial_u",
        msc_ops.shifted_chebyshev_polynomial_u,
    )(*args, **kwargs)


def shifted_chebyshev_polynomial_v(*args, **kwargs):
    return _wrap_op(
        "shifted_chebyshev_polynomial_v",
        msc_ops.shifted_chebyshev_polynomial_v,
    )(*args, **kwargs)


def shifted_chebyshev_polynomial_w(*args, **kwargs):
    return _wrap_op(
        "shifted_chebyshev_polynomial_w",
        msc_ops.shifted_chebyshev_polynomial_w,
    )(*args, **kwargs)


def sigmoid(*args, **kwargs):
    return _wrap_op("sigmoid", msc_ops.activations.sigmoid)(*args, **kwargs)


def sign(*args, **kwargs):
    return _wrap_op("sign", msc_ops.sign)(*args, **kwargs)


signal = msc_ops.signal


def signbit(*args, **kwargs):
    return _wrap_op("signbit", msc_ops.signbit)(*args, **kwargs)


def silu(*args, **kwargs):
    raise NotImplementedError(
        "ops.silu is not yet implemented in ml-switcheroo-compiler"
    )


def sin(*args, **kwargs):
    return _wrap_op("sin", msc_ops.sin)(*args, **kwargs)


def sinc(*args, **kwargs):
    return _wrap_op("sinc", msc_ops.sinc)(*args, **kwargs)


def sinh(*args, **kwargs):
    return _wrap_op("sinh", msc_ops.sinh)(*args, **kwargs)


def size(*args, **kwargs):
    raise NotImplementedError(
        "ops.size is not yet implemented in ml-switcheroo-compiler"
    )


def slice(*args, **kwargs):
    raise NotImplementedError(
        "ops.slice is not yet implemented in ml-switcheroo-compiler"
    )


def slice_update(*args, **kwargs):
    raise NotImplementedError(
        "ops.slice_update is not yet implemented in ml-switcheroo-compiler"
    )


slicing = msc_ops.slicing


def slogdet(*args, **kwargs):
    raise NotImplementedError(
        "ops.slogdet is not yet implemented in ml-switcheroo-compiler"
    )


def smm(*args, **kwargs):
    return _wrap_op("smm", msc_ops.smm)(*args, **kwargs)


def sobol_sample(*args, **kwargs):
    return _wrap_op("sobol_sample", msc_ops.sobol_sample)(*args, **kwargs)


def soft_shrink(*args, **kwargs):
    raise NotImplementedError(
        "ops.soft_shrink is not yet implemented in ml-switcheroo-compiler"
    )


def softmax(*args, **kwargs):
    return _wrap_op("softmax", msc_ops.activations.softmax)(*args, **kwargs)


def softplus(*args, **kwargs):
    return _wrap_op("softplus", msc_ops.activations.softplus)(*args, **kwargs)


def softsign(*args, **kwargs):
    raise NotImplementedError(
        "ops.softsign is not yet implemented in ml-switcheroo-compiler"
    )


def solarize(*args, **kwargs):
    return _wrap_op("solarize", msc_ops.solarize)(*args, **kwargs)


def solve(*args, **kwargs):
    raise NotImplementedError(
        "ops.solve is not yet implemented in ml-switcheroo-compiler"
    )


def solve_triangular(*args, **kwargs):
    raise NotImplementedError(
        "ops.solve_triangular is not yet implemented in ml-switcheroo-compiler"
    )


from ml_switcheroo_compiler.ops.shape.misc import sort as msc_sort


def sort(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("sort", msc_sort)(*args, **kwargs)


space_batch = msc_ops.space_batch


def space_to_batch(*args, **kwargs):
    return _wrap_op("space_to_batch", msc_ops.space_to_batch)(*args, **kwargs)


def space_to_batch_nd(*args, **kwargs):
    return _wrap_op("space_to_batch_nd", msc_ops.space_to_batch_nd)(*args, **kwargs)


sparse = msc_ops.sparse


def sparse_categorical_crossentropy(*args, **kwargs):
    raise NotImplementedError(
        "ops.sparse_categorical_crossentropy is not yet implemented in ml-switcheroo-compiler"
    )


def sparse_concat(*args, **kwargs):
    return _wrap_op("sparse_concat", msc_ops.sparse_concat)(*args, **kwargs)


def sparse_eye(*args, **kwargs):
    return _wrap_op("sparse_eye", msc_ops.sparse_eye)(*args, **kwargs)


def sparse_maximum(*args, **kwargs):
    return _wrap_op("sparse_maximum", msc_ops.sparse_maximum)(*args, **kwargs)


def sparse_plus(*args, **kwargs):
    return _wrap_op("sparse_plus", msc_ops.sparse_plus)(*args, **kwargs)


def sparse_reorder(*args, **kwargs):
    return _wrap_op("sparse_reorder", msc_ops.sparse_reorder)(*args, **kwargs)


def sparse_segment_mean(*args, **kwargs):
    return _wrap_op("sparse_segment_mean", msc_ops.sparse_segment_mean)(*args, **kwargs)


def sparse_sigmoid(*args, **kwargs):
    return _wrap_op("sparse_sigmoid", msc_ops.sparse_sigmoid)(*args, **kwargs)


def sparse_softmax(*args, **kwargs):
    return _wrap_op("sparse_softmax", msc_ops.sparse_softmax)(*args, **kwargs)


def sparse_split(*args, **kwargs):
    return _wrap_op("sparse_split", msc_ops.sparse_split)(*args, **kwargs)


def sparse_to_dense(*args, **kwargs):
    return _wrap_op("sparse_to_dense", msc_ops.sparse_to_dense)(*args, **kwargs)


def sparsemax(*args, **kwargs):
    raise NotImplementedError(
        "ops.sparsemax is not yet implemented in ml-switcheroo-compiler"
    )


special = msc_ops.special

from ml_switcheroo_compiler.ops.shape.splitting import split as msc_split


def split(*args, **kwargs):
    if "axis" in kwargs:
        kwargs["dim"] = kwargs.pop("axis")
    return _wrap_op("split", msc_split)(*args, **kwargs)


splitting = msc_ops.splitting


def sqrt(*args, **kwargs):
    return _wrap_op("sqrt", msc_ops.sqrt)(*args, **kwargs)


def sqrtm(*args, **kwargs):
    return _wrap_op("sqrtm", msc_ops.sqrtm)(*args, **kwargs)


def square(*args, **kwargs):
    return _wrap_op("square", msc_ops.square)(*args, **kwargs)


def squared_difference(*args, **kwargs):
    return _wrap_op("squared_difference", msc_ops.squared_difference)(*args, **kwargs)


def squareplus(*args, **kwargs):
    return _wrap_op("squareplus", msc_ops.activations.squareplus)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.frontend import squeeze as msc_squeeze


def squeeze(*args, **kwargs):
    return _wrap_op("squeeze", msc_squeeze)(*args, **kwargs)


def stack(*args, **kwargs):
    return _wrap_op("stack", msc_ops.stack)(*args, **kwargs)


state = msc_ops.state

stats = msc_ops.stats


def std(*args, **kwargs):
    return _wrap_op("std", msc_ops.std)(*args, **kwargs)


def stft(*args, **kwargs):
    return _wrap_op("stft", msc_ops.stft)(*args, **kwargs)


def stop_gradient(*args, **kwargs):
    return _wrap_op("stop_gradient", msc_ops.stop_gradient)(*args, **kwargs)


def stop_gradient_eager(*args, **kwargs):
    return _wrap_op("stop_gradient_eager", msc_ops.stop_gradient_eager)(*args, **kwargs)


def stop_gradient_tracing(*args, **kwargs):
    return _wrap_op("stop_gradient_tracing", msc_ops.stop_gradient_tracing)(
        *args, **kwargs
    )


def string_join(*args, **kwargs):
    return _wrap_op("string_join", msc_ops.string_join)(*args, **kwargs)


def string_length(*args, **kwargs):
    return _wrap_op("string_length", msc_ops.string_length)(*args, **kwargs)


def string_lower(*args, **kwargs):
    return _wrap_op("string_lower", msc_ops.string_lower)(*args, **kwargs)


def string_split(*args, **kwargs):
    return _wrap_op("string_split", msc_ops.string_split)(*args, **kwargs)


def string_substr(*args, **kwargs):
    return _wrap_op("string_substr", msc_ops.string_substr)(*args, **kwargs)


def string_to_hash(*args, **kwargs):
    return _wrap_op("string_to_hash", msc_ops.string_to_hash)(*args, **kwargs)


def string_to_number(*args, **kwargs):
    return _wrap_op("string_to_number", msc_ops.string_to_number)(*args, **kwargs)


def string_upper(*args, **kwargs):
    return _wrap_op("string_upper", msc_ops.string_upper)(*args, **kwargs)


def subtract(*args, **kwargs):
    return _wrap_op("subtract", msc_ops.subtract)(*args, **kwargs)


def sum(*args, **kwargs):
    return _wrap_op("sum", msc_ops.sum)(*args, **kwargs)


def svd(*args, **kwargs):
    raise NotImplementedError(
        "ops.svd is not yet implemented in ml-switcheroo-compiler"
    )


def svdvals(*args, **kwargs):
    return _wrap_op("svdvals", msc_ops.svdvals)(*args, **kwargs)


def swapaxes(*args, **kwargs):
    return _wrap_op("swapaxes", msc_ops.swapaxes)(*args, **kwargs)


def swish(*args, **kwargs):
    raise NotImplementedError(
        "ops.swish is not yet implemented in ml-switcheroo-compiler"
    )


def switch(*args, **kwargs):
    return _wrap_op("switch", msc_ops.switch)(*args, **kwargs)


def switch_case(*args, **kwargs):
    return _wrap_op("switch_case", msc_ops.switch_case)(*args, **kwargs)


def synchronize(*args, **kwargs):
    return _wrap_op("synchronize", msc_ops.synchronize)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.indexing import take as msc_take


def take(*args, **kwargs):
    return _wrap_op("take", msc_take)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.indexing import (
    take_along_axis as msc_take_along_axis,
)


def take_along_axis(*args, **kwargs):
    return _wrap_op("take_along_axis", msc_take_along_axis)(*args, **kwargs)


def tan(*args, **kwargs):
    return _wrap_op("tan", msc_ops.tan)(*args, **kwargs)


def tanh(*args, **kwargs):
    return _wrap_op("tanh", msc_ops.tanh)(*args, **kwargs)


def tanh_shrink(*args, **kwargs):
    raise NotImplementedError(
        "ops.tanh_shrink is not yet implemented in ml-switcheroo-compiler"
    )


tensor_array = msc_ops.tensor_array


def tensor_diag(*args, **kwargs):
    return _wrap_op("tensor_diag", msc_ops.tensor_diag)(*args, **kwargs)


def tensor_diag_part(*args, **kwargs):
    return _wrap_op("tensor_diag_part", msc_ops.tensor_diag_part)(*args, **kwargs)


def tensordot(*args, **kwargs):
    return _wrap_op("tensordot", msc_ops.tensordot)(*args, **kwargs)


def tensorinv(*args, **kwargs):
    return _wrap_op("tensorinv", msc_ops.tensorinv)(*args, **kwargs)


def tensorsolve(*args, **kwargs):
    return _wrap_op("tensorsolve", msc_ops.tensorsolve)(*args, **kwargs)


text = msc_ops.text


def text_vectorization(*args, **kwargs):
    return _wrap_op("text_vectorization", msc_ops.text_vectorization)(*args, **kwargs)


def threshold(*args, **kwargs):
    raise NotImplementedError(
        "ops.threshold is not yet implemented in ml-switcheroo-compiler"
    )


from ml_switcheroo_compiler.ops.shape.misc import tile as msc_tile


def tile(*args, **kwargs):
    return _wrap_op("tile", msc_tile)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.misc import top_k as msc_top_k


def top_k(*args, **kwargs):
    return _wrap_op("top_k", msc_top_k)(*args, **kwargs)


def trace(*args, **kwargs):
    return _wrap_op("trace", msc_ops.trace)(*args, **kwargs)


tracing = msc_ops.tracing

from ml_switcheroo_compiler.ops.shape.frontend import permute as msc_permute


def transpose(*args, **kwargs):
    if "axes" in kwargs:
        kwargs["dims"] = kwargs.pop("axes")
    elif len(args) > 1:
        args = list(args)
        dims = args.pop(1)
        kwargs["dims"] = dims
    return _wrap_op("transpose", msc_permute)(*args, **kwargs)


def tri(*args, **kwargs):
    raise NotImplementedError(
        "ops.tri is not yet implemented in ml-switcheroo-compiler"
    )


def triangular_solve(*args, **kwargs):
    return _wrap_op("triangular_solve", msc_ops.triangular_solve)(*args, **kwargs)


def tridiagonal_matmul(*args, **kwargs):
    return _wrap_op("tridiagonal_matmul", msc_ops.tridiagonal_matmul)(*args, **kwargs)


def tridiagonal_solve(*args, **kwargs):
    return _wrap_op("tridiagonal_solve", msc_ops.tridiagonal_solve)(*args, **kwargs)


trigonometric = msc_ops.trigonometric


def tril(x, k=0):
    return _wrap_op("band_part", msc_ops.band_part)(x, -1, k)


def triu(x, k=0):
    return _wrap_op("band_part", msc_ops.band_part)(x, k, -1)


def true_divide(*args, **kwargs):
    return _wrap_op("true_divide", msc_ops.true_divide)(*args, **kwargs)


def trunc(*args, **kwargs):
    return _wrap_op("trunc", msc_ops.trunc)(*args, **kwargs)


def truncatediv(*args, **kwargs):
    return _wrap_op("truncatediv", msc_ops.truncatediv)(*args, **kwargs)


def truncatemod(*args, **kwargs):
    return _wrap_op("truncatemod", msc_ops.truncatemod)(*args, **kwargs)


type_inference = msc_ops.type_inference

unary = msc_ops.unary


def unravel_index(*args, **kwargs):
    raise NotImplementedError(
        "ops.unravel_index is not yet implemented in ml-switcheroo-compiler"
    )


def unsorted_segment_max(*args, **kwargs):
    return _wrap_op("unsorted_segment_max", msc_ops.unsorted_segment_max)(
        *args, **kwargs
    )


def unsorted_segment_mean(*args, **kwargs):
    return _wrap_op("unsorted_segment_mean", msc_ops.unsorted_segment_mean)(
        *args, **kwargs
    )


def unsorted_segment_min(*args, **kwargs):
    return _wrap_op("unsorted_segment_min", msc_ops.unsorted_segment_min)(
        *args, **kwargs
    )


def unsorted_segment_prod(*args, **kwargs):
    return _wrap_op("unsorted_segment_prod", msc_ops.unsorted_segment_prod)(
        *args, **kwargs
    )


def unsorted_segment_sqrt_n(*args, **kwargs):
    return _wrap_op("unsorted_segment_sqrt_n", msc_ops.unsorted_segment_sqrt_n)(
        *args, **kwargs
    )


def unsorted_segment_sum(*args, **kwargs):
    return _wrap_op("unsorted_segment_sum", msc_ops.unsorted_segment_sum)(
        *args, **kwargs
    )


def unstack(*args, **kwargs):
    return _wrap_op("unstack", msc_ops.unstack)(*args, **kwargs)


utils = msc_ops.utils


def var(*args, **kwargs):
    return _wrap_op("variance", msc_ops.variance)(*args, **kwargs)


def variance(*args, **kwargs):
    return _wrap_op("variance", msc_ops.variance)(*args, **kwargs)


def vdot(*args, **kwargs):
    return _wrap_op("vdot", msc_ops.vdot)(*args, **kwargs)


def vecdot(*args, **kwargs):
    return _wrap_op("vecdot", msc_ops.vecdot)(*args, **kwargs)


def vector_norm(*args, **kwargs):
    return _wrap_op("vector_norm", msc_ops.vector_norm)(*args, **kwargs)


def vectorize(*args, **kwargs):
    raise NotImplementedError(
        "ops.vectorize is not yet implemented in ml-switcheroo-compiler"
    )


def vectorized_map(*args, **kwargs):
    return _wrap_op("vectorized_map", msc_ops.vectorized_map)(*args, **kwargs)


def view_as_complex(*args, **kwargs):
    return _wrap_op("view_as_complex", msc_ops.view_as_complex)(*args, **kwargs)


def view_as_real(*args, **kwargs):
    return _wrap_op("view_as_real", msc_ops.view_as_real)(*args, **kwargs)


vision = msc_ops.vision


def vmap(*args, **kwargs):
    return _wrap_op("vmap", msc_ops.vmap)(*args, **kwargs)


def vsplit(*args, **kwargs):
    return _wrap_op("vsplit", msc_ops.vsplit)(*args, **kwargs)


def vstack(*args, **kwargs):
    return _wrap_op("vstack", msc_ops.vstack)(*args, **kwargs)


def welch(*args, **kwargs):
    return _wrap_op("welch", msc_ops.welch)(*args, **kwargs)


from ml_switcheroo_compiler.ops.shape.indexing import where as msc_where


def where(*args, **kwargs):
    return _wrap_op("where", msc_where)(*args, **kwargs)


def while_loop(*args, **kwargs):
    return _wrap_op("while_loop", msc_ops.while_loop)(*args, **kwargs)


def while_loop_eager(*args, **kwargs):
    return _wrap_op("while_loop_eager", msc_ops.while_loop_eager)(*args, **kwargs)


def while_loop_tracing(*args, **kwargs):
    return _wrap_op("while_loop_tracing", msc_ops.while_loop_tracing)(*args, **kwargs)


def window_hamming(*args, **kwargs):
    return _wrap_op("window_hamming", msc_ops.window_hamming)(*args, **kwargs)


def window_hann(*args, **kwargs):
    return _wrap_op("window_hann", msc_ops.window_hann)(*args, **kwargs)


def write_file(*args, **kwargs):
    return _wrap_op("write_file", msc_ops.write_file)(*args, **kwargs)


def xdivy(*args, **kwargs):
    return _wrap_op("xdivy", msc_ops.xdivy)(*args, **kwargs)


def xlog1py(*args, **kwargs):
    return _wrap_op("xlog1py", msc_ops.xlog1py)(*args, **kwargs)


def xlogy(*args, **kwargs):
    return _wrap_op("xlogy", msc_ops.xlogy)(*args, **kwargs)


def yiq_to_rgb(*args, **kwargs):
    return _wrap_op("yiq_to_rgb", msc_ops.yiq_to_rgb)(*args, **kwargs)


def yuv_to_rgb(*args, **kwargs):
    return _wrap_op("yuv_to_rgb", msc_ops.yuv_to_rgb)(*args, **kwargs)


def zero_fraction(*args, **kwargs):
    return _wrap_op("zero_fraction", msc_ops.zero_fraction)(*args, **kwargs)


def zeros(*args, **kwargs):
    return _wrap_op("zeros", msc_ops.zeros)(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return _wrap_op("zeros_like", msc_ops.zeros_like)(*args, **kwargs)


def zeta(*args, **kwargs):
    return _wrap_op("zeta", msc_ops.zeta)(*args, **kwargs)


class _OpsNamespace:
    def __getattr__(self, name):
        if name in globals():
            return globals()[name]
        raise AttributeError(f"module 'zero_keras.ops' has no attribute '{name}'")


ops = _OpsNamespace()
