"""Ops wrapper module."""

import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.utils.operation_utils import compute_shape_propagation


def _wrap_op(name, msc_op):
    """Function docstring.

    Args:
        name: Description.
        msc_op: Description.
    """

    def wrapper(*args, **kwargs):
        """Function docstring.

        Args:
            args: Description.
            kwargs: Description.
        """
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
                return [  # pragma: no cover
                    KerasTensor(s, dtype=kt.dtype) for s in shape_or_shapes
                ]  # pragma: no cover
            return KerasTensor(shape_or_shapes, dtype=kt.dtype)  # pragma: no cover

        if "axis" in kwargs:
            if name in {
                "concatenate",
                "split",
                "squeeze",
                "repeat",
                "stack",
                "unstack",
            }:
                kwargs["dim"] = kwargs.pop("axis")
            elif name == "roll":
                kwargs["dims"] = kwargs.pop("axis")

        return msc_op(*args, **kwargs)  # pragma: no cover

    return wrapper  # pragma: no cover


def AllGatherOp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("AllGatherOp", getattr(msc_ops, "AllGatherOp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def AllReduceOp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("AllReduceOp", getattr(msc_ops, "AllReduceOp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Arange(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Arange", getattr(msc_ops, "Arange"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ArgSort(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ArgSort", getattr(msc_ops, "ArgSort"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Assign(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Assign", getattr(msc_ops, "Assign"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def AssignAdd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("AssignAdd", getattr(msc_ops, "AssignAdd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def AssignSub(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("AssignSub", getattr(msc_ops, "AssignSub"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def AssignVariable(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "AssignVariable", getattr(msc_ops, "AssignVariable")
    )(  # pragma: no cover
        *args, **kwargs
    )


def AttentionConfig(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "AttentionConfig", getattr(msc_ops, "AttentionConfig")
    )(  # pragma: no cover
        *args, **kwargs
    )


def AttentionInputs(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "AttentionInputs", getattr(msc_ops, "AttentionInputs")
    )(  # pragma: no cover
        *args, **kwargs
    )


def BroadcastInDim(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "BroadcastInDim", getattr(msc_ops, "BroadcastInDim")
    )(  # pragma: no cover
        *args, **kwargs
    )


def BroadcastTo(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("BroadcastTo", getattr(msc_ops, "BroadcastTo"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ColumnStack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ColumnStack", getattr(msc_ops, "ColumnStack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ComplexWarning(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ComplexWarning", getattr(msc_ops, "ComplexWarning")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Concatenate(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Concatenate", getattr(msc_ops, "Concatenate"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ConvGeneralDilated(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ConvGeneralDilated", getattr(msc_ops, "ConvGeneralDilated")
    )(  # pragma: no cover
        *args, **kwargs
    )


def CreationOp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("CreationOp", getattr(msc_ops, "CreationOp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def DType(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("DType", getattr(msc_ops, "DType"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Dot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Dot", getattr(msc_ops, "Dot"))(*args, **kwargs)  # pragma: no cover


def DotGeneral(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("DotGeneral", getattr(msc_ops, "DotGeneral"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Dsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Dsplit", getattr(msc_ops, "Dsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Dstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Dstack", getattr(msc_ops, "Dstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def DynamicSlice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "DynamicSlice", getattr(msc_ops, "DynamicSlice")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def DynamicUpdateSlice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "DynamicUpdateSlice", getattr(msc_ops, "DynamicUpdateSlice")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Einsum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Einsum", getattr(msc_ops, "Einsum"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ElasticTransform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ElasticTransform", getattr(msc_ops, "ElasticTransform")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Expand(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Expand", getattr(msc_ops, "Expand"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ExtractBoundingBoxes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ExtractBoundingBoxes", getattr(msc_ops, "ExtractBoundingBoxes")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Fft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Fft", getattr(msc_ops, "Fft"))(*args, **kwargs)  # pragma: no cover


def Flatten(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Flatten", getattr(msc_ops, "Flatten"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Full(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Full", getattr(msc_ops, "Full"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Gather(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Gather", getattr(msc_ops, "Gather"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def GatherNd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("GatherNd", getattr(msc_ops, "GatherNd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def GaussianBlur(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "GaussianBlur", getattr(msc_ops, "GaussianBlur")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Hsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Hsplit", getattr(msc_ops, "Hsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Hstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Hstack", getattr(msc_ops, "Hstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def IoU(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("IoU", getattr(msc_ops, "IoU"))(*args, **kwargs)  # pragma: no cover


def Matmul(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Matmul", getattr(msc_ops, "Matmul"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def MedianFilter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "MedianFilter", getattr(msc_ops, "MedianFilter")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Meshgrid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Meshgrid", getattr(msc_ops, "Meshgrid"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Moveaxis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Moveaxis", getattr(msc_ops, "Moveaxis"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def NonMaxSuppression(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "NonMaxSuppression", getattr(msc_ops, "NonMaxSuppression")
    )(  # pragma: no cover
        *args, **kwargs
    )


def NormConfig(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("NormConfig", getattr(msc_ops, "NormConfig"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Ones(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Ones", getattr(msc_ops, "Ones"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def OpDef(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("OpDef", getattr(msc_ops, "OpDef"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Permute(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Permute", getattr(msc_ops, "Permute"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def PerspectiveTransform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "PerspectiveTransform", getattr(msc_ops, "PerspectiveTransform")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Pmean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Pmean", getattr(msc_ops, "Pmean"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Psum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Psum", getattr(msc_ops, "Psum"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def RaggedGather(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "RaggedGather", getattr(msc_ops, "RaggedGather")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def RaggedTensorToDense(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "RaggedTensorToDense", getattr(msc_ops, "RaggedTensorToDense")
    )(  # pragma: no cover
        *args, **kwargs
    )


def ReadVariable(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "ReadVariable", getattr(msc_ops, "ReadVariable")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ReduceScatterOp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ReduceScatterOp", getattr(msc_ops, "ReduceScatterOp")
    )(  # pragma: no cover
        *args, **kwargs
    )


def ReduceWindow(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "ReduceWindow", getattr(msc_ops, "ReduceWindow")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Repeat(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Repeat", getattr(msc_ops, "Repeat"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Reshape(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Reshape", getattr(msc_ops, "Reshape"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Resize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Resize", getattr(msc_ops, "Resize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ResizeBicubic(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "ResizeBicubic", getattr(msc_ops, "ResizeBicubic")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ResizeBilinear(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ResizeBilinear", getattr(msc_ops, "ResizeBilinear")
    )(  # pragma: no cover
        *args, **kwargs
    )


def ResizeLanczos3(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ResizeLanczos3", getattr(msc_ops, "ResizeLanczos3")
    )(  # pragma: no cover
        *args, **kwargs
    )


def ResizeNearest(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "ResizeNearest", getattr(msc_ops, "ResizeNearest")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Rfft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Rfft", getattr(msc_ops, "Rfft"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Roll(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Roll", getattr(msc_ops, "Roll"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def RowStack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("RowStack", getattr(msc_ops, "RowStack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Scatter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Scatter", getattr(msc_ops, "Scatter"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ScatterAdd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ScatterAdd", getattr(msc_ops, "ScatterAdd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ScatterNd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ScatterNd", getattr(msc_ops, "ScatterNd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def SearchSorted(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "SearchSorted", getattr(msc_ops, "SearchSorted")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Select(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Select", getattr(msc_ops, "Select"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ShardTensorOp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "ShardTensorOp", getattr(msc_ops, "ShardTensorOp")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Slice", getattr(msc_ops, "Slice"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def SobolSample(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("SobolSample", getattr(msc_ops, "SobolSample"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Sort(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Sort", getattr(msc_ops, "Sort"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def SpaceConfig(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("SpaceConfig", getattr(msc_ops, "SpaceConfig"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def SparseAdd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("SparseAdd", getattr(msc_ops, "SparseAdd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def SparseDenseMatMul(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "SparseDenseMatMul", getattr(msc_ops, "SparseDenseMatMul")
    )(  # pragma: no cover
        *args, **kwargs
    )


def SparseReduceSum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "SparseReduceSum", getattr(msc_ops, "SparseReduceSum")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Split(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Split", getattr(msc_ops, "Split"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Squeeze(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Squeeze", getattr(msc_ops, "Squeeze"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Stack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Stack", getattr(msc_ops, "Stack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def StridedSlice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "StridedSlice", getattr(msc_ops, "StridedSlice")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Swapaxes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Swapaxes", getattr(msc_ops, "Swapaxes"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Take(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Take", getattr(msc_ops, "Take"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def TakeAlongAxis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "TakeAlongAxis", getattr(msc_ops, "TakeAlongAxis")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Tensor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Tensor", getattr(msc_ops, "Tensor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def TensorArrayRead(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "TensorArrayRead", getattr(msc_ops, "TensorArrayRead")
    )(  # pragma: no cover
        *args, **kwargs
    )


def TensorArrayStack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "TensorArrayStack", getattr(msc_ops, "TensorArrayStack")
    )(  # pragma: no cover
        *args, **kwargs
    )


def TensorArrayWrite(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "TensorArrayWrite", getattr(msc_ops, "TensorArrayWrite")
    )(  # pragma: no cover
        *args, **kwargs
    )


def TensorConfig(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "TensorConfig", getattr(msc_ops, "TensorConfig")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def TensorScatterUpdate(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "TensorScatterUpdate", getattr(msc_ops, "TensorScatterUpdate")
    )(  # pragma: no cover
        *args, **kwargs
    )


def Tile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Tile", getattr(msc_ops, "Tile"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def TopK(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("TopK", getattr(msc_ops, "TopK"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Transpose", getattr(msc_ops, "Transpose"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Tril(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Tril", getattr(msc_ops, "Tril"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Triu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Triu", getattr(msc_ops, "Triu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Vdot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Vdot", getattr(msc_ops, "Vdot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Vsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Vsplit", getattr(msc_ops, "Vsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Vstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Vstack", getattr(msc_ops, "Vstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Where(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Where", getattr(msc_ops, "Where"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def Zeros(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("Zeros", getattr(msc_ops, "Zeros"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def abs(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("abs", getattr(msc_ops, "abs"))(*args, **kwargs)  # pragma: no cover


def absolute(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("absolute", getattr(msc_ops, "absolute"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def acos(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("acos", getattr(msc_ops, "acos"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def acosh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("acosh", getattr(msc_ops, "acosh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def activity_regularization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "activity_regularization", getattr(msc_ops, "activity_regularization")
    )(*args, **kwargs)


def add(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("add", getattr(msc_ops, "add"))(*args, **kwargs)  # pragma: no cover


def adjust_brightness(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "adjust_brightness", getattr(msc_ops, "adjust_brightness")
    )(  # pragma: no cover
        *args, **kwargs
    )


def adjust_contrast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "adjust_contrast", getattr(msc_ops, "adjust_contrast")
    )(  # pragma: no cover
        *args, **kwargs
    )


def adjust_hue(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("adjust_hue", getattr(msc_ops, "adjust_hue"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def adjust_saturation(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "adjust_saturation", getattr(msc_ops, "adjust_saturation")
    )(  # pragma: no cover
        *args, **kwargs
    )


def affine_generator(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "affine_generator", getattr(msc_ops, "affine_generator")
    )(  # pragma: no cover
        *args, **kwargs
    )


def affine_transform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "affine_transform", getattr(msc_ops, "affine_transform")
    )(  # pragma: no cover
        *args, **kwargs
    )


aliases = msc_ops.aliases


def all(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("all", getattr(msc_ops, "all"))(*args, **kwargs)  # pragma: no cover


def all_gather(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("all_gather", getattr(msc_ops, "all_gather"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def all_reduce(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("all_reduce", getattr(msc_ops, "all_reduce"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def allclose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("allclose", getattr(msc_ops, "allclose"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def amax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("amax", getattr(msc_ops, "amax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def amin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("amin", getattr(msc_ops, "amin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def angle(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("angle", getattr(msc_ops, "angle"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


annotations = msc_ops.annotations


def any(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("any", getattr(msc_ops, "any"))(*args, **kwargs)  # pragma: no cover


def append(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("append", getattr(msc_ops, "append"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def apply_along_axis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "apply_along_axis", getattr(msc_ops, "apply_along_axis")
    )(  # pragma: no cover
        *args, **kwargs
    )


def apply_over_axes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "apply_over_axes", getattr(msc_ops, "apply_over_axes")
    )(  # pragma: no cover
        *args, **kwargs
    )


def approx_max_k(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "approx_max_k", getattr(msc_ops, "approx_max_k")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def approx_min_k(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "approx_min_k", getattr(msc_ops, "approx_min_k")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arange(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arange", getattr(msc_ops, "arange"))(
        *args, **kwargs
    )  # pragma: no cover


def arccos(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arccos", getattr(msc_ops, "arccos"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arccosh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arccosh", getattr(msc_ops, "arccosh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arcsin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arcsin", getattr(msc_ops, "arcsin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arcsinh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arcsinh", getattr(msc_ops, "arcsinh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arctan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arctan", getattr(msc_ops, "arctan"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arctan2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arctan2", getattr(msc_ops, "arctan2"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def arctanh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("arctanh", getattr(msc_ops, "arctanh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def argmax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("argmax", getattr(msc_ops, "argmax"))(
        *args, **kwargs
    )  # pragma: no cover


def argmin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("argmin", getattr(msc_ops, "argmin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def argpartition(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "argpartition", getattr(msc_ops, "argpartition")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def argsort(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("argsort", getattr(msc_ops, "argsort"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def argwhere(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("argwhere", getattr(msc_ops, "argwhere"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def around(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("around", getattr(msc_ops, "around"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def array(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array", getattr(msc_ops, "array"))(
        *args, **kwargs
    )  # pragma: no cover


def array_equal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array_equal", getattr(msc_ops, "array_equal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def array_equiv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array_equiv", getattr(msc_ops, "array_equiv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def array_repr(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array_repr", getattr(msc_ops, "array_repr"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def array_split(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array_split", getattr(msc_ops, "array_split"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def array_str(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("array_str", getattr(msc_ops, "array_str"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def as_string(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("as_string", getattr(msc_ops, "as_string"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def asarray(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("asarray", getattr(msc_ops, "asarray"))(
        *args, **kwargs
    )  # pragma: no cover


def asin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("asin", getattr(msc_ops, "asin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def asinh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("asinh", getattr(msc_ops, "asinh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def associative_scan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "associative_scan", getattr(msc_ops, "associative_scan")
    )(  # pragma: no cover
        *args, **kwargs
    )


def astype(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("astype", getattr(msc_ops, "astype"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atan", getattr(msc_ops, "atan"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atan2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atan2", getattr(msc_ops, "atan2"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atanh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atanh", getattr(msc_ops, "atanh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atleast_1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atleast_1d", getattr(msc_ops, "atleast_1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atleast_2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atleast_2d", getattr(msc_ops, "atleast_2d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def atleast_3d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("atleast_3d", getattr(msc_ops, "atleast_3d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def attention(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("attention", getattr(msc_ops, "attention"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


audio = msc_ops.audio


def augmix(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("augmix", getattr(msc_ops, "augmix"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def auto_contrast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "auto_contrast", getattr(msc_ops, "auto_contrast")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def average(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("average", getattr(msc_ops, "average"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def average_pool(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "average_pool", getattr(msc_ops, "average_pool")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def avg_pool(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("avg_pool", getattr(msc_ops, "avg_pool"))(
        *args, **kwargs
    )  # pragma: no cover


def bartlett(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bartlett", getattr(msc_ops, "bartlett"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


base = msc_ops.base


def batch_normalization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "batch_normalization", getattr(msc_ops, "batch_normalization")
    )(  # pragma: no cover
        *args, **kwargs
    )


def bessel_i0(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bessel_i0", getattr(msc_ops, "bessel_i0"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bessel_i0e(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bessel_i0e", getattr(msc_ops, "bessel_i0e"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bessel_i1(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bessel_i1", getattr(msc_ops, "bessel_i1"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bessel_i1e(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bessel_i1e", getattr(msc_ops, "bessel_i1e"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def betainc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("betainc", getattr(msc_ops, "betainc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


bfloat16 = msc_ops.bfloat16


def bidirectional(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "bidirectional", getattr(msc_ops, "bidirectional")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


binary = msc_ops.binary


def binary_crossentropy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "binary_crossentropy", getattr(msc_ops, "binary_crossentropy")
    )(  # pragma: no cover
        *args, **kwargs
    )


def bincount(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bincount", getattr(msc_ops, "bincount"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitcast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bitcast", getattr(msc_ops, "bitcast"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitwise_and(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bitwise_and", getattr(msc_ops, "bitwise_and"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitwise_count(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "bitwise_count", getattr(msc_ops, "bitwise_count")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitwise_invert(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "bitwise_invert", getattr(msc_ops, "bitwise_invert")
    )(  # pragma: no cover
        *args, **kwargs
    )


def bitwise_left_shift(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "bitwise_left_shift", getattr(msc_ops, "bitwise_left_shift")
    )(  # pragma: no cover
        *args, **kwargs
    )


def bitwise_not(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bitwise_not", getattr(msc_ops, "bitwise_not"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitwise_or(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bitwise_or", getattr(msc_ops, "bitwise_or"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bitwise_right_shift(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "bitwise_right_shift", getattr(msc_ops, "bitwise_right_shift")
    )(  # pragma: no cover
        *args, **kwargs
    )


def bitwise_xor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bitwise_xor", getattr(msc_ops, "bitwise_xor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def blackman(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("blackman", getattr(msc_ops, "blackman"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def block(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("block", getattr(msc_ops, "block"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def bool(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("bool", getattr(msc_ops, "bool"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


bool_ = msc_ops.bool_


def boolean_mask(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "boolean_mask", getattr(msc_ops, "boolean_mask")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def broadcast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("broadcast", getattr(msc_ops, "broadcast"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def broadcast_arrays(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "broadcast_arrays", getattr(msc_ops, "broadcast_arrays")
    )(  # pragma: no cover
        *args, **kwargs
    )


def broadcast_in_dim(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "broadcast_in_dim", getattr(msc_ops, "broadcast_in_dim")
    )(  # pragma: no cover
        *args, **kwargs
    )


def broadcast_shapes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "broadcast_shapes", getattr(msc_ops, "broadcast_shapes")
    )(  # pragma: no cover
        *args, **kwargs
    )


def broadcast_to(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "broadcast_to", getattr(msc_ops, "broadcast_to")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


c_ = msc_ops.c_


def can_cast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("can_cast", getattr(msc_ops, "can_cast"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def cast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cast", getattr(msc_ops, "cast"))(
        *args, **kwargs
    )  # pragma: no cover


def categorical_crossentropy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "categorical_crossentropy", getattr(msc_ops, "categorical_crossentropy")
    )(*args, **kwargs)


def categorical_generalized_cross_entropy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "categorical_generalized_cross_entropy",
        getattr(msc_ops, "categorical_generalized_cross_entropy"),
    )(*args, **kwargs)


def cbrt(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cbrt", getattr(msc_ops, "cbrt"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


cdouble = msc_ops.cdouble


def ceil(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ceil", getattr(msc_ops, "ceil"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def celu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("celu", getattr(msc_ops, "celu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


character = msc_ops.character


def cholesky(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cholesky", getattr(msc_ops, "cholesky"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def choose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("choose", getattr(msc_ops, "choose"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def circle_loss(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("circle_loss", getattr(msc_ops, "circle_loss"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def clamp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("clamp", getattr(msc_ops, "clamp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def clip(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("clip", getattr(msc_ops, "clip"))(
        *args, **kwargs
    )  # pragma: no cover


def column_stack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "column_stack", getattr(msc_ops, "column_stack")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def complex(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("complex", getattr(msc_ops, "complex"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


complex128 = msc_ops.complex128

complex64 = msc_ops.complex64

complex_ = msc_ops.complex_

complexfloating = msc_ops.complexfloating


def compress(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("compress", getattr(msc_ops, "compress"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def concat(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("concat", getattr(msc_ops, "concat"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def concatenate(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("concatenate", getattr(msc_ops, "concatenate"))(
        *args, **kwargs
    )  # pragma: no cover


def cond(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cond", getattr(msc_ops, "cond"))(
        *args, **kwargs
    )  # pragma: no cover


config = msc_ops.config

configs = msc_ops.configs


def conj(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conj", getattr(msc_ops, "conj"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def conjugate(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conjugate", getattr(msc_ops, "conjugate"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def conv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conv", getattr(msc_ops, "conv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def conv1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conv1d", getattr(msc_ops, "conv1d"))(
        *args, **kwargs
    )  # pragma: no cover


def conv1d_lstm_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "conv1d_lstm_cell", getattr(msc_ops, "conv1d_lstm_cell")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv1d_transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "conv1d_transpose", getattr(msc_ops, "conv1d_transpose")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conv2d", getattr(msc_ops, "conv2d"))(
        *args, **kwargs
    )  # pragma: no cover


def conv2d_lstm_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "conv2d_lstm_cell", getattr(msc_ops, "conv2d_lstm_cell")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv2d_transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "conv2d_transpose", getattr(msc_ops, "conv2d_transpose")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv3d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("conv3d", getattr(msc_ops, "conv3d"))(
        *args, **kwargs
    )  # pragma: no cover


def conv3d_lstm_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "conv3d_lstm_cell", getattr(msc_ops, "conv3d_lstm_cell")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv3d_transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "conv3d_transpose", getattr(msc_ops, "conv3d_transpose")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv_general_dilated(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "conv_general_dilated", getattr(msc_ops, "conv_general_dilated")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv_lstm_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "conv_lstm_cell", getattr(msc_ops, "conv_lstm_cell")
    )(  # pragma: no cover
        *args, **kwargs
    )


def conv_transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "conv_transpose", getattr(msc_ops, "conv_transpose")
    )(  # pragma: no cover
        *args, **kwargs
    )


def convert_to_numpy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "convert_to_numpy", getattr(msc_ops, "convert_to_numpy")
    )(  # pragma: no cover
        *args, **kwargs
    )


def convert_to_tensor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "convert_to_tensor", getattr(msc_ops, "convert_to_tensor")
    )(  # pragma: no cover
        *args, **kwargs
    )


def convolve(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("convolve", getattr(msc_ops, "convolve"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def copy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("copy", getattr(msc_ops, "copy"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def copysign(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("copysign", getattr(msc_ops, "copysign"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


core_config = msc_ops.core_config


def corrcoef(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("corrcoef", getattr(msc_ops, "corrcoef"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def correlate(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("correlate", getattr(msc_ops, "correlate"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def cos(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cos", getattr(msc_ops, "cos"))(*args, **kwargs)  # pragma: no cover


def cosh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cosh", getattr(msc_ops, "cosh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def count_nonzero(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "count_nonzero", getattr(msc_ops, "count_nonzero")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def cov(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cov", getattr(msc_ops, "cov"))(*args, **kwargs)  # pragma: no cover


def create_eager_alias(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "create_eager_alias", getattr(msc_ops, "create_eager_alias")
    )(  # pragma: no cover
        *args, **kwargs
    )


creation = msc_ops.creation


def crop(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("crop", getattr(msc_ops, "crop"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def crop_and_resize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "crop_and_resize", getattr(msc_ops, "crop_and_resize")
    )(  # pragma: no cover
        *args, **kwargs
    )


def cross(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cross", getattr(msc_ops, "cross"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


csingle = msc_ops.csingle


def ctc_decode(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ctc_decode", getattr(msc_ops, "ctc_decode"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ctc_loss(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ctc_loss", getattr(msc_ops, "ctc_loss"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def cumprod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cumprod", getattr(msc_ops, "cumprod"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def cumsum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cumsum", getattr(msc_ops, "cumsum"))(
        *args, **kwargs
    )  # pragma: no cover


def cumulative_sum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "cumulative_sum", getattr(msc_ops, "cumulative_sum")
    )(  # pragma: no cover
        *args, **kwargs
    )


def custom_gradient(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "custom_gradient", getattr(msc_ops, "custom_gradient")
    )(  # pragma: no cover
        *args, **kwargs
    )


def cutmix(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("cutmix", getattr(msc_ops, "cutmix"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def deg2rad(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("deg2rad", getattr(msc_ops, "deg2rad"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def degeneration(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "degeneration", getattr(msc_ops, "degeneration")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def degrees(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("degrees", getattr(msc_ops, "degrees"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def delete(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("delete", getattr(msc_ops, "delete"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def depthwise_conv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "depthwise_conv", getattr(msc_ops, "depthwise_conv")
    )(  # pragma: no cover
        *args, **kwargs
    )


def depthwise_conv1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "depthwise_conv1d", getattr(msc_ops, "depthwise_conv1d")
    )(  # pragma: no cover
        *args, **kwargs
    )


def depthwise_conv2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "depthwise_conv2d", getattr(msc_ops, "depthwise_conv2d")
    )(  # pragma: no cover
        *args, **kwargs
    )


def det(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("det", getattr(msc_ops, "det"))(*args, **kwargs)  # pragma: no cover


def diag(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("diag", getattr(msc_ops, "diag"))(
        *args, **kwargs
    )  # pragma: no cover


def diag_indices(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "diag_indices", getattr(msc_ops, "diag_indices")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def diag_indices_from(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "diag_indices_from", getattr(msc_ops, "diag_indices_from")
    )(  # pragma: no cover
        *args, **kwargs
    )


def diagflat(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("diagflat", getattr(msc_ops, "diagflat"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def diagonal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("diagonal", getattr(msc_ops, "diagonal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dice_loss(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dice_loss", getattr(msc_ops, "dice_loss"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def diff(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("diff", getattr(msc_ops, "diff"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def digamma(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("digamma", getattr(msc_ops, "digamma"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def digitize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("digitize", getattr(msc_ops, "digitize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dispatch_eager(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "dispatch_eager", getattr(msc_ops, "dispatch_eager")
    )(  # pragma: no cover
        *args, **kwargs
    )


dispatcher = msc_ops.dispatcher

distributed = msc_ops.distributed


def divide(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("divide", getattr(msc_ops, "divide"))(
        *args, **kwargs
    )  # pragma: no cover


def divide_no_nan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "divide_no_nan", getattr(msc_ops, "divide_no_nan")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def divmod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("divmod", getattr(msc_ops, "divmod"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dot", getattr(msc_ops, "dot"))(*args, **kwargs)  # pragma: no cover


def dot_general(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dot_general", getattr(msc_ops, "dot_general"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dot_product_attention(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "dot_product_attention", getattr(msc_ops, "dot_product_attention")
    )(  # pragma: no cover
        *args, **kwargs
    )


double = msc_ops.double


def dropout(
    x, rate=0.5, noise_shape=None, seed=None, training=False, **kwargs
):  # pragma: no cover
    """Function docstring.

    Args:
        x: Description.
        rate: Description.
        noise_shape: Description.
        seed: Description.
        training: Description.
        kwargs: Description.
    """
    from ml_switcheroo_compiler.ops.nn.dropout import DropoutConfig

    config = DropoutConfig(noise_shape=noise_shape, training=training, seed=seed)
    return _wrap_op("dropout", getattr(msc_ops, "dropout"))(
        x, rate=rate, config=config
    )  # pragma: no cover


def dsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dsplit", getattr(msc_ops, "dsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dstack", getattr(msc_ops, "dstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dtype(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("dtype", getattr(msc_ops, "dtype"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dynamic_slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "dynamic_slice", getattr(msc_ops, "dynamic_slice")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def dynamic_update_slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "dynamic_update_slice", getattr(msc_ops, "dynamic_update_slice")
    )(  # pragma: no cover
        *args, **kwargs
    )


e = msc_ops.e

eager_evaluator = msc_ops.eager_evaluator


def ediff1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ediff1d", getattr(msc_ops, "ediff1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def edit_distance(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "edit_distance", getattr(msc_ops, "edit_distance")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def eig(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("eig", getattr(msc_ops, "eig"))(*args, **kwargs)  # pragma: no cover


def eigh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("eigh", getattr(msc_ops, "eigh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def eigvalsh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("eigvalsh", getattr(msc_ops, "eigvalsh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def einsum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("einsum", getattr(msc_ops, "einsum"))(
        *args, **kwargs
    )  # pragma: no cover


def einsum_path(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("einsum_path", getattr(msc_ops, "einsum_path"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def elastic_transform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "elastic_transform", getattr(msc_ops, "elastic_transform")
    )(  # pragma: no cover
        *args, **kwargs
    )


def elu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("elu", getattr(msc_ops, "elu"))(*args, **kwargs)  # pragma: no cover


def embedding(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("embedding", getattr(msc_ops, "embedding"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def emit_ir_node(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "emit_ir_node", getattr(msc_ops, "emit_ir_node")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def empty(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("empty", getattr(msc_ops, "empty"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def empty_like(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("empty_like", getattr(msc_ops, "empty_like"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def equal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("equal", getattr(msc_ops, "equal"))(
        *args, **kwargs
    )  # pragma: no cover


def equalization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "equalization", getattr(msc_ops, "equalization")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def erf(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("erf", getattr(msc_ops, "erf"))(*args, **kwargs)  # pragma: no cover


def erfc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("erfc", getattr(msc_ops, "erfc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def erfcinv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("erfcinv", getattr(msc_ops, "erfcinv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def erfinv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("erfinv", getattr(msc_ops, "erfinv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


euler_gamma = msc_ops.euler_gamma


def exp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("exp", getattr(msc_ops, "exp"))(*args, **kwargs)  # pragma: no cover


def exp2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("exp2", getattr(msc_ops, "exp2"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def expand(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("expand", getattr(msc_ops, "expand"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def expand_dims(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("expand_dims", getattr(msc_ops, "expand_dims"))(
        *args, **kwargs
    )  # pragma: no cover


def expm1(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("expm1", getattr(msc_ops, "expm1"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def extract(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("extract", getattr(msc_ops, "extract"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def extract_bounding_boxes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "extract_bounding_boxes", getattr(msc_ops, "extract_bounding_boxes")
    )(*args, **kwargs)


def extract_sequences(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "extract_sequences", getattr(msc_ops, "extract_sequences")
    )(  # pragma: no cover
        *args, **kwargs
    )


def eye(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("eye", getattr(msc_ops, "eye"))(*args, **kwargs)  # pragma: no cover


def fabs(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fabs", getattr(msc_ops, "fabs"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fft", getattr(msc_ops, "fft"))(*args, **kwargs)  # pragma: no cover


def fft2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fft2", getattr(msc_ops, "fft2"))(
        *args, **kwargs
    )  # pragma: no cover


def fft2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fft2d", getattr(msc_ops, "fft2d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fill_diagonal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "fill_diagonal", getattr(msc_ops, "fill_diagonal")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def finfo(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("finfo", getattr(msc_ops, "finfo"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fix(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fix", getattr(msc_ops, "fix"))(*args, **kwargs)  # pragma: no cover


def flatnonzero(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("flatnonzero", getattr(msc_ops, "flatnonzero"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def flatten(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("flatten", getattr(msc_ops, "flatten"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


flexible = msc_ops.flexible


def flip(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("flip", getattr(msc_ops, "flip"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def flip_left_right(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "flip_left_right", getattr(msc_ops, "flip_left_right")
    )(  # pragma: no cover
        *args, **kwargs
    )


def flip_up_down(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "flip_up_down", getattr(msc_ops, "flip_up_down")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fliplr(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fliplr", getattr(msc_ops, "fliplr"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def flipud(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("flipud", getattr(msc_ops, "flipud"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def float(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("float", getattr(msc_ops, "float"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


float16 = msc_ops.float16

float32 = msc_ops.float32

float64 = msc_ops.float64

float8_e4m3b11fnuz = msc_ops.float8_e4m3b11fnuz

float8_e4m3fn = msc_ops.float8_e4m3fn

float8_e4m3fnuz = msc_ops.float8_e4m3fnuz

float8_e5m2 = msc_ops.float8_e5m2

float8_e5m2fnuz = msc_ops.float8_e5m2fnuz

float_ = msc_ops.float_


def float_power(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("float_power", getattr(msc_ops, "float_power"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


floating = msc_ops.floating


def floor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("floor", getattr(msc_ops, "floor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def floor_divide(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "floor_divide", getattr(msc_ops, "floor_divide")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fmax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fmax", getattr(msc_ops, "fmax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fmin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fmin", getattr(msc_ops, "fmin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fmod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fmod", getattr(msc_ops, "fmod"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fori_loop(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fori_loop", getattr(msc_ops, "fori_loop"))(
        *args, **kwargs
    )  # pragma: no cover


def frexp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("frexp", getattr(msc_ops, "frexp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def from_dlpack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("from_dlpack", getattr(msc_ops, "from_dlpack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def frombuffer(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("frombuffer", getattr(msc_ops, "frombuffer"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fromfile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fromfile", getattr(msc_ops, "fromfile"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fromfunction(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "fromfunction", getattr(msc_ops, "fromfunction")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fromiter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fromiter", getattr(msc_ops, "fromiter"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def frompyfunc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("frompyfunc", getattr(msc_ops, "frompyfunc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def fromstring(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("fromstring", getattr(msc_ops, "fromstring"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def full(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("full", getattr(msc_ops, "full"))(
        *args, **kwargs
    )  # pragma: no cover


def full_like(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("full_like", getattr(msc_ops, "full_like"))(
        *args, **kwargs
    )  # pragma: no cover


def gather(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gather", getattr(msc_ops, "gather"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def gather_nd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gather_nd", getattr(msc_ops, "gather_nd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def gaussian_blur(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "gaussian_blur", getattr(msc_ops, "gaussian_blur")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def gcd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gcd", getattr(msc_ops, "gcd"))(*args, **kwargs)  # pragma: no cover


def gelu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gelu", getattr(msc_ops, "gelu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


generic = msc_ops.generic


def geomspace(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("geomspace", getattr(msc_ops, "geomspace"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def get_active_backend(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "get_active_backend", getattr(msc_ops, "get_active_backend")
    )(  # pragma: no cover
        *args, **kwargs
    )


def get_item(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("get_item", getattr(msc_ops, "get_item"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def get_op(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("get_op", getattr(msc_ops, "get_op"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def get_printoptions(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "get_printoptions", getattr(msc_ops, "get_printoptions")
    )(  # pragma: no cover
        *args, **kwargs
    )


def glu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("glu", getattr(msc_ops, "glu"))(*args, **kwargs)  # pragma: no cover


def gradient(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gradient", getattr(msc_ops, "gradient"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def greater(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("greater", getattr(msc_ops, "greater"))(
        *args, **kwargs
    )  # pragma: no cover


def greater_equal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("greater_equal", getattr(msc_ops, "greater_equal"))(
        *args, **kwargs
    )  # pragma: no cover


def group_mean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("group_mean", getattr(msc_ops, "group_mean"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def group_norm(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("group_norm", getattr(msc_ops, "group_norm"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def group_variance(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "group_variance", getattr(msc_ops, "group_variance")
    )(  # pragma: no cover
        *args, **kwargs
    )


def gru_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("gru_cell", getattr(msc_ops, "gru_cell"))(
        *args, **kwargs
    )  # pragma: no cover


def hamming(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hamming", getattr(msc_ops, "hamming"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hanning(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hanning", getattr(msc_ops, "hanning"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hard_shrink(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hard_shrink", getattr(msc_ops, "hard_shrink"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hard_sigmoid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "hard_sigmoid", getattr(msc_ops, "hard_sigmoid")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hard_silu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hard_silu", getattr(msc_ops, "hard_silu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hard_swish(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hard_swish", getattr(msc_ops, "hard_swish"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hard_tanh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hard_tanh", getattr(msc_ops, "hard_tanh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def heaviside(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("heaviside", getattr(msc_ops, "heaviside"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def histogram(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("histogram", getattr(msc_ops, "histogram"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def histogram2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("histogram2d", getattr(msc_ops, "histogram2d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def histogram_bin_edges(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "histogram_bin_edges", getattr(msc_ops, "histogram_bin_edges")
    )(  # pragma: no cover
        *args, **kwargs
    )


def histogramdd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("histogramdd", getattr(msc_ops, "histogramdd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hsplit", getattr(msc_ops, "hsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hstack", getattr(msc_ops, "hstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hsv_to_rgb(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hsv_to_rgb", getattr(msc_ops, "hsv_to_rgb"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def hypot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("hypot", getattr(msc_ops, "hypot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def i0(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("i0", getattr(msc_ops, "i0"))(*args, **kwargs)  # pragma: no cover


def identity(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("identity", getattr(msc_ops, "identity"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ifft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ifft", getattr(msc_ops, "ifft"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ifft2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ifft2", getattr(msc_ops, "ifft2"))(
        *args, **kwargs
    )  # pragma: no cover


def ifft2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ifft2d", getattr(msc_ops, "ifft2d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def igamma(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("igamma", getattr(msc_ops, "igamma"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def igammac(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("igammac", getattr(msc_ops, "igammac"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def iinfo(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("iinfo", getattr(msc_ops, "iinfo"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def imag(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("imag", getattr(msc_ops, "imag"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


image = msc_ops.image


def image_resize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "image_resize", getattr(msc_ops, "image_resize")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def in_top_k(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("in_top_k", getattr(msc_ops, "in_top_k"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


index_exp = msc_ops.index_exp


def indices(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("indices", getattr(msc_ops, "indices"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


inexact = msc_ops.inexact

inf = msc_ops.inf


def inner(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("inner", getattr(msc_ops, "inner"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def insert(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("insert", getattr(msc_ops, "insert"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def int(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("int", getattr(msc_ops, "int"))(*args, **kwargs)  # pragma: no cover


int16 = msc_ops.int16

int32 = msc_ops.int32

int4 = msc_ops.int4

int64 = msc_ops.int64

int8 = msc_ops.int8

int_ = msc_ops.int_

integer = msc_ops.integer


def interp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("interp", getattr(msc_ops, "interp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def intersect1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("intersect1d", getattr(msc_ops, "intersect1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def inv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("inv", getattr(msc_ops, "inv"))(*args, **kwargs)  # pragma: no cover


def invert(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("invert", getattr(msc_ops, "invert"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def invert_permutation(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "invert_permutation", getattr(msc_ops, "invert_permutation")
    )(  # pragma: no cover
        *args, **kwargs
    )


def iou(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("iou", getattr(msc_ops, "iou"))(*args, **kwargs)  # pragma: no cover


def irfft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("irfft", getattr(msc_ops, "irfft"))(
        *args, **kwargs
    )  # pragma: no cover


def is_tensor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("is_tensor", getattr(msc_ops, "is_tensor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isclose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isclose", getattr(msc_ops, "isclose"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def iscomplex(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("iscomplex", getattr(msc_ops, "iscomplex"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def iscomplexobj(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "iscomplexobj", getattr(msc_ops, "iscomplexobj")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isdtype(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isdtype", getattr(msc_ops, "isdtype"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isfinite(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isfinite", getattr(msc_ops, "isfinite"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isin", getattr(msc_ops, "isin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isinf(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isinf", getattr(msc_ops, "isinf"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isnan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isnan", getattr(msc_ops, "isnan"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isneginf(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isneginf", getattr(msc_ops, "isneginf"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isposinf(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isposinf", getattr(msc_ops, "isposinf"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isreal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isreal", getattr(msc_ops, "isreal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isrealobj(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isrealobj", getattr(msc_ops, "isrealobj"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def isscalar(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("isscalar", getattr(msc_ops, "isscalar"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def issubdtype(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("issubdtype", getattr(msc_ops, "issubdtype"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def istft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("istft", getattr(msc_ops, "istft"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def iterable(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("iterable", getattr(msc_ops, "iterable"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ix_(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ix_", getattr(msc_ops, "ix_"))(*args, **kwargs)  # pragma: no cover


def kaiser(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("kaiser", getattr(msc_ops, "kaiser"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def kron(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("kron", getattr(msc_ops, "kron"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lbeta(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lbeta", getattr(msc_ops, "lbeta"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lcm(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lcm", getattr(msc_ops, "lcm"))(*args, **kwargs)  # pragma: no cover


def ldexp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ldexp", getattr(msc_ops, "ldexp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def leaky_relu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("leaky_relu", getattr(msc_ops, "leaky_relu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def left_shift(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("left_shift", getattr(msc_ops, "left_shift"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def less(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("less", getattr(msc_ops, "less"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def less_equal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("less_equal", getattr(msc_ops, "less_equal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lexsort(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lexsort", getattr(msc_ops, "lexsort"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lgamma(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lgamma", getattr(msc_ops, "lgamma"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


linalg = msc_ops.linalg


def linspace(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("linspace", getattr(msc_ops, "linspace"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def load(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("load", getattr(msc_ops, "load"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def local_response_normalization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "local_response_normalization", getattr(msc_ops, "local_response_normalization")
    )(*args, **kwargs)


def log(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log", getattr(msc_ops, "log"))(*args, **kwargs)  # pragma: no cover


def log10(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log10", getattr(msc_ops, "log10"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def log1p(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log1p", getattr(msc_ops, "log1p"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def log2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log2", getattr(msc_ops, "log2"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def log_sigmoid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log_sigmoid", getattr(msc_ops, "log_sigmoid"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def log_softmax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("log_softmax", getattr(msc_ops, "log_softmax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logaddexp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logaddexp", getattr(msc_ops, "logaddexp"))(
        *args, **kwargs
    )  # pragma: no cover


def logaddexp2(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logaddexp2", getattr(msc_ops, "logaddexp2"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logdet(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logdet", getattr(msc_ops, "logdet"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logical_and(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logical_and", getattr(msc_ops, "logical_and"))(
        *args, **kwargs
    )  # pragma: no cover


def logical_not(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logical_not", getattr(msc_ops, "logical_not"))(
        *args, **kwargs
    )  # pragma: no cover


def logical_or(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logical_or", getattr(msc_ops, "logical_or"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logical_xor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logical_xor", getattr(msc_ops, "logical_xor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logit", getattr(msc_ops, "logit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logspace(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logspace", getattr(msc_ops, "logspace"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def logsumexp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("logsumexp", getattr(msc_ops, "logsumexp"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lookup(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lookup", getattr(msc_ops, "lookup"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lstm_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lstm_cell", getattr(msc_ops, "lstm_cell"))(
        *args, **kwargs
    )  # pragma: no cover


def lstsq(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lstsq", getattr(msc_ops, "lstsq"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def lu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lu", getattr(msc_ops, "lu"))(*args, **kwargs)  # pragma: no cover


def lu_factor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("lu_factor", getattr(msc_ops, "lu_factor"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def map(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("map", getattr(msc_ops, "map"))(*args, **kwargs)  # pragma: no cover


def mask_indices(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "mask_indices", getattr(msc_ops, "mask_indices")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


math = msc_ops.math


def matmul(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("matmul", getattr(msc_ops, "matmul"))(
        *args, **kwargs
    )  # pragma: no cover


def matrix_power(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "matrix_power", getattr(msc_ops, "matrix_power")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def matrix_transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "matrix_transpose", getattr(msc_ops, "matrix_transpose")
    )(  # pragma: no cover
        *args, **kwargs
    )


def max(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("max", getattr(msc_ops, "max"))(*args, **kwargs)  # pragma: no cover


def max_pool(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("max_pool", getattr(msc_ops, "max_pool"))(
        *args, **kwargs
    )  # pragma: no cover


def maximum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("maximum", getattr(msc_ops, "maximum"))(
        *args, **kwargs
    )  # pragma: no cover


def mean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("mean", getattr(msc_ops, "mean"))(
        *args, **kwargs
    )  # pragma: no cover


def median(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("median", getattr(msc_ops, "median"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def median_filter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "median_filter", getattr(msc_ops, "median_filter")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def mel_filterbank(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "mel_filterbank", getattr(msc_ops, "mel_filterbank")
    )(  # pragma: no cover
        *args, **kwargs
    )


def mel_spectrogram(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "mel_spectrogram", getattr(msc_ops, "mel_spectrogram")
    )(  # pragma: no cover
        *args, **kwargs
    )


def meshgrid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("meshgrid", getattr(msc_ops, "meshgrid"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def mfcc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("mfcc", getattr(msc_ops, "mfcc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


mgrid = msc_ops.mgrid


def min(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("min", getattr(msc_ops, "min"))(*args, **kwargs)  # pragma: no cover


def minimum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("minimum", getattr(msc_ops, "minimum"))(
        *args, **kwargs
    )  # pragma: no cover


def mixup(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("mixup", getattr(msc_ops, "mixup"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def mod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("mod", getattr(msc_ops, "mod"))(*args, **kwargs)  # pragma: no cover


def modf(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("modf", getattr(msc_ops, "modf"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def moments(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("moments", getattr(msc_ops, "moments"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def moveaxis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("moveaxis", getattr(msc_ops, "moveaxis"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def multi_hot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("multi_hot", getattr(msc_ops, "multi_hot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def multiply(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("multiply", getattr(msc_ops, "multiply"))(
        *args, **kwargs
    )  # pragma: no cover


def mvlgamma(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("mvlgamma", getattr(msc_ops, "mvlgamma"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nan_to_num(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nan_to_num", getattr(msc_ops, "nan_to_num"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanargmax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanargmax", getattr(msc_ops, "nanargmax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanargmin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanargmin", getattr(msc_ops, "nanargmin"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nancumprod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nancumprod", getattr(msc_ops, "nancumprod"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nancumsum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nancumsum", getattr(msc_ops, "nancumsum"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanmean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanmean", getattr(msc_ops, "nanmean"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanmedian(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanmedian", getattr(msc_ops, "nanmedian"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanpercentile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "nanpercentile", getattr(msc_ops, "nanpercentile")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanquantile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanquantile", getattr(msc_ops, "nanquantile"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanstd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanstd", getattr(msc_ops, "nanstd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def nanvar(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nanvar", getattr(msc_ops, "nanvar"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ndarray(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ndarray", getattr(msc_ops, "ndarray"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ndim(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ndim", getattr(msc_ops, "ndim"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ndtri(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ndtri", getattr(msc_ops, "ndtri"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def negative(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("negative", getattr(msc_ops, "negative"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


newaxis = msc_ops.newaxis


def nextafter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nextafter", getattr(msc_ops, "nextafter"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


nn = msc_ops.nn


# Missing nn ops that are at the top level of msc_ops
nn.moments = _wrap_op("moments", msc_ops.moments)
nn.multi_hot = _wrap_op("multi_hot", msc_ops.multi_hot)
nn.normalize = _wrap_op("normalize", msc_ops.normalize)
nn.one_hot = _wrap_op("one_hot", msc_ops.one_hot)
nn.polar = _wrap_op("polar", msc_ops.polar)
nn.psnr = _wrap_op("psnr", msc_ops.psnr)


class NumpyWrapper:
    pass  # pragma: no cover


numpy = NumpyWrapper()

numpy_ops = [
    "abs",
    "absolute",
    "add",
    "all",
    "amax",
    "amin",
    "angle",
    "any",
    "append",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "array",
    "average",
    "bartlett",
    "bincount",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman",
    "broadcast_to",
    "ceil",
    "clip",
    "concatenate",
    "conj",
    "conjugate",
    "copy",
    "correlate",
    "cos",
    "cosh",
    "count_nonzero",
    "cross",
    "cumprod",
    "cumsum",
    "diag",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "divide",
    "divide_no_nan",
    "dot",
    "einsum",
    "empty",
    "equal",
    "exp",
    "exp2",
    "expand_dims",
    "expm1",
    "eye",
    "flip",
    "floor",
    "floor_divide",
    "full",
    "full_like",
    "get_item",
    "greater",
    "greater_equal",
    "hamming",
    "histogram",
    "hstack",
    "identity",
    "imag",
    "inner",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "left_shift",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "matmul",
    "max",
    "maximum",
    "mean",
    "median",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "moveaxis",
    "multiply",
    "nan_to_num",
    "ndim",
    "negative",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "pad",
    "power",
    "prod",
    "quantile",
    "ravel",
    "real",
    "reciprocal",
    "repeat",
    "reshape",
    "right_shift",
    "roll",
    "rot90",
    "round",
    "select",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "size",
    "slogdet",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "trace",
    "transpose",
    "tri",
    "tril",
    "triu",
    "true_divide",
    "trunc",
    "unravel_index",
    "var",
    "vdot",
    "vectorize",
    "vstack",
    "where",
    "zeros",
    "zeros_like",
]

for op_name in numpy_ops:
    if hasattr(msc_ops, op_name):  # pragma: no cover
        setattr(numpy, op_name, _wrap_op(op_name, getattr(msc_ops, op_name)))
    else:
        # Fallback to dummy wrap
        setattr(numpy, op_name, _wrap_op(op_name, lambda *args, **kwargs: args[0]))


def non_max_suppression(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "non_max_suppression", getattr(msc_ops, "non_max_suppression")
    )(  # pragma: no cover
        *args, **kwargs
    )


def nonzero(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("nonzero", getattr(msc_ops, "nonzero"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def norm(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("norm", getattr(msc_ops, "norm"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


normalization = msc_ops.normalization


def normalize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("normalize", getattr(msc_ops, "normalize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def not_equal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("not_equal", getattr(msc_ops, "not_equal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


number = msc_ops.number


def numpy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("numpy", getattr(msc_ops, "numpy"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


object_ = msc_ops.object_

ogrid = msc_ops.ogrid


def one_hot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("one_hot", getattr(msc_ops, "one_hot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ones(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ones", getattr(msc_ops, "ones"))(
        *args, **kwargs
    )  # pragma: no cover


def ones_like(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ones_like", getattr(msc_ops, "ones_like"))(
        *args, **kwargs
    )  # pragma: no cover


def outer(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("outer", getattr(msc_ops, "outer"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def packbits(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("packbits", getattr(msc_ops, "packbits"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pad(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pad", getattr(msc_ops, "pad"))(*args, **kwargs)  # pragma: no cover


def pad_to_bounding_box(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "pad_to_bounding_box", getattr(msc_ops, "pad_to_bounding_box")
    )(  # pragma: no cover
        *args, **kwargs
    )


def partition(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("partition", getattr(msc_ops, "partition"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def percentile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("percentile", getattr(msc_ops, "percentile"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def permute(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("permute", getattr(msc_ops, "permute"))(
        *args, **kwargs
    )  # pragma: no cover


def permute_dims(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "permute_dims", getattr(msc_ops, "permute_dims")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def perspective_transform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "perspective_transform", getattr(msc_ops, "perspective_transform")
    )(  # pragma: no cover
        *args, **kwargs
    )


pi = msc_ops.pi


def piecewise(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("piecewise", getattr(msc_ops, "piecewise"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pinv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pinv", getattr(msc_ops, "pinv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def place(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("place", getattr(msc_ops, "place"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pmean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pmean", getattr(msc_ops, "pmean"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polar(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polar", getattr(msc_ops, "polar"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def poly(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("poly", getattr(msc_ops, "poly"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polyadd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polyadd", getattr(msc_ops, "polyadd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polyder(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polyder", getattr(msc_ops, "polyder"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polydiv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polydiv", getattr(msc_ops, "polydiv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polyfit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polyfit", getattr(msc_ops, "polyfit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polygamma(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polygamma", getattr(msc_ops, "polygamma"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polyint(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polyint", getattr(msc_ops, "polyint"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polymul(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polymul", getattr(msc_ops, "polymul"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polysub(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polysub", getattr(msc_ops, "polysub"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def polyval(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("polyval", getattr(msc_ops, "polyval"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pool1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pool1d", getattr(msc_ops, "pool1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pool2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pool2d", getattr(msc_ops, "pool2d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pool3d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pool3d", getattr(msc_ops, "pool3d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def positive(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("positive", getattr(msc_ops, "positive"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def posterize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("posterize", getattr(msc_ops, "posterize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def pow(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("pow", getattr(msc_ops, "pow"))(*args, **kwargs)  # pragma: no cover


def power(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("power", getattr(msc_ops, "power"))(
        *args, **kwargs
    )  # pragma: no cover


def power_iteration(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "power_iteration", getattr(msc_ops, "power_iteration")
    )(  # pragma: no cover
        *args, **kwargs
    )


def printoptions(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "printoptions", getattr(msc_ops, "printoptions")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def prod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("prod", getattr(msc_ops, "prod"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def promote_types(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "promote_types", getattr(msc_ops, "promote_types")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def psnr(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("psnr", getattr(msc_ops, "psnr"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def psum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("psum", getattr(msc_ops, "psum"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ptp(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ptp", getattr(msc_ops, "ptp"))(*args, **kwargs)  # pragma: no cover


def put(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("put", getattr(msc_ops, "put"))(*args, **kwargs)  # pragma: no cover


def qr(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("qr", getattr(msc_ops, "qr"))(*args, **kwargs)  # pragma: no cover


def quantile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("quantile", getattr(msc_ops, "quantile"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


r_ = msc_ops.r_


def rad2deg(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rad2deg", getattr(msc_ops, "rad2deg"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def radians(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("radians", getattr(msc_ops, "radians"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


ragged = msc_ops.ragged


def rand_augment(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "rand_augment", getattr(msc_ops, "rand_augment")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def random_color_jitter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_color_jitter", getattr(msc_ops, "random_color_jitter")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_crop(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("random_crop", getattr(msc_ops, "random_crop"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def random_elastic_transform(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_elastic_transform", getattr(msc_ops, "random_elastic_transform")
    )(*args, **kwargs)


def random_erasing(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_erasing", getattr(msc_ops, "random_erasing")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_flip(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("random_flip", getattr(msc_ops, "random_flip"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def random_gaussian_blur(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_gaussian_blur", getattr(msc_ops, "random_gaussian_blur")
    )(  # pragma: no cover
        *args, **kwargs
    )


random_ops = msc_ops.random_ops


def random_perspective(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_perspective", getattr(msc_ops, "random_perspective")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_rotation(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_rotation", getattr(msc_ops, "random_rotation")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_sharpness(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_sharpness", getattr(msc_ops, "random_sharpness")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_shear(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "random_shear", getattr(msc_ops, "random_shear")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def random_translation(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "random_translation", getattr(msc_ops, "random_translation")
    )(  # pragma: no cover
        *args, **kwargs
    )


def random_zoom(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("random_zoom", getattr(msc_ops, "random_zoom"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ravel(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("ravel", getattr(msc_ops, "ravel"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def ravel_multi_index(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "ravel_multi_index", getattr(msc_ops, "ravel_multi_index")
    )(  # pragma: no cover
        *args, **kwargs
    )


def real(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("real", getattr(msc_ops, "real"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rearrange(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rearrange", getattr(msc_ops, "rearrange"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def reciprocal(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("reciprocal", getattr(msc_ops, "reciprocal"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def reduce_scatter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "reduce_scatter", getattr(msc_ops, "reduce_scatter")
    )(  # pragma: no cover
        *args, **kwargs
    )


def reduce_window(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "reduce_window", getattr(msc_ops, "reduce_window")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


reductions = msc_ops.reductions


def regex_replace(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "regex_replace", getattr(msc_ops, "regex_replace")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def register_op(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("register_op", getattr(msc_ops, "register_op"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def relu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("relu", getattr(msc_ops, "relu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def relu6(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("relu6", getattr(msc_ops, "relu6"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def remainder(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("remainder", getattr(msc_ops, "remainder"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def repeat(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("repeat", getattr(msc_ops, "repeat"))(
        *args, **kwargs
    )  # pragma: no cover


def reshape(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("reshape", getattr(msc_ops, "reshape"))(
        *args, **kwargs
    )  # pragma: no cover


def resize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("resize", getattr(msc_ops, "resize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def resize_bicubic(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "resize_bicubic", getattr(msc_ops, "resize_bicubic")
    )(  # pragma: no cover
        *args, **kwargs
    )


def resize_bilinear(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "resize_bilinear", getattr(msc_ops, "resize_bilinear")
    )(  # pragma: no cover
        *args, **kwargs
    )


def resize_lanczos3(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "resize_lanczos3", getattr(msc_ops, "resize_lanczos3")
    )(  # pragma: no cover
        *args, **kwargs
    )


def resize_nearest(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "resize_nearest", getattr(msc_ops, "resize_nearest")
    )(  # pragma: no cover
        *args, **kwargs
    )


def result_type(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("result_type", getattr(msc_ops, "result_type"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rfft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rfft", getattr(msc_ops, "rfft"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rgb_to_grayscale(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "rgb_to_grayscale", getattr(msc_ops, "rgb_to_grayscale")
    )(  # pragma: no cover
        *args, **kwargs
    )


def rgb_to_hsv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rgb_to_hsv", getattr(msc_ops, "rgb_to_hsv"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def right_shift(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("right_shift", getattr(msc_ops, "right_shift"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rint(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rint", getattr(msc_ops, "rint"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rms_normalization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "rms_normalization", getattr(msc_ops, "rms_normalization")
    )(  # pragma: no cover
        *args, **kwargs
    )


def rnn(  # pragma: no cover
    inputs,
    initial_state,
    cell_fn,
    time_major=False,
    go_backwards=False,
    unroll=False,
    return_all_outputs=True,
    **kwargs,
):
    """Function docstring.

    Args:
        inputs: Description.
        initial_state: Description.
        cell_fn: Description.
        time_major: Description.
        go_backwards: Description.
        unroll: Description.
        return_all_outputs: Description.
        kwargs: Description.
    """
    from ml_switcheroo_compiler.ops.nn.rnn_utils import RNNConfig

    config = RNNConfig(
        time_major=time_major,
        go_backwards=go_backwards,
        unroll=unroll,
        return_all_outputs=return_all_outputs,
    )
    return _wrap_op("rnn", getattr(msc_ops, "rnn"))(  # pragma: no cover
        inputs, initial_state, cell_fn, config=config
    )


def roll(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("roll", getattr(msc_ops, "roll"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rollaxis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rollaxis", getattr(msc_ops, "rollaxis"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def roots(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("roots", getattr(msc_ops, "roots"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rot90(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rot90", getattr(msc_ops, "rot90"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def round(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("round", getattr(msc_ops, "round"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def round_(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("round_", getattr(msc_ops, "round_"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def rsqrt(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("rsqrt", getattr(msc_ops, "rsqrt"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


s_ = msc_ops.s_


def saturate_cast(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "saturate_cast", getattr(msc_ops, "saturate_cast")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def save(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("save", getattr(msc_ops, "save"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def savez(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("savez", getattr(msc_ops, "savez"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def scan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("scan", getattr(msc_ops, "scan"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def scatter(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("scatter", getattr(msc_ops, "scatter"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def scatter_add(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("scatter_add", getattr(msc_ops, "scatter_add"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def scatter_nd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("scatter_nd", getattr(msc_ops, "scatter_nd"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def scatter_update(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "scatter_update", getattr(msc_ops, "scatter_update")
    )(  # pragma: no cover
        *args, **kwargs
    )


def searchsorted(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "searchsorted", getattr(msc_ops, "searchsorted")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def segment_max(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("segment_max", getattr(msc_ops, "segment_max"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def segment_mean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "segment_mean", getattr(msc_ops, "segment_mean")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def segment_min(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("segment_min", getattr(msc_ops, "segment_min"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def segment_prod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "segment_prod", getattr(msc_ops, "segment_prod")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def segment_sum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("segment_sum", getattr(msc_ops, "segment_sum"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def select(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("select", getattr(msc_ops, "select"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def selu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("selu", getattr(msc_ops, "selu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def separable_conv(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "separable_conv", getattr(msc_ops, "separable_conv")
    )(  # pragma: no cover
        *args, **kwargs
    )


def separable_conv1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "separable_conv1d", getattr(msc_ops, "separable_conv1d")
    )(  # pragma: no cover
        *args, **kwargs
    )


def separable_conv2d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "separable_conv2d", getattr(msc_ops, "separable_conv2d")
    )(  # pragma: no cover
        *args, **kwargs
    )


def set_printoptions(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "set_printoptions", getattr(msc_ops, "set_printoptions")
    )(  # pragma: no cover
        *args, **kwargs
    )


def setdiff1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("setdiff1d", getattr(msc_ops, "setdiff1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def setxor1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("setxor1d", getattr(msc_ops, "setxor1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


shape = msc_ops.shape

shape_inference = msc_ops.shape_inference


def shard_tensor(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "shard_tensor", getattr(msc_ops, "shard_tensor")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def sharpen(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sharpen", getattr(msc_ops, "sharpen"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def sigmoid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sigmoid", getattr(msc_ops, "sigmoid"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def sign(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sign", getattr(msc_ops, "sign"))(
        *args, **kwargs
    )  # pragma: no cover


def signbit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("signbit", getattr(msc_ops, "signbit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


signedinteger = msc_ops.signedinteger


def silu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("silu", getattr(msc_ops, "silu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def simple_rnn_cell(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "simple_rnn_cell", getattr(msc_ops, "simple_rnn_cell")
    )(  # pragma: no cover
        *args, **kwargs
    )


def sin(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sin", getattr(msc_ops, "sin"))(*args, **kwargs)  # pragma: no cover


def sinc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sinc", getattr(msc_ops, "sinc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


single = msc_ops.single


def sinh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sinh", getattr(msc_ops, "sinh"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def size(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("size", getattr(msc_ops, "size"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("slice", getattr(msc_ops, "slice"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def slice_update(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "slice_update", getattr(msc_ops, "slice_update")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def slogdet(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("slogdet", getattr(msc_ops, "slogdet"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def sobol_sample(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "sobol_sample", getattr(msc_ops, "sobol_sample")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def soft_shrink(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("soft_shrink", getattr(msc_ops, "soft_shrink"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def softmax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("softmax", getattr(msc_ops, "softmax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def softplus(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("softplus", getattr(msc_ops, "softplus"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def softsign(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("softsign", getattr(msc_ops, "softsign"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def solarize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("solarize", getattr(msc_ops, "solarize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def solve(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("solve", getattr(msc_ops, "solve"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def solve_triangular(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "solve_triangular", getattr(msc_ops, "solve_triangular")
    )(  # pragma: no cover
        *args, **kwargs
    )


def sort(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sort", getattr(msc_ops, "sort"))(
        *args, **kwargs
    )  # pragma: no cover


def sort_complex(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "sort_complex", getattr(msc_ops, "sort_complex")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


sparse = msc_ops.sparse


def sparse_categorical_crossentropy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "sparse_categorical_crossentropy",
        getattr(msc_ops, "sparse_categorical_crossentropy"),
    )(*args, **kwargs)


def sparse_plus(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sparse_plus", getattr(msc_ops, "sparse_plus"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def sparse_sigmoid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "sparse_sigmoid", getattr(msc_ops, "sparse_sigmoid")
    )(  # pragma: no cover
        *args, **kwargs
    )


def sparsemax(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sparsemax", getattr(msc_ops, "sparsemax"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


special = msc_ops.special


def spectral_normalization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "spectral_normalization", getattr(msc_ops, "spectral_normalization")
    )(*args, **kwargs)


def split(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("split", getattr(msc_ops, "split"))(
        *args, **kwargs
    )  # pragma: no cover


def sqrt(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sqrt", getattr(msc_ops, "sqrt"))(
        *args, **kwargs
    )  # pragma: no cover


def square(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("square", getattr(msc_ops, "square"))(
        *args, **kwargs
    )  # pragma: no cover


def squareplus(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("squareplus", getattr(msc_ops, "squareplus"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def squeeze(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("squeeze", getattr(msc_ops, "squeeze"))(
        *args, **kwargs
    )  # pragma: no cover


def stack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("stack", getattr(msc_ops, "stack"))(
        *args, **kwargs
    )  # pragma: no cover


state = msc_ops.state


def std(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("std", getattr(msc_ops, "std"))(*args, **kwargs)  # pragma: no cover


def stft(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("stft", getattr(msc_ops, "stft"))(
        *args, **kwargs
    )  # pragma: no cover


def stop_gradient(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("stop_gradient", getattr(msc_ops, "stop_gradient"))(
        *args, **kwargs
    )  # pragma: no cover


def strided_slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "strided_slice", getattr(msc_ops, "strided_slice")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def string_lower(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "string_lower", getattr(msc_ops, "string_lower")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def string_split(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "string_split", getattr(msc_ops, "string_split")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def string_to_hash(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "string_to_hash", getattr(msc_ops, "string_to_hash")
    )(  # pragma: no cover
        *args, **kwargs
    )


def string_to_number(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "string_to_number", getattr(msc_ops, "string_to_number")
    )(  # pragma: no cover
        *args, **kwargs
    )


def string_upper(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "string_upper", getattr(msc_ops, "string_upper")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def subtract(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("subtract", getattr(msc_ops, "subtract"))(
        *args, **kwargs
    )  # pragma: no cover


def sum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("sum", getattr(msc_ops, "sum"))(*args, **kwargs)  # pragma: no cover


def svd(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("svd", getattr(msc_ops, "svd"))(*args, **kwargs)  # pragma: no cover


def swapaxes(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("swapaxes", getattr(msc_ops, "swapaxes"))(
        *args, **kwargs
    )  # pragma: no cover


def swish(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("swish", getattr(msc_ops, "swish"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def switch(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("switch", getattr(msc_ops, "switch"))(
        *args, **kwargs
    )  # pragma: no cover


def take(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("take", getattr(msc_ops, "take"))(
        *args, **kwargs
    )  # pragma: no cover


def take_along_axis(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "take_along_axis", getattr(msc_ops, "take_along_axis")
    )(  # pragma: no cover
        *args, **kwargs
    )


def tan(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tan", getattr(msc_ops, "tan"))(*args, **kwargs)  # pragma: no cover


def tanh(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tanh", getattr(msc_ops, "tanh"))(
        *args, **kwargs
    )  # pragma: no cover


def tanh_shrink(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tanh_shrink", getattr(msc_ops, "tanh_shrink"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


tensor_array = msc_ops.tensor_array


def tensor_scatter_add(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "tensor_scatter_add", getattr(msc_ops, "tensor_scatter_add")
    )(  # pragma: no cover
        *args, **kwargs
    )


def tensor_scatter_max(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "tensor_scatter_max", getattr(msc_ops, "tensor_scatter_max")
    )(  # pragma: no cover
        *args, **kwargs
    )


def tensor_scatter_min(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "tensor_scatter_min", getattr(msc_ops, "tensor_scatter_min")
    )(  # pragma: no cover
        *args, **kwargs
    )


def tensor_scatter_update(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "tensor_scatter_update", getattr(msc_ops, "tensor_scatter_update")
    )(  # pragma: no cover
        *args, **kwargs
    )


def tensordot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tensordot", getattr(msc_ops, "tensordot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


text = msc_ops.text


def text_vectorization(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "text_vectorization", getattr(msc_ops, "text_vectorization")
    )(  # pragma: no cover
        *args, **kwargs
    )


def threshold(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("threshold", getattr(msc_ops, "threshold"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def tile(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tile", getattr(msc_ops, "tile"))(
        *args, **kwargs
    )  # pragma: no cover


def time_distributed(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "time_distributed", getattr(msc_ops, "time_distributed")
    )(  # pragma: no cover
        *args, **kwargs
    )


def top_k(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("top_k", getattr(msc_ops, "top_k"))(
        *args, **kwargs
    )  # pragma: no cover


def trace(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("trace", getattr(msc_ops, "trace"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def transpose(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("transpose", getattr(msc_ops, "transpose"))(
        *args, **kwargs
    )  # pragma: no cover


def trapezoid(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("trapezoid", getattr(msc_ops, "trapezoid"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def tri(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tri", getattr(msc_ops, "tri"))(*args, **kwargs)  # pragma: no cover


def tril(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("tril", getattr(msc_ops, "tril"))(
        *args, **kwargs
    )  # pragma: no cover


def tril_indices(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "tril_indices", getattr(msc_ops, "tril_indices")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def tril_indices_from(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "tril_indices_from", getattr(msc_ops, "tril_indices_from")
    )(  # pragma: no cover
        *args, **kwargs
    )


def trim_zeros(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("trim_zeros", getattr(msc_ops, "trim_zeros"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def triu(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("triu", getattr(msc_ops, "triu"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def triu_indices(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "triu_indices", getattr(msc_ops, "triu_indices")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def triu_indices_from(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "triu_indices_from", getattr(msc_ops, "triu_indices_from")
    )(  # pragma: no cover
        *args, **kwargs
    )


def true_divide(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("true_divide", getattr(msc_ops, "true_divide"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def trunc(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("trunc", getattr(msc_ops, "trunc"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


type_inference = msc_ops.type_inference

ufunc = msc_ops.ufunc

uint = msc_ops.uint

uint16 = msc_ops.uint16

uint32 = msc_ops.uint32

uint4 = msc_ops.uint4

uint64 = msc_ops.uint64

uint8 = msc_ops.uint8

unary = msc_ops.unary


def union1d(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("union1d", getattr(msc_ops, "union1d"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unique(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unique", getattr(msc_ops, "unique"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unique_all(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unique_all", getattr(msc_ops, "unique_all"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unique_counts(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "unique_counts", getattr(msc_ops, "unique_counts")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unique_inverse(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unique_inverse", getattr(msc_ops, "unique_inverse")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unique_values(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "unique_values", getattr(msc_ops, "unique_values")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unpackbits(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unpackbits", getattr(msc_ops, "unpackbits"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unravel_index(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "unravel_index", getattr(msc_ops, "unravel_index")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


unsignedinteger = msc_ops.unsignedinteger


def unsorted_segment_max(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_max", getattr(msc_ops, "unsorted_segment_max")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unsorted_segment_mean(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_mean", getattr(msc_ops, "unsorted_segment_mean")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unsorted_segment_min(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_min", getattr(msc_ops, "unsorted_segment_min")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unsorted_segment_prod(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_prod", getattr(msc_ops, "unsorted_segment_prod")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unsorted_segment_sqrt_n(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_sqrt_n", getattr(msc_ops, "unsorted_segment_sqrt_n")
    )(*args, **kwargs)


def unsorted_segment_sum(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "unsorted_segment_sum", getattr(msc_ops, "unsorted_segment_sum")
    )(  # pragma: no cover
        *args, **kwargs
    )


def unsqueeze(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unsqueeze", getattr(msc_ops, "unsqueeze"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unstack", getattr(msc_ops, "unstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def unwrap(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("unwrap", getattr(msc_ops, "unwrap"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def update_slice(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "update_slice", getattr(msc_ops, "update_slice")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vander(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vander", getattr(msc_ops, "vander"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def var(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("var", getattr(msc_ops, "var"))(*args, **kwargs)  # pragma: no cover


def variance(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("variance", getattr(msc_ops, "variance"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vdot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vdot", getattr(msc_ops, "vdot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vecdot(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vecdot", getattr(msc_ops, "vecdot"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vectorize(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vectorize", getattr(msc_ops, "vectorize"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vectorized_map(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "vectorized_map", getattr(msc_ops, "vectorized_map")
    )(  # pragma: no cover
        *args, **kwargs
    )


def view_as_complex(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(  # pragma: no cover
        "view_as_complex", getattr(msc_ops, "view_as_complex")
    )(  # pragma: no cover
        *args, **kwargs
    )


def view_as_real(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op(
        "view_as_real", getattr(msc_ops, "view_as_real")
    )(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


vision = msc_ops.vision


def vsplit(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vsplit", getattr(msc_ops, "vsplit"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def vstack(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("vstack", getattr(msc_ops, "vstack"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def where(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("where", getattr(msc_ops, "where"))(
        *args, **kwargs
    )  # pragma: no cover


def while_loop(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("while_loop", getattr(msc_ops, "while_loop"))(
        *args, **kwargs
    )  # pragma: no cover


def xlogy(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("xlogy", getattr(msc_ops, "xlogy"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


def zeros(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("zeros", getattr(msc_ops, "zeros"))(
        *args, **kwargs
    )  # pragma: no cover


def zeros_like(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("zeros_like", getattr(msc_ops, "zeros_like"))(
        *args, **kwargs
    )  # pragma: no cover


def zeta(*args, **kwargs):  # pragma: no cover
    """Function docstring.

    Args:
        args: Description.
        kwargs: Description.
    """
    return _wrap_op("zeta", getattr(msc_ops, "zeta"))(  # pragma: no cover
        *args, **kwargs
    )  # pragma: no cover


import sys

ops = sys.modules[__name__]
