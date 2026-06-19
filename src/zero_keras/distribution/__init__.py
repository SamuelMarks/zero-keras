"""Distribution API."""

import threading

_distribution_context = threading.local()


class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).

    Args:
        shape: tuple of list of integers. The shape of the overall
            `DeviceMesh`, e.g. `(8,)` for a data parallel only distribution,
            or `(4, 2)` for a model+data parallel distribution.
        axis_names: List of string. The logical name of the each axis for
            the `DeviceMesh`. The length of the `axis_names` should match to
            the rank of the `shape`. The `axis_names` will be used to
            match/create the `TensorLayout` when distribute the data and
            variables.
        devices: Optional list of devices. Defaults to all the available
            devices locally from `keras.distribution.list_devices()`.

    """

    def __init__(self, shape, axis_names, devices=None):
        self.shape = shape
        self.axis_names = axis_names
        self.devices = devices


class LayoutMap:
    """A dict-like object that maps string to `TensorLayout` instances.

    `LayoutMap` uses a string as key and a `TensorLayout` as value. There is a
    behavior difference between a normal Python dict and this class. The string
    key will be treated as a regex when retrieving the value. See the docstring
    of `get` for more details.

    See below for a usage example. You can define the naming schema
    of the `TensorLayout`, and then retrieve the corresponding
    `TensorLayout` instance.

    In the normal case, the key to query is usually the `variable.path`, which
    is the identifier of the variable.

    As shortcut, tuple or list of axis names are also allowed when inserting
    as value, and will be converted to `TensorLayout`.

    ```python
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    layout_1 = layout_map['dense_1.kernel']             # layout_1 == layout_2d
    layout_2 = layout_map['dense_1.bias']               # layout_2 == layout_1d
    layout_3 = layout_map['dense_2.kernel']             # layout_3 == layout_2d
    layout_4 = layout_map['dense_2.bias']               # layout_4 == layout_1d
    layout_5 = layout_map['my_model/conv2d_123/kernel'] # layout_5 == layout_4d
    layout_6 = layout_map['my_model/conv2d_123/bias']   # layout_6 == layout_1d
    layout_7 = layout_map['my_model/conv3d_1/kernel']   # layout_7 == None
    layout_8 = layout_map['my_model/conv3d_1/bias']     # layout_8 == None
    ```

    Args:
        device_mesh: `keras.distribution.DeviceMesh` instance.

    """

    def __init__(self, device_mesh=None):
        self.device_mesh = device_mesh
        self._map = {}

    def __setitem__(self, key, value):
        """Insert TensorLayout to the LayoutMap.

        Args:
            key: String key for the `TensorLayout`.
            layout: The `TensorLayout`. As a shortcut, tuple of string and None
                are also acceptable, and will be converted to `TensorLayout`.

        """
        self._map[key] = value

    def __getitem__(self, key):
        """Retrieves the corresponding layout by the string key.

        When there isn't an exact match, all the existing keys in the layout map
        will be treated as a regex and map against the input key again. When
        there are multiple matches for the regex, an `ValueError` will be
        raised. Returns `None` if there isn't any match found.

        Args:
            key: String key to query a layout.

        Returns:
            Corresponding layout based on the query.

        """
        return self._map.get(key)

    def get(self, key, default=None):
        """Retrieves the corresponding layout by the string key.

        When there isn't an exact match, all the existing keys in the layout map
        will be treated as a regex and map against the input key again. When
        there are multiple matches for the regex, an `ValueError` will be
        raised. Returns `None` if there isn't any match found.

        Args:
            key: String key to query a layout.

        Returns:
            Corresponding layout based on the query.

        """
        return self._map.get(key, default)


class Distribution:
    """Distribution class."""

    def __init__(self, device_mesh=None):
        self.device_mesh = device_mesh

    def scope(self):
        """Scope function.

        Returns:
        Any: Return value.

        """

        class DistributionScope:
            """DistributionScope class."""

            def __init__(self, dist):
                self.dist = dist

            def __enter__(self):
                """__enter__ function.

                Returns:
                Any: Return value.

                """
                _distribution_context.current = self.dist
                return self.dist

            def __exit__(self, *args):
                """__exit__ function.

                *args: Variable length argument list.

                Returns:
                Any: Return value.

                """
                _distribution_context.current = None

        return DistributionScope(self)


def distribution():
    """Retrieve the current distribution from global context."""
    return getattr(_distribution_context, "current", None)


class DataParallel(Distribution):
    """Distribution for data parallelism.

    You can choose to create this instance by either specifying
    the `device_mesh` or `devices` arguments (but not both).

    The `device_mesh` argument is expected to be a `DeviceMesh` instance,
    and is expected to be 1D only. In case that the mesh has multiple axes,
    then the first axis will be treated as the data parallel dimension
    (and a warning will be raised).

    When a list of `devices` are provided, they will be used to construct a
    1D mesh.

    When both `mesh` and `devices` are absent, then `list_devices()`
    will be used to detect any available devices and create a 1D mesh from
    them.

    Args:
        device_mesh: Optional `DeviceMesh` instance.
        devices: Optional list of devices.
        auto_shard_dataset: Automatically shard the dataset amongst processes.
            Defaults to true.

    """

    def __init__(self, device_mesh=None):
        super().__init__(device_mesh=device_mesh)


class ModelParallel(Distribution):
    """Distribution that shards model variables.

    Compare to `DataParallel` which replicates the variables across all devices,
    `ModelParallel` allows you to shard variables in addition to the input data.

    To construct a `ModelParallel` distribution, you need to provide a
    `DeviceMesh` and a `LayoutMap`.

    1. `DeviceMesh` contains physical device information. The axis names in
        the mesh will be used to map the variable and data layout.
    2. `LayoutMap` contains the mapping between variable paths to their
        corresponding `TensorLayout`.

    Example:
    ```python
    devices = list_devices()    # Assume there are 8 devices.

    # Create a mesh with 2 devices for data parallelism and 4 devices for
    # model parallelism.
    device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                             devices=devices)
    # Create a layout map that shard the `Dense` layer and `Conv2D`
    # layer variables on the last dimension.
    # Based on the `device_mesh`, this means the variables
    # will be split across 4 devices. Any other variable that doesn't
    # match any key in the layout map will be fully replicated.
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch',
    )

    # Set the global distribution, or via `with distribution.scope():`
    set_distribution(distribution)

    model = model_creation()
    model.compile()
    model.fit(data)
    ```

    You can quickly update the device mesh shape to change the sharding factor
    of the variables. E.g.

    ```python
    # With only the shape change for the device mesh, the variables will be
    # sharded across 8 devices instead of 4, which further reduces the memory
    # footprint of variables on each of the device.
    device_mesh = DeviceMesh(
        shape=(1, 8),
        axis_names=('batch', 'model'),
        devices=devices,
    )
    ```

    To figure out a proper layout mapping rule for all the model variables, you
    can first list out all the model variable paths, which will be used as the
    key to map the variables to `TensorLayout`.

    e.g.

    ```python
    model = create_model()
    for v in model.variables:
        print(v.path)
    ```

    Args:
        layout_map: `LayoutMap` instance which map the variable path to the
            corresponding tensor layout.
        batch_dim_name: Optional string, the axis name in the device mesh
            (of the `layout_map` object)
            that will be used to distribute data. If unspecified, the
            first axis from the device mesh will be used.

    """

    def __init__(self, device_mesh=None, layout_map=None, batch_dim_name=None):
        super().__init__(device_mesh=device_mesh)
        self.layout_map = layout_map
        self.batch_dim_name = batch_dim_name


def distribute_tensor(tensor, layout):
    """Change the layout of a Tensor value in the jit function execution.

    Args:
        tensor: a Tensor to change the layout.
        layout: `TensorLayout` to be applied on the value.

    Returns:
        a new value with the specified tensor layout.

    """
    dist = distribution()
    if dist is None:
        return tensor

    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        return tensor

    from ml_switcheroo_compiler.ops.distributed import shard_tensor
    from ml_switcheroo_compiler.distributed.device_mesh import (
        DeviceMesh as CompilerDeviceMesh,
    )
    from zero_keras.activations import _to_tensor, _wrap

    tensor_node = _to_tensor(tensor)

    compiler_mesh = CompilerDeviceMesh(
        shape=dist.device_mesh.shape,
        axis_names=dist.device_mesh.axis_names,
        devices=dist.device_mesh.devices,
    )

    out = shard_tensor(tensor_node, compiler_mesh, layout)
    return _wrap(out)
