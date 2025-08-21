import copy
import numbers
import collections.abc

import numpy as np

__all__ = ["Meta"]


class Meta(dict):
    """
    A sliceable object for storing metadata.

    Metadata can be linked to a data axis which causes it to be sliced when the
    standard Python numeric slicing API is applied to the object.
    Specific pieces of metadata can be obtain using the dict-like string slicing API.
    Metadata associated with an axis/axes must have the same length/shape as those axes.

    Parameters
    ----------
    header: dict-like
        The names and values of metadata.

    comments: dict-like, optional
        Comments associated with any of the above pieces of metadata.

    axes: dict-like, optional
        The axis/axes associated with the metadata denoted by the keys.
        Metadata not included are considered not to be associated with any axis.
        Each axis value must be an iterable of `int`. An `int` itself is also
        acceptable if the metadata is associated with a single axis. An empty
        iterable also means the metadata is not associated with any axes.

    data_shape: iterator of `int`, optional
        The shape of the data with which this metadata is associated.
        Must be set if axes input is set.
    """

    def __init__(self, header=None, comments=None, axes=None, data_shape=None):
        self.__ndcube_can_slice__ = True
        self.__ndcube_can_rebin__ = True
        self.original_header = header

        if header is None:
            header = {}
        else:
            header = dict(header)
        super().__init__(header.items())
        header_keys = header.keys()

        if comments is None:
            self._comments = dict()
        else:
            comments = dict(comments)
            if not set(comments.keys()).issubset(set(header_keys)):
                raise ValueError("All comments must correspond to a value in header under the same key.")
            self._comments = comments

        if data_shape is None:
            self._data_shape = data_shape
        else:
            self._data_shape = np.asarray(data_shape, dtype=int)

        if axes is None:
            self._axes = dict()
        else:
            if not (
                isinstance(data_shape, collections.abc.Iterable)
                and all([isinstance(i, numbers.Integral) for i in data_shape])
            ):
                raise TypeError(
                    "If axes is set, data_shape must be an iterable giving "
                    "the length of each axis of the associated cube."
                )
            axes = dict(axes)
            if not set(axes.keys()).issubset(set(header_keys)):
                raise ValueError("All axes must correspond to a value in header under the same key.")
            self._axes = dict([(key, self._sanitize_axis_value(axis, header[key], key)) for key, axis in axes.items()])

    def _sanitize_axis_value(self, axis, value, key):
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if len(axis) == 0:
            return tuple()
        if self.shape is None:
            raise TypeError("Meta instance does not have a shape so new metadata " "cannot be assigned to an axis.")
        # Verify each entry in axes is an iterable of ints.
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if not (isinstance(axis, collections.abc.Iterable) and all([isinstance(i, numbers.Integral) for i in axis])):
            raise TypeError(
                "Values in axes must be an integer or iterable of integers giving "
                "the data axis/axes associated with the metadata."
            )
        axis = np.asarray(axis)

        shape_error_msg = f"{key} must have shape {tuple(self.shape[axis])} " f"as it is associated with axes {axis}"
        if len(axis) == 1:
            if not hasattr(value, "__len__"):
                raise TypeError(shape_error_msg)
            meta_shape = (len(value),)
        else:
            if not hasattr(value, "shape"):
                raise TypeError(shape_error_msg)
            meta_shape = value.shape
        if not all(meta_shape == self.shape[axis]):
            raise ValueError(shape_error_msg)

        return axis

    @property
    def comments(self):
        return self._comments

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self._data_shape

    def add(self, name, value, comment=None, axis=None, overwrite=False):
        """Add a new piece of metadata to instance.

        Parameters
        ----------
        name: `str`
            The name/label of the metadata.

        value: Any
            The value of the metadata. If axes input is not None, this must have the
            same length/shape as those axes as defined by ``self.shape``.

        comment: `str` or `None`
            Any comment associated with this metadata. Set to None if no comment desired.

        axis: `int`, iterable of `int`, or `None`
            The axis/axes with which the metadata is linked. If not associated with any
            axis, set this to None.

        overwrite: `bool`, optional
            If True, overwrites the entry of the name name if already present.
        """
        if name in self.keys() and overwrite is not True:
            raise KeyError(f"'{name}' already exists. " "To update an existing metadata entry set overwrite=True.")
        if comment is not None:
            self._comments[name] = comment
        if axis is not None:
            axis = self._sanitize_axis_value(axis, value, name)
            self._axes[name] = axis
        elif name in self._axes:
            del self._axes[name]
        # This must be done after updating self._axes otherwise it may error.
        self.__setitem__(name, value)

    def remove(self, name):
        if name in self._comments:
            del self._comments[name]
        if name in self._axes:
            del self._axes[name]
        del self[name]

    def __setitem__(self, key, val):
        axis = self.axes.get(key, None)
        if axis is not None:
            recommendation = "We recommend using the 'add' method to set values."
            if len(axis) == 1:
                if not (hasattr(val, "__len__") and len(val) == self.shape[axis[0]]):
                    raise TypeError(
                        f"{key} must have same length as associated axis, "
                        f"i.e. axis {axis[0]}: {self.shape[axis[0]]}\n"
                        f"{recommendation}"
                    )
            else:
                if not (hasattr(val, "shape") and all(val.shape == self.shape[axis])):
                    raise TypeError(
                        f"{key} must have same shape as associated axes, "
                        f"i.e axes {axis}: {self.shape[axis]}\n"
                        f"{recommendation}"
                    )
        super().__setitem__(key, val)

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key, or
        # by typical python numeric slicing API,
        # i.e. slice the each piece of metadata associated with an axes.

        if isinstance(item, str):
            return super().__getitem__(item)

        elif self.shape is None:
            raise TypeError("Meta object does not have a shape and so cannot be sliced.")

        else:
            new_meta = copy.deepcopy(self)
            if isinstance(item, (numbers.Integral, slice)):
                item = [item]
            item = np.array(list(item) + [slice(None)] * (len(self.shape) - len(item)), dtype=object)

            # Edit data shape and calculate which axis will be dropped.
            dropped_axes = np.zeros(len(self.shape), dtype=bool)
            new_shape = new_meta.shape
            for i, axis_item in enumerate(item):
                if isinstance(axis_item, numbers.Integral):
                    dropped_axes[i] = True
                elif isinstance(axis_item, slice):
                    start = axis_item.start
                    if start is None:
                        start = 0
                    if start < 0:
                        start = self.shape[i] - start
                    stop = axis_item.stop
                    if stop is None:
                        stop = self.shape[i]
                    if stop < 0:
                        stop = self.shape[i] - stop
                    new_shape[i] = stop - start
                else:
                    raise TypeError("Unrecognized slice type. " "Must be an int, slice and tuple of the same.")
            new_meta._data_shape = new_shape[np.invert(dropped_axes)]

            cumul_dropped_axes = np.cumsum(dropped_axes)

            # Slice all metadata associated with axes.
            for key, value in self.items():
                axis = self.axes.get(key, None)
                if axis is not None:
                    new_item = tuple(item[axis])
                    if len(new_item) == 1:
                        new_value = value[new_item[0]]
                    else:
                        new_value = value[new_item]
                    new_axis = np.array([-1 if isinstance(i, numbers.Integral) else a for i, a in zip(new_item, axis)])
                    new_axis -= cumul_dropped_axes[axis]
                    new_axis = new_axis[new_axis >= 0]
                    if len(new_axis) == 0:
                        new_axis = None
                    new_meta.add(key, new_value, self.comments.get(key, None), new_axis, overwrite=True)

            return new_meta

    def rebin(self, bin_shape):
        """
        Adjusts axis-aware metadata to stay consistent with a rebinned `~ndcube.NDCube`.

        This is done by simply removing the axis-awareness of metadata associated with
        rebinned axes. The metadata itself is not changed or removed. This operation
        does not remove axis-awareness from metadata only associated with non-rebinned
        axes, i.e. axes whose corresponding entries in ``bin_shape`` are 1.

        Parameters
        ----------
        bin_shape: `tuple` or `int`
            The new lengths of each axis of the associated data.
        """
        # Sanitize input.
        data_shape = self.shape
        if len(bin_shape) != len(data_shape):
            raise ValueError(f"bin_shape must be same length as data shape: " f"{len(bin_shape)} != {len(self.shape)}")
        if not all([isinstance(dim, numbers.Integral) for dim in bin_shape]):
            raise TypeError("bin_shape must contain only integer types.")
        # Convert bin_shape to array. Do this after checking types of elements to avoid
        # floats being incorrectly rounded down.
        bin_shape = np.asarray(bin_shape, dtype=int)
        if any(data_shape % bin_shape):
            raise ValueError(
                "All elements in bin_shape must be a factor of corresponding element"
                f" of data shape: data_shape mod bin_shape = {self.shape % bin_shape}"
            )
        # Remove axis-awareness from metadata associated with rebinned axes.
        rebinned_axes = set(np.where(bin_shape != 1)[0])
        new_meta = copy.deepcopy(self)
        null_set = set()
        for name, axes in self.axes.items():
            if set(axes).intersection(rebinned_axes) != null_set:
                del new_meta._axes[name]
        # Update data shape.
        new_meta._data_shape = (data_shape / bin_shape).astype(int)
        return new_meta
