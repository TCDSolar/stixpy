"""
Analysis tools.
"""

import operator
from functools import reduce

import astropy.nddata
import numpy as np
from astropy.time import Time
from ndcube import NDMeta, utils
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
from sunpy.util import warn_user

__all__ = ["rebin_by_edges"]

ARRAY_MASK_MAP = {}
ARRAY_MASK_MAP[np.ndarray] = np.ma.masked_array
try:
    import dask.array

    ARRAY_MASK_MAP[dask.array.core.Array] = dask.array.ma.masked_array
except ImportError:
    pass


def rebin_by_edges(
    cube,
    axes_idx_edges,
    operation=np.mean,
    operation_ignores_mask=False,
    handle_mask=np.all,
    propagate_uncertainties=False,
    new_unit=None,
    **kwargs,
):
    """
    Downsample a cube by combining irregularly sized, but contiguous, blocks of elements into bins.

    Metadata is rebinned via ``meta.rebin_by_edges(axes_idx_edges)`` if the metadata provides it,
    otherwise axis-awareness is dropped for metadata associated with rebinned axes (as `ndcube.NDMeta.rebin`).

    Parameters
    ----------
    cube : `ndcube.NDCube`
        The cube to rebin.
    axes_idx_edges : iterable of iterable of `int` or `None`
        For each array axis, the indices defining the edges of the contiguous blocks to rebin the data to.
        Each block includes the lower index and up to but not including the upper index. Elements
        outside the first and last edge are discarded. `None` leaves the axis unchanged.
    operation : func
        The operation defining how the elements in each block should be combined.
        Must be able to be applied over a specific axis via an axis= kwarg. Default=`numpy.mean`.
    operation_ignores_mask : `bool`
        If `True` masked values are included in ``operation``.
    handle_mask : func
        Combines the mask values in each block. `None` drops the mask.
    propagate_uncertainties : `bool`
        If `True` propagate the uncertainties using `ndcube.utils.cube.propagate_rebin_uncertainties`.
    new_unit : `astropy.units.Unit`, optional
        Unit of the data after ``operation``, passed to the uncertainty propagation.

    Returns
    -------
    `ndcube.NDCube`
        New cube of the same type as ``cube``. Coordinates are the mean of the coordinates in each block.

    Example
    -------
    >>> import numpy as np
    >>> from ndcube import NDCube
    >>> from ndcube.extra_coords import QuantityTableCoordinate
    >>> import astropy.units as u
    >>> from stixpy.product.tools import rebin_by_edges
    >>> wcs = (QuantityTableCoordinate(np.arange(5) * u.keV) & QuantityTableCoordinate(np.arange(3) * u.s)).wcs
    >>> cube = NDCube(np.ones((3, 5)), wcs=wcs)
    >>> rebin_by_edges(cube, [[0, 2, 3], [0, 3, 5]], operation=np.sum).data
    array([[6., 4.],
           [3., 2.]])
    """
    ndim = cube.data.ndim
    if len(axes_idx_edges) != ndim:
        raise ValueError(f"length of axes_idx_edges must be {ndim}")

    # Extract coordinate grids for rebinning and other coord info.
    # All values below are in world axis order, the order returned by axis_world_coords.
    orig_wcs = cube.combined_wcs
    new_coord_values = list(cube.axis_world_coords(wcs=orig_wcs))
    orig_wcs = orig_wcs.low_level_wcs
    new_coord_physical_types = orig_wcs.world_axis_physical_types
    new_coord_names = [
        name if name else phys_type for name, phys_type in zip(orig_wcs.world_axis_names, new_coord_physical_types)
    ]
    # Array axes of each world coordinate, from the correlation matrix (columns are pixel axes,
    # the reverse of array axes) to avoid ambiguous physical type lookups.
    wc_arr_idxs = [np.where(row[::-1])[0] for row in orig_wcs.axis_correlation_matrix]

    # Combine data and mask and determine whether mask also needs rebinning.
    new_data = _create_masked_array_for_rebinning(cube.data, cube.mask, operation_ignores_mask)
    rebin_mask = False
    if handle_mask is None or operation_ignores_mask:
        new_mask = None
    else:
        new_mask = cube.mask
        if not isinstance(cube.mask, (type(None), bool)):
            rebin_mask = True

    # Determine whether to propagate uncertainties
    propagate_uncertainties = propagate_uncertainties and _uncertainties_can_propagate(
        cube.uncertainty, cube.mask, operation_ignores_mask
    )
    new_uncertainty = cube.uncertainty if propagate_uncertainties else None

    # Iterate through dimensions and perform rebinning operation.
    for i, edges in enumerate(axes_idx_edges):
        if edges is None:
            continue
        tmp_data = []
        tmp_mask = []
        tmp_uncertainties = []
        tmp_coords = [[] for _ in new_coord_values]
        item = [slice(None)] * ndim
        for j in range(len(edges) - 1):
            item[i] = slice(edges[j], edges[j + 1])
            tuple_item = tuple(item)
            data_chunk = new_data[tuple_item]
            tmp_data.append(operation(data_chunk, axis=i))
            if rebin_mask:
                mask_chunk = new_mask[tuple_item]
                tmp_mask.append(handle_mask(mask_chunk, axis=i))
            if propagate_uncertainties:
                # Reorder axis i to be 0th axis as this is format required by propagate_rebin_uncertainties
                uncertainty_chunk = type(new_uncertainty)(
                    np.moveaxis(new_uncertainty.array[tuple_item], i, 0), unit=new_uncertainty.unit
                )
                # Copy as propagate_rebin_uncertainties modifies data in place when there is a mask
                data_unc = np.moveaxis(data_chunk, i, 0).copy()
                mask_unc = np.moveaxis(mask_chunk, i, 0) if rebin_mask else new_mask
                tmp_uncertainty = utils.cube.propagate_rebin_uncertainties(
                    uncertainty_chunk,
                    data_unc,
                    mask_unc,
                    operation,
                    operation_ignores_mask=operation_ignores_mask,
                    handle_mask=handle_mask,
                    new_unit=new_unit,
                    **kwargs,
                )
                tmp_uncertainties.append(tmp_uncertainty.array)
            # Rebin coordinates
            for k, (coord, coord_arr_idx) in enumerate(zip(new_coord_values, wc_arr_idxs)):
                if i in coord_arr_idx:
                    idx_i = np.where(coord_arr_idx == i)[0][0]
                    coord_item = [slice(None)] * len(coord_arr_idx)
                    coord_item[idx_i] = slice(edges[j], edges[j + 1])
                    tmp_coords[k].append(coord[tuple(coord_item)].mean(axis=idx_i))
        # Restore rebinned axes.
        if rebin_mask:
            new_mask = np.stack(tmp_mask, axis=i)
        new_data = np.ma.masked_array(np.ma.getdata(np.stack(tmp_data, axis=i)), mask=new_mask)
        if propagate_uncertainties:
            new_uncertainty = type(new_uncertainty)(np.stack(tmp_uncertainties, axis=i), unit=new_uncertainty.unit)
        for k, coord_arr_idx in enumerate(wc_arr_idxs):
            if i in coord_arr_idx:
                idx_i = np.where(coord_arr_idx == i)[0][0]
                new_coord_values[k] = np.stack(tmp_coords[k], axis=idx_i)

    # Build new WCS from the coordinates of the original WCS, combined_wcs lists these before the extra coords.
    coords = list(zip(new_coord_names, wc_arr_idxs, new_coord_values, new_coord_physical_types))
    if any(len(axis) != 1 for _, axis, _, _ in coords):
        raise NotImplementedError("Rebinning multi-dimensional coordinates not supported.")
    n_wcs = cube.wcs.low_level_wcs.world_n_dim
    wcs_coords = sorted(coords[:n_wcs], key=lambda c: c[1][0])
    if len({int(axis[0]) for _, axis, _, _ in wcs_coords}) != ndim:
        raise NotImplementedError("Rebinning requires exactly one WCS coordinate per array axis.")
    tables = [
        (TimeTableCoordinate if isinstance(coord, Time) else QuantityTableCoordinate)(
            coord, names=name, physical_types=physical_type
        )
        for name, _, coord, physical_type in wcs_coords
    ]
    # Compound models are in WCS (reversed array) order
    new_wcs = reduce(operator.and_, tables[::-1]).wcs

    # Rebin metadata.
    if hasattr(cube.meta, "rebin_by_edges"):
        new_meta = cube.meta.rebin_by_edges(axes_idx_edges)
    elif getattr(cube.meta, "__ndcube_can_rebin__", False):
        new_meta = _drop_rebinned_axis_awareness(cube.meta, axes_idx_edges, new_data.shape)
    else:
        new_meta = cube.meta

    new_cube = type(cube)(
        np.ma.getdata(new_data), wcs=new_wcs, uncertainty=new_uncertainty, mask=new_mask, meta=new_meta, unit=cube.unit
    )
    for name, axis, coord, physical_type in coords[n_wcs:]:
        new_cube.extra_coords.add(name, int(axis[0]), coord, physical_types=physical_type)
    return new_cube


def _create_masked_array_for_rebinning(data, mask, operation_ignores_mask):
    m = None if (mask is None or mask is False or operation_ignores_mask) else mask
    if m is None:
        return data
    for array_type, masked_type in ARRAY_MASK_MAP.items():
        if isinstance(data, array_type):
            break
    else:
        masked_type = np.ma.masked_array
        warn_user("data and mask arrays of different or unrecognized types. Casting them into a numpy masked array.")
    return masked_type(data, m)


def _uncertainties_can_propagate(uncertainty, mask, operation_ignores_mask):
    if uncertainty is None:
        warn_user(
            "Uncertainties cannot be propagated as there are no uncertainties, "
            "i.e., the `uncertainty` keyword was never set on creation of this NDCube."
        )
        return False
    elif isinstance(uncertainty, astropy.nddata.UnknownUncertainty):
        warn_user(
            "The uncertainty on this NDCube has no known way to propagate forward and so will be dropped. "
            "To create an uncertainty that can propagate, please see "
            "https://docs.astropy.org/en/stable/uncertainty/index.html"
        )
        return False
    elif not operation_ignores_mask and (
        mask is True or (mask is not None and not isinstance(mask, bool) and mask.all())
    ):
        warn_user("Uncertainties cannot be propagated as all values are masked and operation_ignores_mask is False.")
        return False
    return True


def _drop_rebinned_axis_awareness(meta, axes_idx_edges, new_shape):
    """
    Generic `ndcube.NDMeta` fallback: remove axis-awareness from metadata associated with rebinned axes.
    """
    rebinned_axes = {i for i, edges in enumerate(axes_idx_edges) if edges is not None}
    new_meta = NDMeta(data_shape=new_shape)
    for name, value in meta.items():
        axes = meta.axes.get(name)
        if axes is not None and rebinned_axes.intersection(axes):
            axes = None
        new_meta.add(name, value, key_comment=meta.key_comments.get(name), axes=axes)
    return new_meta
