"""
Analysis tools.
"""

import astropy.nddata
import numpy as np
from ndcube import ExtraCoords
from ndcube.utils.cube import propagate_rebin_uncertainties
from ndcube.utils.wcs import convert_between_array_and_pixel_axes, physical_type_to_pixel_axes
from sunpy.util import warn_user

ARRAY_MASK_MAP = {}
ARRAY_MASK_MAP[np.ndarray] = np.ma.masked_array
_NUMPY_COPY_IF_NEEDED = False if np.__version__.startswith("1.") else None
try:
    import dask.array
    ARRAY_MASK_MAP[dask.array.core.Array] = dask.array.ma.masked_array
except ImportError:
    pass


# TODO: Delete once this function is broken out in ndcube to a util and import from there
def _convert_to_masked_array(data, mask, operation_ignores_mask):
    m = None if (mask is None or mask is False or operation_ignores_mask) else mask
    if m is not None:
        for array_type, masked_type in ARRAY_MASK_MAP.items():
            if isinstance(data, array_type):
                break
        else:
            masked_type = np.ma.masked_array
            warn_user("data and mask arrays of different or unrecognized types. Casting them into a numpy masked array.")
        return masked_type(data, m)


def rebin_irregular(cube, axes_idx_edges, operation=np.mean, operation_ignores_mask=False,
                    handle_mask=np.all, propagate_uncertainties=False, new_unit=None, **kwargs):
    """
    Downsample array by combining irregularly sized, but contiguous, blocks of elements into bins.

    Parameters
    ----------
    data:

    axes_idx_edges: iterable of iterable of `int`
        The array indices for each axis defined in the edges of the contiguous blocks
        to rebin the data to. Each block includes the lower index and up to but not
        including the upper index.

    operation: func
        The operation defining how the elements in each block should be combined.
        Must be able to be applied over a specific axis via an axis= kwarg.
        Default=`numpy.mean`

    Returns
    -------
    :

    Example
    -------
    >>> import numpy as np
    >>> from stixpy.product.tools import rebin_irregular
    >>> axes_idx_edges = [0, 2, 3], [0, 2, 4], [0, 3, 5]
    >>> data = np.ones((3, 4, 5))
    >>> rebin_irregular(data, axes_idx_edges, operation=np.sum) # doctest: +SKIP
    array([[[12.,  8.],
        [12.,  8.]],
       [[ 6.,  4.],
        [ 6.,  4.]]])
    """
    # Sanitize inputs
    ndim = len(cube.data.shape)
    if len(axes_idx_edges) != ndim:
        raise ValueError(f"length of axes_idx_edges must be {ndim}")

    # Extract coordinate grids for rebinning and other coord info.
    orig_wcs = cube.combined_wcs
    new_coord_values = list(cube.axis_world_coords_values(wcs=orig_wcs))
    new_coord_physical_types = orig_wcs.world_axis_physical_types[::-1]
    new_coord_pix_idxs = [physical_type_to_pixel_axes(phys_type, orig_wcs)
                          for phys_type in new_coord_physical_types]
    new_coord_names = (
        name if name else phys_type
        for name, phys_type in zip(orig_wcs.world_axis_names[::-1], new_coord_physical_types))
    wc_arr_idxs = [convert_between_array_and_pixel_axes(idx, ndim) for idx in new_coord_pix_idxs]

    # Combine data and mask and determine whether mask also needs rebinning.
    new_data = _convert_to_masked_array(cube.data, cube.mask, operation_ignores_mask)
    rebin_mask = False
    if handle_mask is None or operation_ignores_mask:
        new_mask = None
    else:
        new_mask = cube.mask
        if not isinstance(cube.mask, (type(None), bool)):
            rebin_mask = True

    # Determine whether to propagate uncertainties
    if propagate_uncertainties:
        # Temporarily set propagate_uncertainties to False and then determine whether they
        # can/should be set back to True.
        propagate_uncertainties = False
        if cube.uncertainty is None:
            warn_user("Uncertainties cannot be propagated as there are no uncertainties, "
                          "i.e., the `uncertainty` keyword was never set on creation of this NDCube.")
        elif isinstance(cube.uncertainty, astropy.nddata.UnknownUncertainty):
            warn_user("The uncertainty on this NDCube has no known way to propagate forward and so will be dropped. "
                          "To create an uncertainty that can propagate, please see "
                          "https://docs.astropy.org/en/stable/uncertainty/index.html")
        elif (not operation_ignores_mask
              and (cube.mask is True or (cube.mask is not None
                                         and not isinstance(cube.mask, bool)
                                         and cube.mask.all()))):
            warn_user("Uncertainties cannot be propagated as all values are masked and "
                      "operation_ignores_mask is False.")
        else:
            propagate_uncertainties = True  # Uncertainties can be propagated. Reset to True.
            new_uncertainty = cube.uncertainty
    else:
        new_uncertainty = None

    # Iterate through dimensions and perform rebinning operation.
    for i, edges in enumerate(axes_idx_edges):
        # If no edge indices provided for dimension, skip to next one.
        if edges is None:
            continue
        # Iterate through pixel blocks and collapse via rebinning operation.
        tmp_data = []
        tmp_mask = []
        tmp_uncertainties = []
        tmp_coords = [[]] * len(new_coord_values)
        item = [slice(None)] * ndim
        coord_item = np.array([slice(None)] * ndim)
        for j in range(len(edges)-1):
            # Rebin data
            item[i] = slice(edges[j], edges[j+1])
            tuple_item = tuple(item)
            data_chunk = new_data[tuple_item]
            tmp_data.append(operation(data_chunk, axis=i))
            # Rebin mask
            if rebin_mask is True:
                mask_chunk = new_mask[tuple_item]
                tmp_mask.append(handle_mask(mask_chunk, axis=i))
            # Rebin uncertainties
            if propagate_uncertainties:
                # Reorder axis i to be 0th axis as this is format required by propagate_rebin_uncertainties
                uncertainty_chunk = type(new_uncertainty)(np.moveaxis(new_uncertainty.array[tuple_item], i, 0), unit=new_uncertainty.unit)
                data_unc = np.moveaxis(data_chunk, i, 0)
                mask_unc = np.moveaxis(mask_chunk, i, 0) if rebin_mask else new_mask
                tmp_uncertainty = propagate_rebin_uncertainties(uncertainty_chunk, data_unc, mask_unc, operation,
                                                                operation_ignores_mask=operation_ignores_mask,
                                                                handle_mask=handle_mask, new_unit=new_unit, **kwargs)
                tmp_uncertainties.append(tmp_uncertainty.array)
            # Rebin coordinates
            coord_item[i] = np.array([edges[j], edges[j+1]-1])
            for k, (coord, coord_arr_idx) in enumerate(zip(new_coord_values, wc_arr_idxs)):
                if i in coord_arr_idx:
                    idx_i = np.where(coord_arr_idx == i)[0][0]
                    tmp_coords[k].append(
                        np.mean(coord[tuple(coord_item[coord_arr_idx])], axis=idx_i))
        # Restore rebinned axes.
        if rebin_mask:
            new_mask = np.stack(tmp_mask, axis=i)
        new_data = np.ma.masked_array(np.stack(tmp_data, axis=i).data, mask=new_mask)
        if propagate_uncertainties:
            new_uncertainty = type(new_uncertainty)(np.stack(tmp_uncertainties, axis=i),unit=new_uncertainty.unit)
        for k, coord_arr_idx in enumerate(wc_arr_idxs):
            if i in coord_arr_idx:
                idx_i = np.where(coord_arr_idx == i)[0][0]
                new_coord_values[k] = np.stack(tmp_coords[k], axis=idx_i)

    # Build new WCS.
    ec = ExtraCoords()
    for name, axis, coord, physical_type in zip(new_coord_names, new_coord_pix_idxs,
                                                new_coord_values, new_coord_physical_types):
        if len(axis) != 1:
            raise NotImplementedError("Rebinning multi-dimensional coordinates not supported.")
        ec.add(name, axis, coord, physical_types=physical_type)
    new_wcs = ec.wcs

    # Rebin metadata if possible.
    if hasattr(cube.meta, "__ndcube_can_rebin__") and cube.meta.__ndcube_can_rebin__:
        new_shape = [len(ax)-1 for ax in axes_idx_edges]
        rebinned_axes = set(tuple(ax) != tuple(range(n + 1))
                            for ax, n in zip(axes_idx_edges, cube.shape))
        new_meta = cube.meta.rebin(rebinned_axes, new_shape)
    else:
        new_meta = cube.meta

    # Build new cube.
    new_cube = type(cube)(new_data.data, new_wcs, uncertainty=new_uncertainty, mask=new_data.mask, meta=new_meta)

    return new_cube
