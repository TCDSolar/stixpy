import astropy.units as u
import numpy as np

from astropy.time import Time, TimeDelta
from numpy.testing import assert_equal


def times_to_indices(in_times, obs_times, unit=u.ms, decimals=3):
    """
    Convert the input time/s to indices.

    Parameters
    ----------
    in_times : `astropy.units.Quantity`
        A target duration e.g `5*u.s` an array of `Time` if 2d
    obs_times : `astropy.units.Quantity`

    unit : `astropy.units.Unit`

    decimals : int

    Returns
    -------
    Array of indices
    """
    unique_only = False
    shape = None
    relative_times = np.around((obs_times - obs_times[0]).to(unit), decimals=decimals)

    if in_times.isscalar:
        interval = in_times
        total_duration = obs_times[-1] - obs_times[0]
        if interval < total_duration:
            num_intervals = np.ceil((total_duration / interval).decompose()) + 1
            target_times = np.around((np.arange(num_intervals) * interval).to(unit),
                                     decimals=decimals)
            unique_only = True
        else:
            target_times = relative_times[[0,-1]]
    elif isinstance(in_times, Time):
        ndim = in_times.ndim
        target_times = np.around((in_times - obs_times[0]).to(unit), decimals=decimals)
        if ndim == 1:
            unique_only = True
        elif ndim == 2:
            shape = in_times.shape
            if 2 not in shape:
                raise ValueError('If 2d list given must have shape (2, x) or (x, 2).')
            if shape[0] == 2:
                target_times = target_times.T
                shape = shape[::-1]

            target_times = target_times.flatten()
        else:
            raise ValueError('Only 1d or 2d time arrays are supported')

    indices = np.abs(target_times[:, None] - relative_times[None, :]).argmin(axis=-1)
    i1 = _closest_index_loop(target_times, relative_times)
    i2 = _closest_index_broadcast(target_times, relative_times)
    i3 = _closest_index_searchsorted(target_times, relative_times)
    assert_equal(i1, i2)
    assert_equal(i2, i3)
    if shape:
        indices = indices.reshape(shape)
    if unique_only:
        indices = np.unique(indices)
    return indices


def _closest_index_loop(a, b):
    out = np.zeros(a.shape, int)
    for i, val in enumerate(a):
        out[i] = (np.abs(val-b)).argmin()
    return out

def _closest_index_searchsorted(a, b):
    L = b.size
    sidx_b = b.argsort()
    sorted_b = b[sidx_b]
    sorted_idx = np.searchsorted(sorted_b, a)
    sorted_idx[sorted_idx == L] = L - 1
    mask = (sorted_idx > 0) & ((np.abs(a - sorted_b[sorted_idx - 1]) <= np.abs(a - sorted_b[sorted_idx])))
    return sidx_b[sorted_idx - mask]

def _closest_index_broadcast(a, b):
    return np.abs(a[:, None] - b[None, :]).argmin(axis=-1)
