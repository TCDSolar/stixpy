"""
Helpers for the rate control regime (RCR) state of STIX science data.
"""

import warnings

import numpy as np

__all__ = []


def _rcr_signal(counts, n_detectors):
    """
    The counts STIX-GSW looks for RCR jumps in:
    ``total(spec, 1) * 24. / n_elements(detectors_used)``.

    The counts are summed over every axis except time (detectors or detector
    groups, pixels, energies) and scaled to 24 detectors, so that the fixed 1e4
    jump threshold in `_rcr_shift` means the same whatever the number of
    detectors used.

    Parameters
    ----------
    counts : astropy.units.Quantity or numpy.ndarray
        Counts with time on the first axis.
    n_detectors : int
        Number of detectors used (selected and switched on).

    Returns
    -------
    numpy.ndarray
        One value per time bin.
    """
    counts = np.asarray(getattr(counts, "value", counts), dtype=np.float64)
    return np.nansum(counts, axis=tuple(range(1, counts.ndim))) * 24.0 / max(n_detectors, 1)


def _rcr_shift(rcr, signal):
    """
    Move the reported RCR state changes onto the jumps in the counts, as in
    STIX-GSW.

    A line-by-line port of the RCR correction in STIX-GSW
    ``stx_convert_spectrogram2ospex`` (ECMD, 2022-06-27):

    1. ``find_changes``: the first bin of each RCR state, starting with bin 0.
    2. Only if ``max(rcr) > 0``: a jump is at bin ``i`` when
       ``|signal[i] - signal[i + 1]| > 1e4`` (IDL ``shift`` wraps around, so the
       last bin is also compared with the first). Bin 0 is put in front.
    3. A jump is kept if the next jump (wrapping around) is more than 2 bins
       away, so from each group of close jumps the last one is kept.
    4. Each reported change is moved to the closest kept jump (``value_closest``).

    IDL behaviour is kept where no jump or no kept jump is found: ``where``
    returns -1, a scalar -1 subscript takes the last element, and a -1 in a
    subscript array is clipped to 0.

    The per-bin states are then rebuilt as STIX-GSW ``stx_read_sp_data`` does:
    each bin takes the state in force at its start (``value_locate``, clipped
    at 0). ``stx_read_sp_data`` also marks bins in which a state change falls
    as uncertain (-99); that is not done here.

    Parameters
    ----------
    rcr : array_like
        Reported RCR state of each time bin.
    signal : array_like
        GSW's ``total(spec, 1) * 24. / n_elements(detectors_used)``: counts per
        time bin summed over energies and scaled to 24 detectors.

    Returns
    -------
    numpy.ndarray
        The RCR state of each time bin.
    """
    rcr_arr = np.asarray(rcr)
    n_times = rcr_arr.size

    # find_changes, rcr, index, state
    index = np.concatenate(([0], np.flatnonzero(rcr_arr[1:] != rcr_arr[:-1]) + 1))
    state = rcr_arr[index]

    if np.max(rcr_arr) > 0:
        signal = np.asarray(signal, dtype=np.float64)

        # jumps = where(abs(x - shift(x, -1)) gt 1e4)   (-1 if none)
        jumps = np.flatnonzero(np.abs(signal - np.roll(signal, -1)) > 1e4)
        if jumps.size == 0:
            jumps = np.array([-1])
        # jumps = [0, jumps]
        jumps = np.concatenate(([0], jumps))
        # idx_jumps = where(abs(jumps - shift(jumps, -1)) gt 2)   (-1 if none)
        idx_jumps = np.flatnonzero(np.abs(jumps - np.roll(jumps, -1)) > 2)
        # jumps_use = [jumps[idx_jumps]]   (scalar -1 subscript = last element)
        jumps_use = jumps[idx_jumps] if idx_jumps.size else jumps[-1:]
        # index = jumps_use[value_closest(jumps_use, index)]
        index = jumps_use[np.abs(jumps_use[None, :] - index[:, None]).argmin(axis=1)]
        # ut_rcr[index]: a negative subscript in an array is clipped to 0
        index = np.clip(index, 0, n_times - 1)

    # stx_read_sp_data: state[value_locate(time, bin_start) > 0]
    which = np.searchsorted(index, np.arange(n_times), side="right") - 1
    return state[np.clip(which, 0, None)]


def _rcr_warning(time_indices, rcr):
    """
    Warn if the RCR state varies across flat time indices.

    Parameters
    ----------
    time_indices : list of int
        Time indices to check.
    rcr : array_like
        RCR state of each data bin.

    Warns
    -----
    UserWarning
        Once for each index whose RCR state differs from that of the first
        index.
    """

    first_rcr = rcr[time_indices[0]]
    for i in time_indices[1:]:
        if rcr[i] != first_rcr:
            warnings.warn(
                f"RCR state change detected "
                f"index {time_indices[0]} has RCR={first_rcr!r}, "
                f"index {i} has RCR={rcr[i]!r}."
                f"Use with caution!",
                stacklevel=4,
            )
    return None


def _rcr_error(indices, rcr):
    """
    Raise if the RCR state changes across the given time indices.

    Parameters
    ----------
    indices : list of int
        Time indices to check. An empty list passes.
    rcr : array_like
        RCR state of each data bin.

    Raises
    ------
    ValueError
        If any index has a different RCR state from the first.
    """
    if not indices:
        return None
    first_rcr = rcr[indices[0]]
    for i in indices[1:]:
        if rcr[i] != first_rcr:
            raise ValueError(
                f"RCR state change detected. index {indices[0]} has RCR={first_rcr!r}, index {i} has RCR={rcr[i]!r}."
            )


def _handle_nested_pairs(time_indices: list[list[int]], rcr: list) -> list[list[int]]:
    """
    Check [start, end] time index pairs for RCR changes.

    Parameters
    ----------
    time_indices : list of list of int
        [start, end] time index pairs.
    rcr : array_like
        RCR state of each data bin.

    Returns
    -------
    list of list of int
        `time_indices`, unchanged.

    Raises
    ------
    ValueError
        If the RCR state changes inside any pair.

    Warns
    -----
    UserWarning
        If the pairs are in different RCR states from each other.
    """
    # Check within each pair
    for pair in time_indices:
        indices_in_pair = list(range(pair[0], pair[1] + 1))
        _rcr_error(indices_in_pair, rcr)

    # Warn if RCR state differs across pairs
    pair_representatives = [rcr[pair[0]] for pair in time_indices]
    if len(set(pair_representatives)) > 1:
        warnings.warn(
            f"RCR state differs across nested pairs: "
            f"{[f'pair {n}={r!r}' for n, r in enumerate(pair_representatives)]}.",
            stacklevel=4,
        )

    return time_indices
