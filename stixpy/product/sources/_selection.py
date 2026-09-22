"""
Helpers that turn the selections passed to `~stixpy.product.sources.science.ScienceData.get_data`
into indices and check them against the product.
"""

import warnings

import numpy as np

import astropy.units as u
from astropy.time import Time

__all__ = []


def _indices_check(product, detector_indices, pixel_indices, energy_indices):
    """
    Check the requested detector, pixel and energy indices against what the
    product contains, and fill in defaults.

    Detectors and pixels are checked against `product.detector_masks` and
    `product.pixel_masks`. They may be a flat list of indices or a list of
    [start, end] pairs; any that are not in the product raise a warning and the
    selection is kept as given. None selects every detector or pixel in the
    product.

    Detectors can also be given as a label, in any case. Labels are not checked
    against the mask:

    - "top24": the 24 imaging detectors of sub-collimators 3-10 (indices 0-7,
      13-15 and 19-31), as in STIX-GSW.
    - "bkg": the background detector (index 9) with its small-aperture pixels
      [2, 5]. If `pixel_indices` is None it is set to [2, 5] with a warning;
      any other pixels raise a ValueError. Use ``detector_indices=[9]`` to
      choose other pixels.

    For spectrogram products (counts with fewer than 4 dimensions) detector and
    pixel selections do not apply, and any that were given are dropped with a
    warning.

    Energy indices number the rows of `product.energies`, i.e. the last axis of
    the counts. That table only holds the bins in the product's energy mask, so
    an index outside it raises a ValueError naming the valid indices and energy
    range.

    Parameters
    ----------
    product : ScienceData
        Product whose masks and energy table define what is available.
    detector_indices : list, numpy.ndarray, str or None
        Flat detector indices, [start, end] pairs, "top24", "bkg", or None for
        all detectors in the product.
    pixel_indices : list, numpy.ndarray or None
        Flat pixel indices, [start, end] pairs, or None for all pixels in the
        product ([2, 5] with ``detector_indices="bkg"``).
    energy_indices : list, numpy.ndarray or None
        Flat energy bin indices or [start, end] pairs, numbering the rows of
        `product.energies`. None skips the energy check.

    Returns
    -------
    detector_indices : list, numpy.ndarray or None
        The detector selection, with a label resolved to indices. None for
        spectrogram products.
    pixel_indices : list, numpy.ndarray or None
        The pixel selection. None for spectrogram products.
    energy_indices : list, numpy.ndarray or None
        `energy_indices`, unchanged.

    Raises
    ------
    ValueError
        If a detector label is not recognised, if ``detector_indices="bkg"`` is
        given with pixels other than [2, 5], or if a requested energy bin is
        outside the product's energy table.

    Warns
    -----
    UserWarning
        If a requested detector or pixel is not in the product, if detector or
        pixel indices are given for a spectrogram product, or if
        ``detector_indices="bkg"`` sets the pixels to [2, 5].
    """

    # --- Detector indices ---

    if detector_indices is not None:
        if len(product.data["counts"].shape) < 4:
            warnings.warn(
                f"As a spectrogram file is being used, the user selected detector indices \
                            {detector_indices} will not be used, defaulting to the indices used in the creation \
                            of the spectrgram file.",
                stacklevel=3,
            )

            detector_indices = None

        else:
            detector_indices_working = detector_indices

            if isinstance(detector_indices_working, str):
                # named detector sets, as in STIX-GSW stx_label2det_ind
                detector_labels = {
                    "top24": [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        13,
                        14,
                        15,
                        19,
                        20,
                        21,
                        22,
                        23,
                        24,
                        25,
                        26,
                        27,
                        28,
                        29,
                        30,
                        31,
                    ],
                    "bkg": [9],
                }
                label = detector_indices_working.lower()
                if label not in detector_labels:
                    raise ValueError(
                        f"Unknown detector label {detector_indices_working!r}, use one of {list(detector_labels)}."
                    )
                detector_indices = np.array(detector_labels[label])

                if label == "bkg":
                    if pixel_indices is None:
                        pixel_indices = [2, 5]
                        warnings.warn(
                            'detector_indices="bkg" with no pixel_indices given: using the BKG detector\'s '
                            "small-aperture pixels [2, 5].",
                            stacklevel=3,
                        )
                    elif np.ndim(pixel_indices) != 1 or sorted(np.asarray(pixel_indices).tolist()) != [2, 5]:
                        raise ValueError(
                            f'detector_indices="bkg" uses the BKG detector\'s small-aperture pixels [2, 5], '
                            f"but pixel_indices={pixel_indices} was given. Either leave pixel_indices unset "
                            f'(None) or set pixel_indices=[2, 5] if using the "bkg" preset.'
                            f"To use background detectors with other pixel_indices use detector_indices=[9]"
                            f"with your choice of pixel_indices."
                        )

                detector_indices_full = np.unique(np.where(product.detector_masks.masks == 1)[1])
                missing = np.setdiff1d(detector_indices, detector_indices_full)
                if missing.size > 0:
                    usable = np.intersect1d(detector_indices, detector_indices_full)
                    raise ValueError(
                        f'detector_indices="{label}" includes detectors {missing.tolist()} that are switched off in '
                        f"this file (detectors on: {detector_indices_full.tolist()}). Give the detectors explicitly, "
                        f"e.g. detector_indices={usable.tolist()}."
                    )

            else:
                detector_indices_full = np.unique(np.where(product.detector_masks.masks == 1)[1])

                if np.ndim(detector_indices_working) == 2:
                    # [[start, end], ...] range format
                    for start, end in detector_indices_working:
                        requested = np.arange(start, end + 1)
                        missing = np.setdiff1d(requested, detector_indices_full)
                        if missing.size > 0:
                            raise ValueError(
                                f"Detector range [{start}, {end}] includes detectors {missing.tolist()} that are "
                                f"switched off in this file (detectors on: {detector_indices_full.tolist()}). Split "
                                "the range around them or give the detectors as a flat list."
                            )
                else:
                    missing = np.setdiff1d(detector_indices_working, detector_indices_full)
                    if missing.size > 0:
                        raise ValueError(
                            f"Detectors {missing.tolist()} are switched off in this file (detectors on: "
                            f"{detector_indices_full.tolist()}). Remove them, or leave detector_indices unset to "
                            "use every detector that is on."
                        )

    else:
        if len(product.data["counts"].shape) < 4:
            detector_indices = None
        else:
            detector_indices = np.where(product.detector_masks.masks == 1)[1]

    # --- Pixel indices ---
    if pixel_indices is not None:
        pixel_indices_full = np.unique(np.where(product.pixel_masks.masks == 1)[1])

        if len(product.data["counts"].shape) < 4:
            warnings.warn(
                f"As a spectrogram file is being used, the user selected detector indices \
                            {pixel_indices} will not be used, defaulting to the indices used in the creation \
                            of the spectrgram file.",
                stacklevel=3,
            )
            pixel_indices = None

        else:
            if np.ndim(pixel_indices) == 2:
                for start, end in pixel_indices:
                    requested = np.arange(start, end + 1)
                    missing = np.setdiff1d(requested, pixel_indices_full)
                    if missing.size > 0:
                        raise ValueError(
                            f"Pixel range [{start}, {end}] includes pixels {missing.tolist()} that are switched off "
                            f"in this file (pixels on: {pixel_indices_full.tolist()}). Split the range around them "
                            "or give the pixels as a flat list."
                        )
            else:
                missing = np.setdiff1d(pixel_indices, pixel_indices_full)
                if missing.size > 0:
                    raise ValueError(
                        f"Pixels {missing.tolist()} are switched off in this file (pixels on: "
                        f"{pixel_indices_full.tolist()}). Remove them, or leave pixel_indices unset to use every "
                        "pixel that is on."
                    )

    else:
        if len(product.data["counts"].shape) < 4:
            pixel_indices = None
        else:
            pixel_indices = np.where(product.pixel_masks.masks == 1)[1]

    # --- Energy indices ---
    if energy_indices is not None:
        energy_indices_full = np.arange(len(product.energies))
        e_min = np.nanmin(product.energies["e_low"].value)
        e_max = np.nanmax(product.energies["e_high"].value)

        energy_range = (
            f"The product has energy indices {energy_indices_full[0]}-{energy_indices_full[-1]} "
            f"({e_min} - {e_max} keV)."
        )

        if np.ndim(energy_indices) == 2:
            for start, end in energy_indices:
                requested = np.arange(start, end + 1)
                missing = np.setdiff1d(requested, energy_indices_full)
                if missing.size > 0:
                    raise ValueError(
                        f"Energy indices {missing.tolist()} in range [{start}, {end}] are not included in the product's energy mask. {energy_range}"
                    )
        else:
            missing = np.setdiff1d(energy_indices, energy_indices_full)
            if missing.size > 0:
                raise ValueError(
                    f"The following energy indices are not included in the product's energy mask: {missing.tolist()}. {energy_range}"
                )

    return detector_indices, pixel_indices, energy_indices


def _time_indices_format(time_indices, times, dt):
    """
    Turn a time selection into integer indices, checking it against the file.

    Accepted forms:

    - flat integer indices, e.g. ``[0, 2, 5]``, returned unchanged;
    - flat strings or `~astropy.time.Time`, taken as consecutive range edges:
      ``["2023-01-01T10:00", "2023-01-01T10:05", "2023-01-01T10:10"]`` gives
      two ranges;
    - [start, end] pairs of strings or `~astropy.time.Time`;
    - [start, end] pairs of integer indices.

    Indices and times are checked against the file first. Times are then
    resolved to the data bins that lie wholly inside each range. Numpy integers
    and arrays are accepted as well as Python ints and lists. The selection is
    checked against the RCR states later, in `_data_select`, once the RCR has
    been aligned to the counts.

    Parameters
    ----------
    time_indices : list or numpy.ndarray
        The time selection, in one of the forms above.
    times : astropy.time.Time
        Centre time of each data bin.
    dt : astropy.units.Quantity
        Duration of each data bin.

    Returns
    -------
    list or numpy.ndarray
        Flat integer indices, or a list of [start, end] integer pairs.

    Raises
    ------
    ValueError
        If an index or time is outside the file, or if the format is not
        recognised.
    IndexError
        If a time range contains no complete data bin.
    """

    first = time_indices[0]

    # file limits: start of the first bin to the end of the last bin
    file_start = times[0] - 0.5 * dt[0]
    file_end = times[-1] + 0.5 * dt[-1]

    if isinstance(first, (int, np.integer)):
        _check_index_limits(time_indices, len(times))
        return time_indices

    if isinstance(first, (str, Time)):
        bins = [[time_indices[i], time_indices[i + 1]] for i in range(len(time_indices) - 1)]
        _check_time_limits(bins, file_start, file_end)
        return _handle_datetime_strings(bins, times, dt)

    if isinstance(first, (list, tuple, np.ndarray)):
        if isinstance(first[0], (str, Time)):
            _check_time_limits(time_indices, file_start, file_end)
            return _handle_datetime_strings(time_indices, times, dt)
        if len(first) == 2 and all(isinstance(v, (int, np.integer)) for v in first):
            _check_index_limits(time_indices, len(times))
            return time_indices
        raise ValueError(f"Nested lists must be [start, end] integer or time pairs, got: {first}")

    raise ValueError(f"Cannot determine format from first element: {first!r}")


def _check_time_limits(bins, file_start, file_end):
    """
    Check that time ranges lie within the file.

    A tolerance of 1 ms absorbs floating-point error in the file limits, so a
    range that starts or ends exactly on the file edge is accepted.

    Parameters
    ----------
    bins : list
        [start, end] pairs of strings, `~astropy.time.Time`, or anything else
        `~astropy.time.Time` accepts.
    file_start : astropy.time.Time
        Start of the first data bin.
    file_end : astropy.time.Time
        End of the last data bin.

    Raises
    ------
    ValueError
        If any start is before `file_start` or any end is after `file_end`.
    """
    tol = 1 * u.ms  # absorbs floating-point error in file_start / file_end
    for start, end in bins:
        if Time(start) < file_start - tol or Time(end) > file_end + tol:
            raise ValueError(
                f"Requested times [{Time(start).isot} - {Time(end).isot}] fall outside the "
                f"file time range [{file_start.isot} - {file_end.isot}]."
            )

@staticmethod
def _check_index_limits(indices, n_times):
    """
    Check that time indices lie within the file.

    Parameters
    ----------
    indices : array_like
        Flat integer indices or [start, end] pairs.
    n_times : int
        Number of time bins in the file.

    Raises
    ------
    ValueError
        If any index is negative or not less than `n_times`.
    """
    indices = np.asarray(indices)
    bad = indices[(indices < 0) | (indices >= n_times)]
    if bad.size > 0:
        raise ValueError(
            f"Time indices {np.unique(bad).tolist()} are outside the file, which has time indices 0-{n_times - 1}."
        )


def _handle_datetime_strings(time_bin: list[list[str | Time]], times: list[str | Time], dt) -> list[list[int]]:
    """
    Resolve time ranges to the data bins that lie wholly inside them.

    Parameters
    ----------
    time_bin : list
        [start, end] pairs of strings, `~astropy.time.Time`, or anything else
        `~astropy.time.Time` accepts.
    times : astropy.time.Time
        Centre time of each data bin.
    dt : astropy.units.Quantity
        Duration of each data bin.

    Returns
    -------
    list of list of int
        For each range, the [first, last] index of the data bins that start at
        or after its start and end at or before its end.

    Raises
    ------
    ValueError
        If a range does not have exactly two elements.
    IndexError
        If a range contains no complete data bin.
    """

    data_bin_start = times - (0.5 * dt)
    data_bin_end = times + (0.5 * dt)

    results = []
    for n, pair in enumerate(time_bin):
        if len(pair) != 2:
            raise ValueError(
                f"Each time bin must have exactly 2 elements [start, end], got {len(pair)} at index {n}."
            )

        bin_start = Time(pair[0])
        bin_end = Time(pair[1])

        matched = [
            i for i, t in enumerate(times) if (bin_start <= data_bin_start[i]) and (data_bin_end[i] <= bin_end)
        ]

        results.append([matched[0], matched[-1]])

    return results


def _find_bin_index(start, end, e_low, e_high):
    """
    Find the energy bins whose centres lie in an energy range.

    A bin belongs to the range its centre, ``(e_low + e_high) / 2``, falls in,
    inclusive at both ends. The open top bin has a NaN upper edge and so is
    never selected.

    Parameters
    ----------
    start : float
        Lower edge of the range, in keV.
    end : float
        Upper edge of the range, in keV.
    e_low : numpy.ndarray
        Lower edge of each bin, in keV.
    e_high : numpy.ndarray
        Upper edge of each bin, in keV.

    Returns
    -------
    list of int
        ``[first, last]`` index of the selected bins.

    Raises
    ------
    ValueError
        If no bin centre lies in the range.
    """

    e_centre = (e_low + e_high) / 2

    matches = np.where((e_centre >= start) & (e_centre <= end))[0]

    if matches.size == 0:
        raise ValueError(
            f"Energy range [{start} - {end}] keV does not contain the centre of any product energy bin."
        )

    return [np.min(matches), np.max(matches)]


def _energy_indices_from_flat_edges(values, e_low, e_high):
    """
    Turn a flat list of energy edges into [start, end] bin index pairs.

    N edges give N - 1 consecutive ranges: ``[5, 10, 25]`` gives 5-10 keV and
    10-25 keV. Each range is resolved with `_find_bin_index`.

    Parameters
    ----------
    values : numpy.ndarray
        Energy edges, in keV.
    e_low : numpy.ndarray
        Lower edge of each bin, in keV.
    e_high : numpy.ndarray
        Upper edge of each bin, in keV.

    Returns
    -------
    list of list of int
        One [start, end] bin index pair per range.
    """
    pairs = []
    for i in range(len(values) - 1):
        idx = _find_bin_index(values[i], values[i + 1], e_low, e_high)
        pairs.append(idx)
    return pairs


def _energy_indices_from_range_pairs(values, e_low, e_high):
    """
    Turn [start, end] energy ranges into [start, end] bin index pairs.

    Each range is resolved with `_find_bin_index`.

    Parameters
    ----------
    values : numpy.ndarray
        [start, end] energy ranges, in keV, e.g. ``[[5, 10], [15, 25]]``.
    e_low : numpy.ndarray
        Lower edge of each bin, in keV.
    e_high : numpy.ndarray
        Upper edge of each bin, in keV.

    Returns
    -------
    list of list of int
        One [start, end] bin index pair per range.
    """
    pairs = []
    for start_val, end_val in values:
        idx = _find_bin_index(start_val, end_val, e_low, e_high)
        pairs.append(idx)
    return pairs


def _energy_indices_format(energy_indices, energies):
    """
    Turn an energy selection given in energy units into bin index pairs.

    Anything that is not an `~astropy.units.Quantity`, including None, is
    returned unchanged and treated as bin indices already. A Quantity is
    converted to keV and can be:

    - 1D, a flat list of edges: ``[4, 10, 28] * u.keV`` gives 4-10 and
      10-28 keV;
    - 2D, a list of [start, end] ranges: ``[[4, 10], [15, 28]] * u.keV``.

    Each range takes the bins whose centres lie inside it (see
    `_find_bin_index`).

    Parameters
    ----------
    energy_indices : astropy.units.Quantity, list, numpy.ndarray or None
        The energy selection.
    energies : astropy.table.QTable
        Energy table with 'e_low' and 'e_high' columns.

    Returns
    -------
    list of list of int
        [start, end] bin index pairs, numbering the rows of `energies`, or
        `energy_indices` unchanged if it is not a Quantity.

    Raises
    ------
    ValueError
        If a requested energy is outside the file's energy range, if a range
        contains no bin centre, or if the Quantity is neither 1D nor 2D.
    """

    if not isinstance(energy_indices, u.Quantity):
        return energy_indices

    energy_indices = energy_indices.to(u.keV)

    e_low = energies["e_low"].value
    e_high = energies["e_high"].value

    # The requested energies must lie within the range covered by the file. nanmin/nanmax
    # skip the NaN upper edge of the open top bin present in a full energy table.
    e_file_min = np.nanmin(e_low)
    e_file_max = np.nanmax(e_high)

    e_requested_min = np.nanmin(energy_indices.value)
    e_requested_max = np.nanmax(energy_indices.value)

    if e_requested_min < e_file_min or e_requested_max > e_file_max:
        raise ValueError(
            f"Requested energies [{e_requested_min} - {e_requested_max}] keV fall outside the "
            f"energy range of the file [{e_file_min} - {e_file_max}] keV."
        )

    if energy_indices.ndim == 1:
        return _energy_indices_from_flat_edges(energy_indices.value, e_low, e_high)

    elif energy_indices.ndim == 2:
        return _energy_indices_from_range_pairs(energy_indices.value, e_low, e_high)

    else:
        raise ValueError(
            "energy_indices given as a Quantity must be either 1D "
            "(flat list of bin edges) or 2D (list of [start, end] pairs)."
        )
