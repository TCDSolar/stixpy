"""
Conversion of selected STIX science data to `sunkit_spex` spectra.
"""

import numpy as np
from ndcube import NDCollection, NDCubeSequence, NDMeta
from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
from sunkit_spex.spectrum.uncertainty import PoissonUncertainty

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from sunpy.coordinates import Helioprojective

from stixpy.calibration.srm import get_srm
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.utils import heliocentric_angle
from stixpy.product.sources._selection import DETECTOR, PIXEL, TIME

__all__ = []


def _return_spec_object(case, data, flare_angle, distance, srm_dict):
    """
    Build one `sunkit_spex` `Spectrum` from a slice of the selected data.

    Counts are summed over every axis except energy, and uncertainties in
    quadrature. The exposure time is the integration time weighted by the
    livetime fraction, averaged over detectors and summed over time. The
    spectral response matrix from `srm_dict` is multiplied by the count energy
    bin widths.

    Parameters
    ----------
    case : str
        Which slice this is: 'spec_1D_detector_collapse',
        'spec_sequence_detector_collapse', 'spec_1D_detector_expand' or
        'spec_sequence_detector_expand'. With the "legacy" error model, for
        every case except 'spec_1D_detector_collapse', negative summed counts are
        set to zero.
    data : _CountData
        The selected data, or one time bin or detector of it, as sliced by
        `_get_sunkit_spex_spectrum`.
    flare_angle : astropy.units.Quantity or None
        Flare angle, stored in the metadata.
    distance : astropy.units.Quantity
        Spacecraft-Sun distance, stored in the metadata.
    srm_dict : dict
        Output of `_masked_srm`, with keys 'srm', 'ph_axis' and 'geo_area'.

    Returns
    -------
    sunkit_spex.spectrum.Spectrum
        Spectrum with the counts, their uncertainty, and the count energy edges
        as the spectral axis. Its metadata holds 'exposure_time', 'geo_area',
        'angle', 'distance', 'srm', 'ph_axis' and 'time_range'.
    """
    energies = data.energies
    counts_axis = np.concatenate([energies["e_low"], [energies["e_high"][-1]]])

    legacy = data.error_model == "legacy"
    counts_final = np.nansum(data.counts, axis=(0, 1, 2))
    if legacy and case != "spec_1D_detector_collapse":
        counts_final[counts_final < 0] = 0
    if legacy:
        counts_uncertainity_final = np.sqrt(np.nansum(data.counts_err**2, axis=(0, 1, 2)))
    else:
        # each part of the uncertainty summed as its correlations need, detectors before time bins
        summed = data
        for axis in (DETECTOR, PIXEL, TIME):
            summed = summed.combine(axis, [[0, summed.counts.shape[axis] - 1]])
        counts_uncertainity_final = summed.total_err()[0, 0, 0]

    durations = data.durations.to(u.s)
    live_time = durations[:, None, None, None] * data.livefrac
    exposure_time = np.sum(np.nanmean(live_time, axis=(1, 2, 3)))

    times = data.times
    if case in ("spec_sequence_detector_collapse", "spec_sequence_detector_expand"):
        times, durations = times[0], durations[0]  # one time bin of a sequence
    time_range_actual = Time([(times - 0.5 * durations).value, (times + 0.5 * durations).value])

    counts_uncertainity_pu = PoissonUncertainty(counts_uncertainity_final)

    counts_spectral_axis = SpectralAxis(counts_axis, bin_specification="edges")

    meta = NDMeta()

    ct_de = np.diff(counts_axis.value)

    srm = srm_dict["srm"] * ct_de[None, :]

    meta.add("exposure_time", exposure_time)
    meta.add("geo_area", srm_dict["geo_area"])
    meta.add("angle", flare_angle)
    meta.add("distance", distance)
    meta.add("srm", srm)
    meta.add("ph_axis", srm_dict["ph_axis"] * u.keV)
    meta.add("time_range", time_range_actual)

    spec_1d = Spectrum(
        data=counts_final, uncertainty=counts_uncertainity_pu, spectral_axis=counts_spectral_axis, meta=meta
    )

    return spec_1d


def _srm_format_flat_or_ranges(indices, case):
    """
    Put detector or pixel indices into the form `_masked_srm` expects.

    Parameters
    ----------
    indices : list, numpy.ndarray or None
        Flat indices, or a list of [start, end] pairs.
    case : str
        Spectrum case (see `_return_spec_object`). For the two
        '..._detector_collapse' cases ranges are expanded into one flat list;
        for the two '..._detector_expand' cases, into one array per range.

    Returns
    -------
    list or None
        An empty list if `indices` is None, `indices` unchanged if they are
        flat, otherwise the expanded ranges. None if the ranges are given as
        tuples or `case` is not recognised.
    """

    if indices is None:
        return []

    elif isinstance(indices[0], (int, np.integer)):
        return indices

    elif isinstance(indices[0], (list, np.ndarray)):
        indices = [np.arange(start, end + 1) for start, end in indices]

        if case in ("spec_1D_detector_collapse", "spec_sequence_detector_collapse"):
            return [idx for ls in indices for idx in ls]

        elif case in ("spec_1D_detector_expand", "spec_sequence_detector_expand"):
            return indices


def _srm_det_pix_indices_format(detector_indices, pixel_indices, case):
    """
    Format detector and pixel indices for `_masked_srm`.

    Applies `_srm_format_flat_or_ranges` to each.

    Parameters
    ----------
    detector_indices : list, numpy.ndarray or None
        Flat detector indices, or a list of [start, end] pairs.
    pixel_indices : list, numpy.ndarray or None
        Flat pixel indices, or a list of [start, end] pairs.
    case : str
        Spectrum case (see `_return_spec_object`).

    Returns
    -------
    detector_indices : list
        Formatted detector indices.
    pixel_indices : list
        Formatted pixel indices.
    """

    det_formatted = _srm_format_flat_or_ranges(detector_indices, case)
    pix_formatted = _srm_format_flat_or_ranges(pixel_indices, case)

    return det_formatted, pix_formatted


def _masked_srm(energies, flare_location, detector_indices_input, pixel_indices_input, rcr, srm_e_min):
    """
    Build the SRM with `~stixpy.calibration.srm.get_srm`, in the form `_return_spec_object` expects.

    Returns
    -------
    dict
        'srm' in ct / (keV ph) as an array, 'ph_axis' the photon edges in keV as an array, and
        'geo_area' the geometric area.
    """
    count_edges = np.hstack([energies["e_low"], energies["e_high"][-1]])
    srm = get_srm(
        count_edges,
        detector_indices_input,
        pixel_indices_input,
        rcr,
        flare_location=flare_location,
        ph_energy_min=srm_e_min,
    )
    return {
        "srm": srm.srm.to_value(u.ct / (u.keV * u.ph)),
        "ph_axis": srm.photon_edges.to_value(u.keV),
        "geo_area": srm.geometric_area,
    }


def _get_sunkit_spex_spectrum(
    product,
    detector_indices,
    pixel_indices,
    data,
    flare_location,
    flare_angle,
    detector_sum=True,
    srm_e_min=3.5 * u.keV,
):
    """
    Turn the selected data into `sunkit_spex` spectra.

    What is returned depends on the number of time bins and on `detector_sum`:

    =========  ============  ===============================================
    time bins  detector_sum  returns
    =========  ============  ===============================================
    1          True          one `Spectrum`
    several    True          `NDCubeSequence` of spectra, one per time bin
    1          False         `NDCollection` of spectra, keyed by detector
    several    False         `NDCollection` of `NDCubeSequence`, keyed by
                             detector
    =========  ============  ===============================================

    A spectral response matrix is computed with `_masked_srm` for each detector
    (or for the summed detectors), once per RCR state present. Spectrogram
    products are always treated as ``detector_sum=True``, with the detectors
    and pixels taken from the product's masks.

    Parameters
    ----------
    product : ScienceData
        Product the data came from, used for the flare angle and the spacecraft
        distance (``meta["DSUN_OBS"]``).
    detector_indices : numpy.ndarray
        Detector selection, flat or [start, end] pairs. With
        ``detector_sum=False`` each flat index or range gives one spectrum.
    pixel_indices : numpy.ndarray
        Pixel selection, flat or [start, end] pairs.
    data : _CountData
        The selected data, not normalised, from ``get_data``.
    flare_location : astropy.coordinates.SkyCoord or None
        Flare position in any frame that can be transformed to
        `~stixpy.coordinates.frames.STIXImaging` (see `_flare_location_stix`).
        If None, the SRM is built on-axis and no flare angle is computed.
    flare_angle : astropy.units.Quantity or None
        Flare angle stored in the metadata. If None and `flare_location` is
        given, it is computed with `_flare_angle`.
    detector_sum : bool, optional
        Sum detectors into one spectrum (True, default), or make one spectrum
        per detector or detector range (False).
    srm_e_min : astropy.units.Quantity or None, optional
        Lower photon energy limit of the SRM.

    Returns
    -------
    sunkit_spex.spectrum.Spectrum, ndcube.NDCubeSequence or ndcube.NDCollection
        See the table above.
    """
    if flare_location is not None:
        flare_location_stx = _flare_location_stix(product, flare_location)
        if flare_angle is None:
            flare_angle = _flare_angle(flare_location_stx)
    else:
        flare_location_stx = None

    distance = (product.meta["DSUN_OBS"] * u.m).to(u.AU)
    rcr = data.rcr
    n_times, n_detectors = data.counts.shape[:2]

    if product.data["counts"].ndim < 4:
        detector_indices = np.where(product.detector_masks.masks == 1)[1]
        pixel_indices = np.where(product.pixel_masks.masks == 1)[1]
        detector_sum = True

    def srm_by_rcr(detectors, pixels):
        return {
            state: _masked_srm(
                data.energies,
                flare_location=flare_location_stx,
                detector_indices_input=detectors,
                pixel_indices_input=pixels,
                rcr=state,
                srm_e_min=srm_e_min,
            )
            for state in np.unique(rcr)
        }

    def sequence(case, data, srms):
        spectra = [
            _return_spec_object(case, data[i], flare_angle, distance, srms[int(data.rcr[i])])
            for i in range(data.counts.shape[0])
        ]
        return NDCubeSequence(spectra, meta={"detector": "det1", "instrument": "STIX"}, common_axis=0)

    if detector_sum:
        if n_times == 1:
            case = "spec_1D_detector_collapse"
            detectors, pixels = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)
            srm_dict = _masked_srm(
                data.energies,
                flare_location=flare_location_stx,
                detector_indices_input=detectors,
                pixel_indices_input=pixels,
                rcr=np.unique(rcr)[0],
                srm_e_min=srm_e_min,
            )
            return _return_spec_object(case, data, flare_angle, distance, srm_dict)

        case = "spec_sequence_detector_collapse"
        detectors, pixels = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)
        return sequence(case, data, srm_by_rcr(detectors, pixels))

    if n_times == 1:
        case = "spec_1D_detector_expand"
        detectors, pixels = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)
        spectra = []
        for i in range(n_detectors):
            srm_dict = _masked_srm(
                data.energies,
                flare_location=flare_location_stx,
                detector_indices_input=detectors[i],
                pixel_indices_input=pixels,
                rcr=np.unique(rcr),
                srm_e_min=srm_e_min,
            )
            spectra.append(
                (f"{detector_indices[i]}", _return_spec_object(case, data[:, i], flare_angle, distance, srm_dict))
            )
        return NDCollection(spectra, aligned_axes="all")

    case = "spec_sequence_detector_expand"
    detectors, pixels = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)
    sequences = [
        (f"{detector_indices[i]}", sequence(case, data[:, i], srm_by_rcr(detectors[i], pixels)))
        for i in range(n_detectors)
    ]
    return NDCollection(sequences, aligned_axes="all")


def _flare_location_stix(product, flare_location):
    """
    Return the flare position in the STIX imaging frame.

    Parameters
    ----------
    product : ScienceData
        Product whose time range sets the STIX frame when `flare_location` is
        in another frame and has no ``obstime``.
    flare_location : astropy.coordinates.SkyCoord
        Flare position. A coordinate in the
        `~stixpy.coordinates.frames.STIXImaging` frame is used as it is, and
        must have an ``obstime``. Any other frame, e.g.
        `~sunpy.coordinates.Helioprojective` from another observer, is
        transformed to the STIX frame at its own ``obstime``, or for the
        product's time range if it has none.

    Returns
    -------
    astropy.coordinates.SkyCoord
        The flare position in the STIX imaging frame.
    """
    if isinstance(flare_location.frame, STIXImaging):
        if flare_location.obstime is None:
            raise ValueError("flare_location in the STIXImaging frame must have an obstime.")
        return flare_location

    if flare_location.obstime is not None:
        stix_frame = STIXImaging(obstime=flare_location.obstime)
    else:
        stix_frame = STIXImaging(obstime=product.time_range.start, obstime_end=product.time_range.end)
    # transform the frame, not the SkyCoord, so the input's observer is not carried over to the STIX frame
    return SkyCoord(flare_location.frame.transform_to(stix_frame))


def _flare_angle(flare_location_stx):
    """
    Compute the heliocentric angle of the flare as seen from Solar Orbiter.

    The STIX coordinate is transformed to helioprojective coordinates with
    Solar Orbiter as the observer. The spacecraft position and the observation
    time both come from the coordinate's STIX frame: the middle of
    ``obstime``-``obstime_end``, or ``obstime`` alone.

    Parameters
    ----------
    flare_location_stx : astropy.coordinates.SkyCoord
        Flare position in the `~stixpy.coordinates.frames.STIXImaging` frame.

    Returns
    -------
    astropy.units.Quantity
        The angle returned by `~stixpy.coordinates.utils.heliocentric_angle`,
        NaN deg (with a warning) if the flare is off the limb.
    """
    frame = flare_location_stx.frame
    obstime = frame.obstime if frame.obstime_end is None else frame.obstime + (frame.obstime_end - frame.obstime) / 2
    # with no observer given, the STIX to helioprojective transform uses Solar Orbiter's position at obstime; the
    # frame is transformed, not the SkyCoord, so no other observer is carried over from it
    flare_hpc = SkyCoord(flare_location_stx.frame.transform_to(Helioprojective(obstime=obstime)))
    return heliocentric_angle(flare_hpc)
