"""
Conversion of selected STIX science data to `sunkit_spex` spectra.
"""

import numpy as np
from ndcube import NDCollection, NDCubeSequence, NDMeta
from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
from sunkit_spex.spectrum.uncertainty import PoissonUncertainty

import astropy.units as u
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst

from stixpy.calibration.srm import get_srm
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.coordinates.utils import heliocentric_angle
from stixpy.product.sources._corrections import _indices_expand_ranges

__all__ = []


def _return_spec_object(case, sci_data, flare_angle, distance, srm_dict, bkg):
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
        'spec_sequence_detector_expand'. Sets which axes are summed. For every
        case except 'spec_1D_detector_collapse', negative summed counts are set
        to zero.
    sci_data : tuple
        The 10-element tuple from `_data_select`, or a single time bin or
        detector of it, as sliced by `_get_sunkit_spex_spectrum`.
    flare_angle : astropy.units.Quantity or None
        Flare angle, stored in the metadata.
    distance : astropy.units.Quantity
        Spacecraft-Sun distance, stored in the metadata.
    srm_dict : dict
        Output of `_masked_srm`, with keys 'srm', 'ph_axis' and 'geo_area'.
    bkg : bool
        Not used.

    Returns
    -------
    sunkit_spex.spectrum.Spectrum
        Spectrum with the counts, their uncertainty, and the count energy edges
        as the spectral axis. Its metadata holds 'exposure_time', 'geo_area',
        'angle', 'distance', 'srm', 'ph_axis' and 'time_range'.
    """

    counts, counts_uncertainity, t_norm, _, livefrac, _, elut_cor_fac, times_full, energies, _ = sci_data

    t_norm = t_norm.to(u.s)

    counts_axis = np.concatenate([energies["e_low"], [energies["e_high"][-1]]])

    if case == "spec_1D_detector_collapse":
        counts_final = np.nansum(counts, axis=(0, 1, 2))
        counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0, 1, 2)))

        t_norm_or = t_norm
        t_norm = t_norm[:, None, None, None] * livefrac
        t_norm = np.nanmean(t_norm, axis=(1, 2, 3))

    elif case == "spec_sequence_detector_collapse" or case == "spec_1D_detector_expand":
        counts_final = np.nansum(counts, axis=(0, 1))
        counts_final[counts_final < 0] = 0
        counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0, 1)))

        t_norm_or = np.nanmean(t_norm, axis=(0))
        t_norm = t_norm * livefrac
        t_norm = np.nanmean(t_norm, axis=(0, 1, 2))

    elif case == "spec_sequence_detector_expand":
        counts_final = np.nansum(counts, axis=(0))
        counts_final[counts_final < 0] = 0
        counts_uncertainity_final = np.sqrt(np.nansum(counts_uncertainity**2, axis=(0)))

        t_norm_or = np.nanmean(t_norm, axis=(0))
        t_norm = t_norm * livefrac
        t_norm = np.nanmean(t_norm, axis=(0))

    counts_uncertainity_pu = PoissonUncertainty(counts_uncertainity_final)

    counts_spectral_axis = SpectralAxis(counts_axis, bin_specification="edges")

    meta = NDMeta()

    time_range_actual = Time([(times_full - 0.5 * t_norm_or).value, (times_full + 0.5 * t_norm_or).value])

    ct_de = np.diff(counts_axis.value)

    srm = srm_dict["srm"] * ct_de[None, :]

    meta.add("exposure_time", np.sum(t_norm))
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
        indices = _indices_expand_ranges(indices)

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
    sci_data,
    flare_location,
    flare_angle,
    systematic,
    detector_sum=True,
    rcr=None,
    bkg=False,
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
    sci_data : tuple
        The 10-element tuple from `_data_select`.
    flare_location : dict or None
        Flare position with keys 'stx' (STIX-frame Tx/Ty) and 'hpc'. If None,
        no flare angle is computed and the SRM gets no flare position.
    flare_angle : astropy.units.Quantity or None
        Flare angle. If None and `flare_location` is given, it is computed with
        `_flare_angle`.
    systematic : bool
        Not used; the systematic uncertainty is added in `_data_select`.
    detector_sum : bool, optional
        Sum detectors into one spectrum (True, default), or make one spectrum
        per detector or detector range (False).
    rcr : array_like, optional
        Not used; the RCR states are taken from `sci_data`.
    bkg : bool, optional
        Passed on to `_return_spec_object`, which does not use it.
    srm_e_min : astropy.units.Quantity or None, optional
        Lower photon energy limit of the SRM.

    Returns
    -------
    sunkit_spex.spectrum.Spectrum, ndcube.NDCubeSequence or ndcube.NDCollection
        See the table above.
    """

    counts, counts_uncertainity, t_norm, e_norm, livefrac, _, elut_cor_fac, times_full, energies, rcr = sci_data

    if flare_location is not None:
        flare_location_stx = u.Quantity([flare_location["stx"].Tx, flare_location["stx"].Ty])
        if flare_angle is None:
            flare_angle = _flare_angle(product, flare_location)
    else:
        flare_location_stx = None
        flare_angle = None

    distance = (product.meta["DSUN_OBS"] * u.m).to(u.AU)
    rcr_unique = np.unique(rcr)

    shape = np.shape(product.data["counts"])

    if len(shape) < 4:
        detector_indices = np.where(product.detector_masks.masks == 1)[1]
        pixel_indices = np.where(product.pixel_masks.masks == 1)[1]
        detector_sum = True

    if detector_sum:
        if np.shape(counts)[0] == 1:
            case = "spec_1D_detector_collapse"

            detector_indices_srm, pixel_indices_srm = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)

            srm_dict = _masked_srm(
                energies,
                flare_location=flare_location_stx,
                detector_indices_input=detector_indices_srm,
                pixel_indices_input=pixel_indices_srm,
                rcr=rcr_unique[0],
                srm_e_min=srm_e_min,
            )

            return _return_spec_object(case, sci_data, flare_angle, distance, srm_dict, bkg)

        else:
            case = "spec_sequence_detector_collapse"

            detector_indices_srm, pixel_indices_srm = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)

            rcr_unique = np.unique(rcr)

            srm_dict_by_rcr = {
                rcr_val: _masked_srm(
                    energies,
                    flare_location=flare_location_stx,
                    detector_indices_input=detector_indices_srm,
                    pixel_indices_input=pixel_indices_srm,
                    rcr=rcr_val,
                    srm_e_min=srm_e_min,
                )
                for rcr_val in rcr_unique
            }

            spec_list_working = []

            for i in range(np.shape(counts)[0]):
                (
                    counts,
                    counts_uncertainity,
                    t_norm,
                    e_norm,
                    livefrac,
                    _,
                    elut_cor_fac,
                    times_full,
                    energies,
                    rcr,
                ) = sci_data

                sci_data_indexed = (
                    counts[i, ...],
                    counts_uncertainity[i, ...],
                    t_norm[i, ...],
                    e_norm,
                    livefrac[i, ...],
                    _,
                    elut_cor_fac,
                    times_full[i, ...],
                    energies,
                    rcr,
                )

                spec_1d = _return_spec_object(
                    case, sci_data_indexed, flare_angle, distance, srm_dict_by_rcr[int(rcr[i])], bkg
                )

                spec_list_working.append(spec_1d)

            spec_sequence = NDCubeSequence(
                spec_list_working, meta={"detector": "det1", "instrument": "STIX"}, common_axis=0
            )

            return spec_sequence

    else:
        if np.shape(counts)[0] == 1:
            case = "spec_1D_detector_expand"

            spec_list_working = []

            detector_indices_srm, pixel_indices_srm = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)

            for i in range(np.shape(counts)[1]):
                srm_dict = _masked_srm(
                    energies,
                    flare_location=flare_location_stx,
                    detector_indices_input=detector_indices_srm[i],
                    pixel_indices_input=pixel_indices_srm,
                    rcr=rcr_unique,
                    srm_e_min=srm_e_min,
                )

                (
                    counts,
                    counts_uncertainity,
                    t_norm,
                    e_norm,
                    livefrac,
                    _,
                    elut_cor_fac,
                    times_full,
                    energies,
                    rcr,
                ) = sci_data

                sci_data_indexed = (
                    counts[:, i, ...],
                    counts_uncertainity[:, i, ...],
                    t_norm,
                    e_norm,
                    livefrac[:, i, ...],
                    _,
                    elut_cor_fac,
                    times_full,
                    energies,
                    rcr,
                )

                spec_1d = _return_spec_object(case, sci_data_indexed, flare_angle, distance, srm_dict, bkg)

                spec_list_working.append((f"{detector_indices[i]}", spec_1d))

            spec_collection = NDCollection(spec_list_working, aligned_axes="all")

            return spec_collection

        else:
            spec_list_collection_working = []

            case = "spec_sequence_detector_expand"

            detector_indices_srm, pixel_indices_srm = _srm_det_pix_indices_format(detector_indices, pixel_indices, case)

            for i in range(np.shape(counts)[1]):
                rcr_unique = np.unique(rcr)

                srm_dict_by_rcr = {
                    rcr_val: _masked_srm(
                        energies,
                        flare_location=flare_location_stx,
                        detector_indices_input=detector_indices_srm[i],
                        pixel_indices_input=pixel_indices_srm,
                        rcr=rcr_val,
                        srm_e_min=srm_e_min,
                    )
                    for rcr_val in rcr_unique
                }

                (
                    counts,
                    counts_uncertainity,
                    t_norm,
                    e_norm,
                    livefrac,
                    _,
                    elut_cor_fac,
                    times_full,
                    energies,
                    rcr,
                ) = sci_data

                counts = counts[:, i, ...]
                counts_uncertainity = counts_uncertainity[:, i, ...]
                livefrac = livefrac[:, i, ...]

                spec_list_sequence_working = []

                for j in range(np.shape(counts)[0]):
                    sci_data_indexed = (
                        counts[j, ...],
                        counts_uncertainity[j, ...],
                        t_norm[j, ...],
                        e_norm,
                        livefrac[j, ...],
                        _,
                        elut_cor_fac,
                        times_full[j, ...],
                        energies,
                        rcr,
                    )

                    spec_1d = _return_spec_object(
                        case, sci_data_indexed, flare_angle, distance, srm_dict_by_rcr[int(rcr[j])], bkg
                    )

                    spec_list_sequence_working.append(spec_1d)

                spec_sequence = NDCubeSequence(
                    spec_list_sequence_working,
                    meta={"detector": "det1", "instrument": "STIX"},  # sequence-level
                    common_axis=0,
                )

                spec_list_collection_working.append((f"{detector_indices[i]}", spec_sequence))

            spec_collection = NDCollection(spec_list_collection_working, aligned_axes="all")

            return spec_collection


def _flare_angle(product, flare_location):
    """
    Compute the flare angle for the product.

    Solar Orbiter's position is taken at the start of the product's time range
    and passed, with the flare position, to `heliocentric_angle`.

    Parameters
    ----------
    product : ScienceData
        Product whose time range sets the spacecraft position.
    flare_location : dict
        Flare position, with key 'hpc' holding its helioprojective coordinate.

    Returns
    -------
    astropy.units.Quantity
        The angle returned by `heliocentric_angle`.
    """

    _, solo_xyz, _ = get_hpc_info(product.time_range.start, product.time_range.start)

    solo = HeliographicStonyhurst(*solo_xyz, obstime=product.time_range.center, representation_type="cartesian")

    flare_angle = heliocentric_angle(flare_location["hpc"], observer=solo)

    return flare_angle
