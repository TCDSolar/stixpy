"""
Spectra of selected STIX science data, for spectral fitting with `sunkit_spex`.

Needs sunkit-spex, which can be installed with ``pip install 'stixpy[spectroscopy]'``.
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from sunpy.coordinates import Helioprojective

try:
    from ndcube import NDCollection, NDCubeSequence, NDMeta
    from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum
    from sunkit_spex.spectrum.uncertainty import PoissonUncertainty
except ImportError as e:
    raise ImportError(
        "stixpy.spectroscopy needs sunkit-spex, which can be installed with pip install 'stixpy[spectroscopy]'."
    ) from e

from stixpy.calibration.srm import get_srm
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.utils import get_heliocentric_angle

__all__ = ["to_spectrum"]


def to_spectrum(data, *, flare_location=None, systematic_error=False, srm_e_min=3.5 * u.keV):
    """
    Turn selected data into `sunkit_spex` spectra, with their spectral response matrices.

    Each time bin and detector position of `data` gives one spectrum, so sum the times and detectors to fit with
    `~stixpy.product.sources.science.ScienceData.get_data` (``time_indices``, ``sum_all_times`` and
    ``sum_detectors``), and subtract a background with `~stixpy.product.sources.science.subtract_background`, first.
    The spectra hold counts, whatever the ``vtype`` of `data`, with the live time as their exposure time.

    =========  ===============  ===============================================
    time bins  detector         returns
               positions
    =========  ===============  ===============================================
    1          1                one `Spectrum`
    several    1                `NDCubeSequence` of spectra, one per time bin
    1          several          `NDCollection` of spectra, keyed by detector
    several    several          `NDCollection` of `NDCubeSequence`, keyed by
                                detector
    =========  ===============  ===============================================

    Detector positions without a detector that is on (with no ``detector_indices`` given, those of detectors that are
    off) are left out.

    Parameters
    ----------
    data : SelectedData
        Data from `~stixpy.product.sources.science.ScienceData.get_data`, with the livetime correction.
    flare_location : `~astropy.coordinates.SkyCoord`, optional
        Flare position, used for the grid transmission in the spectral response and for the heliocentric angle of
        the flare as seen from Solar Orbiter, stored in the metadata as 'angle' (None without a position). A
        `~stixpy.coordinates.frames.STIXImaging` coordinate (which must have an ``obstime``) is used as it is; any
        other frame, e.g. `~sunpy.coordinates.Helioprojective`, is transformed to the STIX frame at its own
        ``obstime``, or for the product's time range if it has none. The ``stx`` or ``hpc`` entry of
        `~stixpy.imaging.flare_location.estimate_flare_location` can be used. Default None, on-axis.
    systematic_error : bool, optional
        Add an energy-dependent systematic uncertainty of 7% of the counts below 7 keV, 5% from 7 to 10 keV and 3%
        from 10 keV to each spectrum, in quadrature. Default False.
    srm_e_min : astropy.units.Quantity, float, bool or None, optional
        Lower photon energy limit of the spectral response matrix. A float is taken as keV, True means 3.5 keV and
        False means None. Default 3.5 keV.

    Returns
    -------
    sunkit_spex.spectrum.Spectrum, ndcube.NDCubeSequence or ndcube.NDCollection
        See the table above. The metadata of each spectrum holds 'exposure_time', 'geo_area', 'angle', 'distance',
        'srm' (ct / ph, for the count bin widths), 'ph_axis' and 'time_range'.

    Raises
    ------
    TypeError
        If `flare_location` is not a `~astropy.coordinates.SkyCoord`.
    ValueError
        If `data` is not livetime corrected, or the response matrix can not be built for its detectors (see
        `~stixpy.calibration.srm.get_srm`).
    """
    selection = data.selection
    if flare_location is not None and not isinstance(flare_location, SkyCoord):
        raise TypeError(f"flare_location must be a SkyCoord, not {type(flare_location).__name__}.")
    if data.livefrac is None:
        raise ValueError(
            "to_spectrum needs data with the livetime correction, as the exposure time of each spectrum is its live "
            "time. Use get_data with livetime_correction=True."
        )
    if isinstance(srm_e_min, bool):
        srm_e_min = 3.5 * u.keV if srm_e_min else None
    elif srm_e_min is not None:
        srm_e_min = u.Quantity(srm_e_min, u.keV)

    flare_location_stx, flare_angle = None, None
    if flare_location is not None:
        flare_location_stx = _flare_location_stix(selection.time_range, flare_location)
        flare_angle = _flare_angle(flare_location_stx)
    distance = None if selection.distance is None else selection.distance.to(u.AU)

    live_time = data.t_norm * data.livefrac * np.ones(data.counts.shape[:2] + (1, 1))  # (time, detector, 1, 1)
    norm = {"c": 1, "cr": live_time, "dcr": live_time * data.e_norm}[data.vtype]
    counts = (data.counts * norm).to_value(u.ct)[:, :, 0, :]
    counts_err = (data.counts_err * norm).to_value(u.ct)[:, :, 0, :]
    if systematic_error:
        e_mean = ((data.energies["e_low"] + data.energies["e_high"]) / 2).to_value(u.keV)
        fraction = np.select([e_mean < 7, e_mean < 10], [0.07, 0.05], 0.03)
        counts_err = np.sqrt(counts_err**2 + (fraction * counts) ** 2)

    count_edges = np.append(data.energies["e_low"].to_value(u.keV), data.energies["e_high"][-1].to_value(u.keV))
    pixels = np.flatnonzero(selection.pixels)
    srms = {}

    def spectrum(t, d):
        detectors = np.flatnonzero(selection.detectors[d])
        rcr = int(data.rcr[t])
        if (d, rcr) not in srms:
            srms[d, rcr] = get_srm(
                count_edges * u.keV, detectors, pixels, rcr, flare_location=flare_location_stx, ph_energy_min=srm_e_min
            )
        srm = srms[d, rcr]
        duration = data.t_norm[t, 0, 0, 0]
        meta = NDMeta()
        meta.add("exposure_time", live_time[t, d, 0, 0])
        meta.add("geo_area", srm.geometric_area)
        meta.add("angle", flare_angle)
        meta.add("distance", distance)
        meta.add("srm", srm.srm.to_value(u.ct / (u.keV * u.ph)) * np.diff(count_edges)[None, :])
        meta.add("ph_axis", srm.photon_edges.to(u.keV))
        meta.add("time_range", Time([data.times[t] - duration / 2, data.times[t] + duration / 2]))
        return Spectrum(
            data=counts[t, d],
            uncertainty=PoissonUncertainty(counts_err[t, d]),
            spectral_axis=SpectralAxis(count_edges * u.keV, bin_specification="edges"),
            meta=meta,
        )

    def detector(d):
        if counts.shape[0] == 1:
            return spectrum(0, d)
        spectra = [spectrum(t, d) for t in range(counts.shape[0])]
        return NDCubeSequence(spectra, meta={"detector": "det1", "instrument": "STIX"}, common_axis=0)

    positions = [d for d in range(counts.shape[1]) if selection.detectors[d].any()]
    if len(positions) == 1:
        return detector(positions[0])
    return NDCollection(
        [(",".join(map(str, np.flatnonzero(selection.detectors[d]))), detector(d)) for d in positions],
        aligned_axes="all",
    )


def _flare_location_stix(time_range, flare_location):
    """
    Return the flare position in the STIX imaging frame.

    Parameters
    ----------
    time_range : sunpy.time.TimeRange
        Time range of the data, which sets the STIX frame when `flare_location` is in another frame and has no
        ``obstime``.
    flare_location : astropy.coordinates.SkyCoord
        Flare position. A coordinate in the `~stixpy.coordinates.frames.STIXImaging` frame is used as it is, and
        must have an ``obstime``. Any other frame, e.g. `~sunpy.coordinates.Helioprojective`, is transformed to the
        STIX frame at its own ``obstime``, or for `time_range` if it has none.

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
        stix_frame = STIXImaging(obstime=time_range.start, obstime_end=time_range.end)
    # transform the frame, not the SkyCoord, so the input's observer is not carried over to the STIX frame
    return SkyCoord(flare_location.frame.transform_to(stix_frame))


def _flare_angle(flare_location_stx):
    """
    Compute the heliocentric angle of the flare as seen from Solar Orbiter.

    The STIX coordinate is transformed to helioprojective coordinates with Solar Orbiter as the observer. The
    spacecraft position and the observation time both come from the coordinate's STIX frame: the middle of
    ``obstime``-``obstime_end``, or ``obstime`` alone.

    Parameters
    ----------
    flare_location_stx : astropy.coordinates.SkyCoord
        Flare position in the `~stixpy.coordinates.frames.STIXImaging` frame.

    Returns
    -------
    astropy.units.Quantity
        The angle returned by `~stixpy.coordinates.utils.get_heliocentric_angle`, NaN deg (with a warning) if the
        flare is off the limb.
    """
    frame = flare_location_stx.frame
    obstime = frame.obstime if frame.obstime_end is None else frame.obstime + (frame.obstime_end - frame.obstime) / 2
    # with no observer given, the STIX to helioprojective transform uses Solar Orbiter's position at obstime; the
    # frame is transformed, not the SkyCoord, so no other observer is carried over from it
    flare_hpc = SkyCoord(flare_location_stx.frame.transform_to(Helioprojective(obstime=obstime)))
    return get_heliocentric_angle(flare_hpc)
