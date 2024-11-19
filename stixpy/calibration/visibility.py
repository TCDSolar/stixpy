from types import SimpleNamespace
from typing import Union
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.time import TimeRange

from stixpy.calibration.energy import get_elut
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.io.readers import read_subc_params

__all__ = [
    "get_subcollimator_info",
    "create_meta_pixels",
    "create_visibility",
    "get_uv_points_data",
    "calibrate_visibility",
]

from xrayvision.visibility import Visibilities, VisMeta

from stixpy.product.sources import CompressedPixelData, RawPixelData, SummedCompressedPixelData


def get_subcollimator_info():
    r"""
    Return resolutions, orientations, and labels for sub-collimators.

    Returns
    -------

    """
    grids = {
        9: np.array([3, 20, 22]) - 1,
        8: np.array([16, 14, 32]) - 1,
        7: np.array([21, 26, 4]) - 1,
        6: np.array([24, 8, 28]) - 1,
        5: np.array([15, 27, 31]) - 1,
        4: np.array([6, 30, 2]) - 1,
        3: np.array([25, 5, 23]) - 1,
        2: np.array([7, 29, 1]) - 1,
        1: np.array([12, 19, 17]) - 1,
        0: np.array([11, 13, 18]) - 1,
    }

    labels = {i: [str(i + 1) + "a", str(i) + "b", str(i) + "c"] for i in range(10)}

    res32 = np.zeros(32)
    res32[grids[9]] = 178.6
    res32[grids[8]] = 124.9
    res32[grids[7]] = 87.3
    res32[grids[6]] = 61.0
    res32[grids[5]] = 42.7
    res32[grids[4]] = 29.8
    res32[grids[3]] = 20.9
    res32[grids[2]] = 14.6
    res32[grids[1]] = 10.2
    res32[grids[0]] = 7.1
    res10 = [7.1, 10.2, 14.6, 20.9, 29.8, 42.7, 61.0, 87.3, 124.9, 178.6]
    o32 = np.zeros(32)
    o32[grids[9]] = [150, 90, 30]
    o32[grids[8]] = [170, 110, 50]
    o32[grids[7]] = [10, 130, 70]
    o32[grids[6]] = [30, 150, 90]
    o32[grids[5]] = [50, 170, 110]
    o32[grids[4]] = [70, 10, 130]
    o32[grids[3]] = [90, 30, 150]
    o32[grids[2]] = [110, 50, 170]
    o32[grids[1]] = [130, 70, 10]
    o32[grids[0]] = [150, 90, 30]

    # g03_10 = [g03, g04, g05, g06, g07, g08, g09, g10]
    # g01_10 = [g01, g02, g03, g04, g05, g06, g07, g08, g09, g10]
    # g_plot = [g10, g05, g09, g04, g08, g03, g07, g02, g06, g01]
    # l_plot = [l10, l05, l09, l04, l08, l03, l07, l02, l06, l01]

    res = SimpleNamespace(res32=res32, o32=o32, res10=res10, label=labels)

    return res


@u.quantity_input
def create_meta_pixels(
    pixel_data: Union[RawPixelData, CompressedPixelData, SummedCompressedPixelData],
    time_range: Time,
    energy_range: Quantity["energy"],  # noqa: F821
    flare_location: SkyCoord,
    no_shadowing: bool = False,
):
    r"""
    Create meta-pixels by summing data with in given time and energy range.

    Parameters
    ----------
    pixel_data
        Input pixel data
    time_range :
        Start and end times
    energy_range
        Start and end energies
    flare_location
        The coordinates of flare used to calculate grid transmission
    no_shadowing : bool optional
        If set to True turn grid shadowing correction off
    Returns
    -------
    `dict`
        Visibility data and sub-collimator information
    """

    # checks if a time bin fully overlaps, is fully within, starts within, or ends within the specified time range.
    pixel_starts = pixel_data.times - pixel_data.duration / 2
    pixel_ends = pixel_data.times + pixel_data.duration / 2

    time_range_start = Time(time_range[0])
    time_range_end = Time(time_range[1])

    t_mask = (
        (pixel_starts >= time_range_start) & (pixel_ends <= time_range_end)  # fully within the time range
        | (time_range_start <= pixel_starts) & (time_range_end >= pixel_ends)  #  fully overlaps the time range
        | (pixel_starts <= time_range_start) & (pixel_ends >= time_range_start)  # starts within the time range
        | (pixel_starts <= time_range_end) & (pixel_ends >= time_range_end)  #  ends within the time range
    )

    e_mask = (pixel_data.energies["e_low"] >= energy_range[0]) & (pixel_data.energies["e_high"] <= energy_range[1])

    t_ind = np.argwhere(t_mask).ravel()
    e_ind = np.argwhere(e_mask).ravel()

    time_range = TimeRange(
        pixel_data.times[t_ind[0]] - pixel_data.duration[t_ind[0]] / 2,
        pixel_data.times[t_ind[-1]] + pixel_data.duration[t_ind[-1]] / 2,
    )

    changed = []
    for column in ["rcr", "pixel_masks", "detector_masks"]:
        if np.unique(pixel_data.data[column][t_ind], axis=0).shape[0] != 1:
            changed.append(column)
    if len(changed) > 0:
        raise ValueError(
            f"The following: {', '.join(changed)} changed in the selected time interval "
            f"please select a time interval where these are constant."
        )

    # fmt: off
    # For the moment copied from idl
    trigger_to_detector = [0, 0, 7, 7, 2, 1, 1, 6, 6, 5, 2, 3, 3, 4, 4, 5, 13, 12,
                           12, 11, 11, 10, 13, 14, 14, 9, 9, 10, 15, 15, 8, 8,]
    # fmt: on

    # Map the triggers to all 32 detectors
    triggers = pixel_data.data["triggers"][:, trigger_to_detector].astype(float)[...]

    _, livefrac, _ = get_livetime_fraction(triggers / pixel_data.data["timedel"].to("s").reshape(-1, 1))

    pixel_data.data["livefrac"] = livefrac

    e_cor_high, e_cor_low = get_elut_correction(e_ind, pixel_data)

    counts = pixel_data.data["counts"].astype(float)
    count_errors = np.sqrt(pixel_data.data["counts_comp_err"].astype(float).value ** 2 + counts.value) * u.ct
    ct = counts[t_ind][..., e_ind]
    ct[..., 0:8, 0] = ct[..., 0:8, 0] * e_cor_low[..., 0:8]
    ct[..., 0:8, -1] = ct[..., 0:8, -1] * e_cor_high[..., 0:8]
    ct_error = count_errors[t_ind][..., e_ind]
    ct_error[..., 0:8, 0] = ct_error[..., 0:8, 0] * e_cor_low[..., 0:8]
    ct_error[..., 0:8, -1] = ct_error[..., 0:8, -1] * e_cor_high[..., 0:8]

    lt = (livefrac * pixel_data.data["timedel"].reshape(-1, 1).to("s"))[t_ind].sum(axis=0)

    ct_summed = ct.sum(axis=(0, 3))  # .astype(float)
    ct_error_summed = np.sqrt(np.sum(ct_error**2, axis=(0, 3)))

    if not no_shadowing:
        if not isinstance(flare_location, STIXImaging) and flare_location.obstime != time_range.center:
            roll, solo_heeq, stix_pointing = get_hpc_info(time_range.start, time_range.stop)
            flare_location = flare_location.transform_to(STIXImaging(obstime=time_range.center, observer=solo_heeq))
        grid_shadowing = get_grid_transmission(flare_location.xyz[::-1])
        ct_summed = ct_summed / grid_shadowing.reshape(-1, 1) / 4  # transmission grid ~ 0.5*0.5 = .25
        ct_error_summed = ct_error_summed / grid_shadowing.reshape(-1, 1) / 4

    abcd_counts = ct_summed.reshape(ct_summed.shape[0], -1, 4)[:, [0, 1], :].sum(axis=1)
    abcd_count_errors = np.sqrt(
        (ct_error_summed.reshape(ct_error_summed.shape[0], -1, 4)[:, [0, 1], :] ** 2).sum(axis=1)
    )

    abcd_rate = abcd_counts / lt.reshape(-1, 1)
    abcd_rate_error = abcd_count_errors / lt.reshape(-1, 1)

    e_bin = pixel_data.energies[e_ind][-1]["e_high"] - pixel_data.energies[e_ind][0]["e_low"]
    abcd_rate_kev = abcd_rate / e_bin
    abcd_rate_error_kev = abcd_rate_error / e_bin

    # fmt: off
    # Taken from IDL
    pixel_areas = [0.096194997, 0.096194997, 0.096194997, 0.096194997, 0.096194997, 0.096194997,
                   0.096194997, 0.096194997, 0.010009999, 0.010009999, 0.010009999, 0.010009999,] * u.cm**2
    # fmt: on

    areas = pixel_areas.reshape(-1, 4)[0:2].sum(axis=0)

    meta_pixels = {
        "abcd_rate_kev_cm": abcd_rate_kev / areas,
        "abcd_rate_error_kev_cm": abcd_rate_error_kev / areas,
        "time_range": time_range,
        "energy_range": energy_range,
    }

    return meta_pixels


def get_elut_correction(e_ind, pixel_data):
    r"""
    Get ELUT correction factors

    Only correct the low and high energy edges as internal bins are contiguous.

    Parameters
    ----------
    e_ind : `np.ndarray`
        Energy channel indices
    pixel_data : `~stixpy.products.sources.CompressedPixelData`
        Pixel data

    Returns
    -------

    """
    energy_mask = pixel_data.energy_masks.energy_mask.astype(bool)
    elut = get_elut(pixel_data.time_range.center)
    ebin_edges_low = np.zeros((32, 12, 32), dtype=float)
    ebin_edges_low[..., 1:] = elut.e_actual
    ebin_edges_low = ebin_edges_low[..., energy_mask]
    ebin_edges_high = np.zeros((32, 12, 32), dtype=float)
    ebin_edges_high[..., 0:-1] = elut.e_actual
    ebin_edges_high[..., -1] = np.nan
    ebin_edges_high = ebin_edges_high[..., energy_mask]
    ebin_widths = ebin_edges_high - ebin_edges_low
    ebin_sci_edges_low = elut.e[..., 0:-1].value
    ebin_sci_edges_low = ebin_sci_edges_low[..., energy_mask]
    ebin_sci_edges_high = elut.e[..., 1:].value
    ebin_sci_edges_high = ebin_sci_edges_high[..., energy_mask]
    e_cor_low = (ebin_edges_high[..., e_ind[0]] - ebin_sci_edges_low[..., e_ind[0]]) / ebin_widths[..., e_ind[0]]
    e_cor_high = (ebin_sci_edges_high[..., e_ind[-1]] - ebin_edges_low[..., e_ind[-1]]) / ebin_widths[..., e_ind[-1]]
    return e_cor_high, e_cor_low


def create_visibility(meta_pixels):
    r"""
    Create visibilities from meta-pixels

    Parameters
    ----------
    meta_pixels

    Returns
    -------

    """
    abcd_rate_kev_cm = meta_pixels["abcd_rate_kev_cm"]
    abcd_rate_error_kev_cm = meta_pixels["abcd_rate_error_kev_cm"]
    real = abcd_rate_kev_cm[:, 2] - abcd_rate_kev_cm[:, 0]
    imag = abcd_rate_kev_cm[:, 3] - abcd_rate_kev_cm[:, 1]

    real_err = np.sqrt(abcd_rate_error_kev_cm[:, 2] ** 2 + abcd_rate_error_kev_cm[:, 0] ** 2).value * real.unit
    imag_err = np.sqrt(abcd_rate_error_kev_cm[:, 3] ** 2 + abcd_rate_error_kev_cm[:, 1] ** 2).value * real.unit

    vis_info = get_uv_points_data()

    # Compute raw phases
    phase = np.arctan2(imag, real)

    # Compute amplitudes
    observed_amplitude = np.sqrt(real**2 + imag**2)

    # Compute error on amplitudes
    amplitude_error = np.sqrt((real / observed_amplitude * real_err) ** 2 + (imag / observed_amplitude * imag_err) ** 2)

    # Apply pixel correction
    phase += 46.1 * u.deg  # Center of large pixel in terms morie pattern phase
    # TODO add correction for small pixel

    obsvis = (np.cos(phase) + np.sin(phase) * 1j) * observed_amplitude

    imaging_detector_indices = vis_info["isc"] - 1

    # vis = SimpleNamespace(
    #     obsvis=obsvis[imaging_detector_indices],
    #     amplitude=observed_amplitude[imaging_detector_indices],
    #     amplitude_error=amplitude_error[imaging_detector_indices],
    #     phase=phase[imaging_detector_indices],
    #     **vis_info,
    # )

    vis_meta = VisMeta(
        instrumet="STIX",
        spectral_range=meta_pixels["energy_range"],
        time_range=Time([meta_pixels["time_range"].start, meta_pixels["time_range"].end]),
        vis_labels=vis_info["label"].tolist(),
        isc=vis_info["isc"].value,
        calibrated=False,
    )
    vis = Visibilities(
        obsvis[imaging_detector_indices],
        u=vis_info["u"],
        v=vis_info["v"],
        amplitude=observed_amplitude[imaging_detector_indices],
        amplitude_uncertainty=amplitude_error[imaging_detector_indices],
        meta=vis_meta,
    )
    return vis


@u.quantity_input
def get_uv_points_data(d_det: u.Quantity[u.mm] = 47.78 * u.mm, d_sep: u.Quantity[u.mm] = 545.30 * u.mm):
    r"""
    Return the STIX (u,v) points coordinates defined in [1], ordered with respect to the detector index.

    Parameters
    ----------
    d_det: `u.Quantity[u.mm]` optional
        Distance between the rear grid and the detector plane (in mm). Default, 47.78 * u.mm

    d_sep: `u.Quantity[u.mm]` optional
        Distance between the front and the rear grid (in mm). Default, 545.30 * u.mm

    Returns
    -------
    A dictionary containing sub-collimator indices, sub-collimator labels and coordinates of the STIX (u,v) points (defined in arcsec :sup:`-1`)

    References
    ----------
    [1] Massa et al., 2023, The STIX Imaging Concept, Solar Physics, 298,
        https://doi.org/10.1007/s11207-023-02205-7

    """

    subc = read_subc_params()
    imaging_ind = np.where((subc["Grid Label"] != "cfl") & (subc["Grid Label"] != "bkg"))

    # filter out background monitor and flare locator
    subc_imaging = subc[imaging_ind]

    # assign detector numbers to visibility index of subcollimator (isc)
    isc = subc_imaging["Det #"]

    # assign the stix sc label for convenience
    label = subc_imaging["Grid Label"]

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging["Phase Sense"]

    # see Equation (9) in [1]
    front_unit_vector_comp = (((d_det + d_sep) / subc_imaging["Front Pitch"]) / u.rad).to(1 / u.arcsec)
    rear_unit_vector_comp = ((d_det / subc_imaging["Rear Pitch"]) / u.rad).to(1 / u.arcsec)

    uu = (
        np.cos(subc_imaging["Front Orient"].to("deg")) * front_unit_vector_comp
        - np.cos(subc_imaging["Rear Orient"].to("deg")) * rear_unit_vector_comp
    )
    vv = (
        np.sin(subc_imaging["Front Orient"].to("deg")) * front_unit_vector_comp
        - np.sin(subc_imaging["Rear Orient"].to("deg")) * rear_unit_vector_comp
    )

    uu = -uu * phase_sense
    vv = -vv * phase_sense

    uv_data = {
        "isc": isc,  # sub-collimator indices
        "label": label,  # sub-collimator labels
        "u": uu,
        "v": vv,
    }

    return uv_data


def calibrate_visibility(vis: Visibilities, flare_location: SkyCoord = STIXImaging(0 * u.arcsec, 0 * u.arcsec)):
    """
    Calibrate visibility phase and amplitudes.

    Parameters
    ----------
    vis :
        Uncalibrated Visibilities
    flare_location
        The location of the flare the visibility are shifted to have the phase center at this location

    Returns
    -------

    """
    modulation_efficiency = np.pi**3 / (8 * np.sqrt(2))

    # Grid correction factors
    grid_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "GridCorrection.csv"
    grid_cal = Table.read(grid_cal_file, header_start=2, data_start=3)
    grid_corr = grid_cal["Phase correction factor"][vis.meta["isc"] - 1] * u.deg

    # Phase correction
    phase_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "PhaseCorrFactors.csv"
    phase_cal = Table.read(phase_cal_file, header_start=3, data_start=4)
    phase_corr = phase_cal["Phase correction factor"][vis.meta["isc"] - 1] * u.deg

    tr = TimeRange(vis.meta.time_range)

    if not isinstance(flare_location, STIXImaging) and flare_location.obstime != tr.center:
        roll, solo_heeq, stix_pointing = get_hpc_info(vis.meta.time_range[0], vis.meta.time_range[1])
        solo_coord = HeliographicStonyhurst(solo_heeq, representation_type="cartesian", obstime=tr.center)
        flare_location = flare_location.transform_to(STIXImaging(obstime=tr.center, observer=solo_coord))

    phase_mapcenter_corr = -2 * np.pi * (flare_location.Tx * vis.u + flare_location.Ty * vis.v) * u.rad

    ovis = vis.visibilities[:]
    ovis_real = np.real(ovis)
    ovis_imag = np.imag(ovis)

    ovis_amp = np.sqrt(ovis_real**2 + ovis_imag**2)
    ovis_phase = np.arctan2(ovis_imag, ovis_real).to("deg")

    calibrated_amplitude = ovis_amp * modulation_efficiency
    calibrated_phase = ovis_phase + grid_corr + phase_corr + phase_mapcenter_corr

    # Compute error on amplitudes
    systematic_error = 5 * u.percent  # from G. Hurford 5% systematics

    calibrated_amplitude_error = np.sqrt(
        (vis.amplitude_uncertainty * modulation_efficiency) ** 2
        + (systematic_error * calibrated_amplitude).to(ovis_amp.unit) ** 2
    )

    calibrated_visibility = (np.cos(calibrated_phase) + np.sin(calibrated_phase) * 1j) * calibrated_amplitude

    vis.meta["calibrated"] = True
    cal_vis = Visibilities(
        calibrated_visibility,
        u=vis.u,
        v=vis.v,
        phase_center=flare_location,
        amplitude=calibrated_amplitude,
        amplitude_uncertainty=calibrated_amplitude_error,
        meta=vis.meta,
    )

    return cal_vis
