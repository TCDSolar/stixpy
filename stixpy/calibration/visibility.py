from types import SimpleNamespace
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.time import Time

from stixpy.calibration.energy import get_elut
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.io.readers import read_subc_params


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


def create_meta_pixels(pixel_data, time_range, energy_range, phase_center, no_shadowing=False):
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
    phase_center
        The coordinates of the phase center
    no_shadowing : bool optional
        If set to True turn grid shadowing correction off
    Returns
    -------
    `dict`
        Visibility data and sub-collimator information
    """
    t_mask = (pixel_data.times >= Time(time_range[0])) & (pixel_data.times <= Time(time_range[1]))
    e_mask = (pixel_data.energies["e_low"] >= energy_range[0] * u.keV) & (
        pixel_data.energies["e_high"] <= energy_range[1] * u.keV
    )

    t_ind = np.argwhere(t_mask).ravel()
    e_ind = np.argwhere(e_mask).ravel()

    changed = []
    for column in ["rcr", "pixel_masks", "detector_masks"]:
        if np.unique(pixel_data.data[column][t_ind], axis=0).shape[0] != 1:
            changed.append(column)
    if len(changed) > 0:
        raise ValueError(
            f"The following: {', '.join(changed)} changed in the selected time interval "
            f"please select a time interval where these are constant."
        )

    # For the moment copied from idl
    trigger_to_detector = [
        0,
        0,
        7,
        7,
        2,
        1,
        1,
        6,
        6,
        5,
        2,
        3,
        3,
        4,
        4,
        5,
        13,
        12,
        12,
        11,
        11,
        10,
        13,
        14,
        14,
        9,
        9,
        10,
        15,
        15,
        8,
        8,
    ]

    # Map the triggers to all 32 detectors
    triggers = pixel_data.data["triggers"][:, trigger_to_detector].astype(float)[...]

    _, livefrac, _ = get_livetime_fraction(triggers / pixel_data.data["timedel"].to("s").reshape(-1, 1))

    pixel_data.data["livefrac"] = livefrac

    elut = get_elut(pixel_data.time_range.center)
    ebin_edges_low = elut.e_actual[..., 0:-1]
    ebin_edges_high = elut.e_actual[..., 1:]
    ebin_widths = ebin_edges_high - ebin_edges_low

    e_cor_low = (ebin_edges_high[..., e_ind[0]] - elut.e[..., e_ind[0] + 1].value) / ebin_widths[..., e_ind[0]]
    e_cor_high = (elut.e[..., e_ind[-1] + 2].value - ebin_edges_high[..., e_ind[-1] - 1]) / ebin_widths[..., e_ind[-1]]

    e_cor = (  # noqa
        (ebin_edges_high[..., e_ind[-1]] - ebin_edges_low[..., e_ind[0]])
        * u.keV
        / (elut.e[e_ind[-1] + 1] - elut.e[e_ind[0]])
    )

    counts = pixel_data.data["counts"]
    ct = counts[t_ind][..., e_ind].astype(float)
    ct[..., 0:8, 0] = ct[..., 0:8, 0] * e_cor_low[..., 0:8]
    ct[..., 0:8, -1] = ct[..., 0:8, -1] * e_cor_high[..., 0:8]

    lt = (livefrac * pixel_data.data["timedel"].reshape(-1, 1).to("s"))[t_ind].sum(axis=0)

    ct_sumed = ct.sum(axis=(0, 3)).astype(float)

    err_sumed = (np.sqrt(ct.sum(axis=(0, 3)).value)) * u.ct

    if not no_shadowing:
        grid_shadowing = get_grid_transmission(phase_center)
        ct_sumed = ct_sumed / grid_shadowing.reshape(-1, 1) / 4  # transmission grid ~ 0.5*0.5 = .25
        err_sumed = err_sumed / grid_shadowing.reshape(-1, 1) / 4

    abcd_counts = ct_sumed.reshape(ct_sumed.shape[0], -1, 4)[:, [0, 1], :].sum(axis=1)
    abcd_count_errors = np.sqrt((err_sumed.reshape(ct_sumed.shape[0], -1, 4)[:, [0, 1], :] ** 2).sum(axis=1))

    abcd_rate = abcd_counts / lt.reshape(-1, 1)
    abcd_rate_error = abcd_count_errors / lt.reshape(-1, 1)

    e_bin = pixel_data.energies[e_ind][-1]["e_high"] - pixel_data.energies[e_ind][0]["e_low"]
    abcd_rate_kev = abcd_rate / e_bin
    abcd_rate_error_kev = abcd_rate_error / e_bin

    # Taken from IDL
    pixel_areas = [
        0.096194997,
        0.096194997,
        0.096194997,
        0.096194997,
        0.096194997,
        0.096194997,
        0.096194997,
        0.096194997,
        0.010009999,
        0.010009999,
        0.010009999,
        0.010009999,
    ] * u.cm**2

    areas = np.array(pixel_areas).reshape(-1, 4)[0:2].sum(axis=0)

    meta_pixels = {"abcd_rate_kev_cm": abcd_rate_kev / areas, "abcd_rate_error_kev_cm": abcd_rate_error_kev / areas}

    return meta_pixels


def create_visibility(meta_pixels):
    r"""
    Create visibilities from meta-pixels

    Parameters
    ----------
    meta_pixels

    Returns
    -------

    """
    abcd_rate_kev_cm, abcd_rate_error_kev_cm = meta_pixels.values()
    real = abcd_rate_kev_cm[:, 2] - abcd_rate_kev_cm[:, 0]
    imag = abcd_rate_kev_cm[:, 3] - abcd_rate_kev_cm[:, 1]

    real_err = np.sqrt(abcd_rate_error_kev_cm[:, 2] ** 2 + abcd_rate_error_kev_cm[:, 0] ** 2).value * real.unit
    imag_err = np.sqrt(abcd_rate_error_kev_cm[:, 3] ** 2 + abcd_rate_error_kev_cm[:, 1] ** 2).value * real.unit

    vis = get_visibility_info()

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

    imaging_detector_indices = vis["isc"] - 1

    vis = SimpleNamespace(
        obsvis=obsvis[imaging_detector_indices],
        amplitude=observed_amplitude[imaging_detector_indices],
        amplitude_error=amplitude_error[imaging_detector_indices],
        phase=phase[imaging_detector_indices],
        **vis,
    )

    return vis


def get_visibility_info(grid_separation=550 * u.mm):
    r"""
    Return the sub-collimator data and corresponding u, v points.

    Parameters
    ----------
    grid_separation : `float`, optional
        Front to read grid separation

    Returns
    -------

    """
    # imaging sub-collimators
    subc = read_subc_params()
    imaging_ind = np.where((subc["Grid Label"] != "cfl") & (subc["Grid Label"] != "bkg"))

    # filter out background monitor and flare locator
    subc_imaging = subc[imaging_ind]

    # take average of front and rear grid pitches (mm)
    pitch = (subc_imaging["Front Pitch"] + subc_imaging["Rear Pitch"]) / 2.0

    # convert pitch from mm to arcsec
    # TODO check diff not using small angle approx
    pitch = (pitch / grid_separation).decompose() * u.rad

    # take average of front and rear grid orientation
    orientation = (subc_imaging["Front Orient"] + subc_imaging["Rear Orient"]) / 2.0

    # calculate number of frequency components
    len(subc_imaging)

    # assign detector numbers to visibility index of subcollimator (isc)
    isc = subc_imaging["Det #"]

    # assign the stix sc label for convenience
    label = subc_imaging["Grid Label"]

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging["Phase Sense"]

    # calculate u and v
    uv = 1.0 / pitch.to("arcsec")
    uu = uv * np.cos(orientation.to("rad"))
    vv = uv * np.sin(orientation.to("rad"))

    # Add the lifetime isc association. This gives the lifetime pairings
    # vis.live_time = stx_ltpair_assignment(vis.isc)

    vis = {
        "pitch": pitch,
        "orientation": orientation,
        "isc": isc,
        "label": label,
        "phase_sense": phase_sense,
        "u": uu,
        "v": vv,
    }
    return vis


def get_visibility_info_giordano():
    L1 = 545.30 * u.mm
    L2 = 47.78 * u.mm

    subc = read_subc_params()
    imaging_ind = np.where((subc["Grid Label"] != "cfl") & (subc["Grid Label"] != "bkg"))

    # np.where((subc['Grid Label'] != 'cfl') & (subc['Grid Label'] != 'bkg'))

    # filter out background monitor and flare locator
    subc_imaging = subc[imaging_ind]

    # take average of front and rear grid pitches (mm)
    # pitch = (subc_imaging['Front Pitch'] + subc_imaging['Rear Pitch']) / 2.0

    # convert pitch from mm to arcsec
    # TODO check diff not using small angle approx
    # pitch = (pitch / L1).decompose() * u.rad

    # take average of front and rear grid orientation
    # (subc_imaging['Front Orient'] + subc_imaging['Rear Orient']) / 2.0

    # calculate number of frequency components
    # len(subc_imaging)

    # assign detector numbers to visibility index of subcollimator (isc)
    # isc = subc_imaging['Det #']

    # assign the stix sc label for convenience
    subc_imaging["Grid Label"]

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging["Phase Sense"]

    pitch_front = (1 / subc_imaging["Front Pitch"] * (L2 + L1)).value * 1 / u.rad
    pitch_rear = (1 / subc_imaging["Rear Pitch"] * L2).value * 1 / u.rad

    uu = np.cos(subc_imaging["Front Orient"].to("deg")) * pitch_front.to(1 / u.arcsec) - np.cos(
        subc_imaging["Rear Orient"].to("deg")
    ) * pitch_rear.to(1 / u.arcsec)
    vv = np.sin(subc_imaging["Front Orient"].to("deg")) * pitch_front.to(1 / u.arcsec) - np.sin(
        subc_imaging["Rear Orient"].to("deg")
    ) * pitch_rear.to(1 / u.arcsec)

    uu = -uu * phase_sense
    vv = -vv * phase_sense

    return uu.to(1 / u.arcsec), vv.to(1 / u.arcsec)


def calibrate_visibility(vis, flare_location=(0, 0) * u.arcsec):
    """
    Calibrate visibility phase and amplitudes.

    Parameters
    ----------
    vis :
        Uncalibrated Visibilities

    Returns
    -------

    """
    modulation_efficiency = np.pi**3 / (8 * np.sqrt(2))

    # Grid correction factors
    grid_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "GridCorrection.csv"
    grid_cal = Table.read(grid_cal_file, header_start=2, data_start=3)
    grid_corr = grid_cal["Phase correction factor"][vis.isc - 1] * u.deg

    # Phase correction
    phase_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "PhaseCorrFactors.csv"
    phase_cal = Table.read(phase_cal_file, header_start=3, data_start=4)
    phase_corr = phase_cal["Phase correction factor"][vis.isc - 1] * u.deg

    phase_mapcenter_corr = (
        -2 * np.pi * (flare_location[0].value * u.arcsec * vis.u + flare_location[1].value * u.arcsec * vis.v) * u.rad
    )

    ovis = vis.obsvis[:]
    ovis_real = np.real(ovis)
    ovis_imag = np.imag(ovis)

    ovis_amp = np.sqrt(ovis_real**2 + ovis_imag**2)
    ovis_phase = np.arctan2(ovis_imag, ovis_real).to("deg")

    calibrated_amplitude = ovis_amp * modulation_efficiency
    calibrated_phase = ovis_phase + grid_corr + phase_corr + phase_mapcenter_corr

    # Compute error on amplitudes
    systematic_error = 5 * u.percent  # from G. Hurford 5% systematics

    calibrated_amplitude_error = np.sqrt(
        (vis.amplitude_error * modulation_efficiency) ** 2
        + (systematic_error * calibrated_amplitude).to(ovis_amp.unit) ** 2
    )

    calibrated_visibility = (np.cos(calibrated_phase) + np.sin(calibrated_phase) * 1j) * calibrated_amplitude

    orig_vis = vis.__dict__
    [orig_vis.pop(n) for n in ["obsvis", "amplitude", "amplitude_error", "phase"]]

    cal_vis = SimpleNamespace(
        obsvis=calibrated_visibility,
        amplitude=calibrated_amplitude,
        amplitude_error=calibrated_amplitude_error,
        phase=calibrated_phase,
        **orig_vis,
    )

    return cal_vis


def correct_phase_projection(vis, flare_xy):
    r"""
    Correct visibilities for projection of the grids onto the detectors.

    The rear grids are not in direct contact with the detectors so the phase of the visibilities
    needs to be project onto the grid and this is dependent on the position of the xray emission

    Parameters
    ----------
    vis
        Input visibilities
    flare_xy
        Position of the flare

    Returns
    -------

    """
    # Phase projection correction
    l1 = 550 * u.mm  # separation of front rear grid
    l2 = 47 * u.mm  # separation of rear grid and detector
    # TODO check how this works out to degrees
    vis.phase -= (
        flare_xy[1].to_value("arcsec") * 360.0 * np.pi / (180.0 * 3600.0 * 8.8 * u.mm) * (l2 + l1 / 2.0)
    ) * u.deg

    vis.u *= -vis.phase_sense
    vis.v *= -vis.phase_sense

    # Compute real and imaginary part of the visibility
    vis.obsvis = vis.amplitude * (np.cos(vis.phase.to("rad")) + np.sin(vis.phase.to("rad")) * 1j)

    # Add phase factor for shifting the flare_xy
    map_center = flare_xy  # - [26.1, 58.2] * u.arcsec
    # Subtract Frederic's mean shift values

    # TODO check the u, v why is flare_xy[1] used with u (prob stix vs spacecraft orientation
    phase_mapcenter = -2 * np.pi * (map_center[0] * vis.u - map_center[1] * vis.v) * u.rad
    vis.obsvis *= np.cos(phase_mapcenter) + np.sin(phase_mapcenter) * 1j

    return vis


def sas_map_center():
    # receter map at 0,0 taking account of mean or actual sas sol
    pass
