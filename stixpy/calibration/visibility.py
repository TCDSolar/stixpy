from pathlib import Path
from types import SimpleNamespace

import numpy as np

import astropy.units as u
from astropy.table import Table
from astropy.time import Time

from stixpy.calibration.energy import get_elut
from stixpy.calibration.grid import get_grid_internal_shadowing
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.io.readers import read_subc_params


def get_subcollimator_info():
    r"""
    Return resolutions, orientations and labels for sub-collimators.

    Returns
    -------

    """
    grids = {9: np.array([3, 20, 22]) - 1, 8: np.array([16, 14, 32]) - 1,
             7: np.array([21, 26, 4]) - 1, 6: np.array([24, 8, 28]) - 1,
             5: np.array([15, 27, 31]) - 1, 4: np.array([6, 30, 2]) - 1,
             3: np.array([25, 5, 23]) - 1, 2: np.array([7, 29, 1]) - 1,
             1: np.array([12, 19, 17]) - 1, 0: np.array([11, 13, 18]) - 1}

    labels = {i: [str(i+1)+'a', str(i)+'b', str(i)+'c'] for i in range(10)}

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


def create_visibility(pixel_data, time_range, energy_range, phase_center, no_shadowing=False):
    r"""
    Return visibility created from pixel data with in given time and energy range.


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
        If set to Ture turn grid shadowing correction off
    Returns
    -------
    `dict`
        Visibility data and sub-collimator information
    """
    t_ind = (pixel_data.times >= Time(time_range[0])) & (pixel_data.times <= Time(time_range[1]))
    e_ind = ((pixel_data.energies['e_low'] >= energy_range[0] * u.keV)
             & (pixel_data.energies['e_high'] <= energy_range[1] * u.keV))

    counts, counts_err, times, t_norm, energies = pixel_data.get_data(time_indices=t_ind,
                                                                      energy_indices=e_ind)

    # For the moment copied from idl
    trigger_to_detector = [0, 0, 7, 7, 2, 1, 1, 6, 6, 5, 2, 3, 3, 4, 4, 5, 13,
                           12, 12, 11, 11, 10, 13, 14, 14, 9, 9, 10, 15, 15, 8, 8]

    # Map the triggers to all 32 detectors
    triggers = pixel_data.data['triggers'][:, trigger_to_detector].astype(float)[...]

    # Update the triggers for the CFL and BK groups as the exposed area are very different
    # Sam / Ewan ??
    # bkg_ratio = 1.1
    # cfl_ratio = 1.33
    # triggers[:, 8] = triggers[:, 8] / (1 + cfl_ratio) * cfl_ratio  # CFL
    # triggers[:, 7] = triggers[:, 7] / (1 + cfl_ratio)  # Detector share with CFL
    #
    # triggers[:, 9] = triggers[:, 9] / (1 + bkg_ratio) * bkg_ratio  # BKG
    # triggers[:, 15] = triggers[:, 15] / (1 + bkg_ratio)  # Detector shared with BKG

    _, livefrac, _ = get_livetime_fraction(
        triggers / pixel_data.data['timedel'].to('s').reshape(-1, 1))

    pixel_data.data['livefrac'] = livefrac

    elut = get_elut(pixel_data.time_range.center)
    real_dE = elut.e_width_actual[..., (pixel_data.energy_masks.masks[0] == 1)[1:-1]] * u.keV

    counts = pixel_data.data['counts']
    rate = (counts / (real_dE * u.keV * livefrac.reshape(45, 32, 1, 1)
                      * pixel_data.data['timedel'].to('s').reshape(-1, 1, 1, 1)))

    ct = counts[t_ind][..., e_ind]
    rate[t_ind][..., e_ind]

    # er = real_dE[..., e_ind].sum(-1)
    er = elut.e_width_actual[:, :, 3:7].sum(axis=-1) * u.keV
    lt = (livefrac * pixel_data.data['timedel'].reshape(-1, 1).to('s'))[t_ind].sum(axis=0)

    ct_sumed = ct.sum(axis=(0, 3)) / (er * lt.to('s').reshape(-1, 1))
    err_sumed = np.sqrt(ct.sum(axis=(0, 3))) / (er * lt.to('s').reshape(-1, 1))

    grid_shadowing = get_grid_internal_shadowing(phase_center)

    if not no_shadowing:
        ct_sumed = ct_sumed / grid_shadowing.reshape(-1, 1) / 4  # transmission grid ~ 0.5*0.5 = .25
        err_sumed = err_sumed / grid_shadowing.reshape(-1, 1) / 4

    abcd_rate = ct_sumed.reshape(-1, 3, 4)[:, [0, 1], :].sum(axis=1)

    # Taken from IDL
    pixel_areas = [0.096194997, 0.096194997, 0.096194997, 0.096194997, 0.096194997, 0.096194997,
                   0.096194997, 0.096194997, 0.010009999, 0.010009999, 0.010009999, 0.010009999]

    areas = np.tile(np.array(pixel_areas).reshape(1, -1), (32, 1)) * u.cm

    areas_summed = areas.reshape(-1, 3, 4)[:, [0, 1], :].sum(axis=1)

    aaaa = abcd_rate / areas_summed

    aaaa_err = np.sqrt(np.sum(err_sumed.reshape(-1, 3, 4)[:, [0, 1], :] ** 2, axis=1))
    aaaa_err = aaaa_err / areas_summed

    real = aaaa[:, 2] - aaaa[:, 0]
    imag = aaaa[:, 3] - aaaa[:, 1]

    real_err = np.sqrt(aaaa_err[:, 2] ** 2 + aaaa_err[:, 0] ** 2).value * real.unit
    imag_err = np.sqrt(aaaa_err[:, 3] ** 2 + aaaa_err[:, 1] ** 2).value * real.unit

    vis = get_visibility_info()
    vis['real'] = real
    vis['imag'] = imag
    vis['real_err'] = real_err
    vis['imag_err'] = imag_err
    vis['rate'] = aaaa
    vis['reate_err'] = aaaa_err

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
    imaging_ind = np.where((subc['Grid Label'] != 'cfl') & (subc['Grid Label'] != 'bkg'))

    # filter out background monitor and flare locator
    subc_imaging = subc[imaging_ind]

    # take average of front and rear grid pitches (mm)
    pitch = (subc_imaging['Front Pitch'] + subc_imaging['Rear Pitch']) / 2.0

    # convert pitch from mm to arcsec
    # TODO check diff not using small angle approx
    pitch = (pitch / grid_separation).decompose() * u.rad

    # take average of front and rear grid orientation
    orientation = (subc_imaging['Front Orient'] + subc_imaging['Rear Orient']) / 2.0

    # calculate number of frequency components
    len(subc_imaging)

    # assign detector numbers to visibility index of subcollimator (isc)
    isc = subc_imaging['Det #']

    # assign the stix sc label for convenience
    label = subc_imaging['Grid Label']

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging['Phase Sense']

    # calculate u and v
    uv = 1.0 / pitch.to('arcsec')
    uu = uv * np.cos(orientation.to('rad'))
    vv = uv * np.sin(orientation.to('rad'))

    # Add the livetime isc association. This gives the livetime pairings
    # vis.live_time = stx_ltpair_assignment(vis.isc)

    vis = {
        'pitch': pitch,
        'orientation': orientation,
        'isc': isc,
        'label': label,
        'phase_sense': phase_sense,
        'u': uu,
        'v': vv
    }
    return vis


def get_visibility_info_giordano():
    L1 = 550 * u.mm
    L2 = 47 * u.mm

    subc = read_subc_params()
    imaging_ind = np.arange(32)
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
    subc_imaging['Grid Label']

    # save phase orientation of the grids to the visibility
    phase_sense = subc_imaging['Phase Sense']

    pitch_front = (1 / subc_imaging['Front Pitch'] * (L2 + L1)).value * 1/u.rad
    pitch_rear = (1/subc_imaging['Rear Pitch'] * L2).value * 1/u.rad

    uu = (np.cos(subc_imaging['Front Orient'].to('deg')) * pitch_front.to(1/u.arcsec)
          - np.cos(subc_imaging['Rear Orient'].to('deg')) * pitch_rear.to(1/u.arcsec))
    vv = (np.sin(subc_imaging['Front Orient'].to('deg')) * pitch_front.to(1/u.arcsec)
          - np.sin(subc_imaging['Rear Orient'].to('deg')) * pitch_rear.to(1/u.arcsec))

    uu = -uu * phase_sense
    vv = -vv * phase_sense

    return uu.to(1/u.arcsec), vv.to(1/u.arcsec)


def calibrate_visibility(vis, flare_location=(0, 0) * u.arcsec):
    """
    Return phase and amplitude calibrated visibilities.

    Parameters
    ----------
    vis :
        Uncalibrated Visibilities

    Returns
    -------

    """
    real_component = vis.pop('real')
    imag_component = vis.pop('imag')
    real_component_err = vis.pop('real_err')
    imag_component_err = vis.pop('imag_err')

    # Compute raw phases
    phase = np.arctan2(imag_component, real_component)

    # Compute amplitudes
    modulation_efficiency = np.pi ** 3 / (8 * np.sqrt(2))  # Modulation efficiency
    observed_amplitude = np.sqrt(real_component ** 2 + imag_component ** 2)
    calibrated_amplitude = observed_amplitude * modulation_efficiency

    # Compute error on amplitudes
    systematic_error = 5 * u.percent  # from G. Hurford 5% systematics
    amplitude_stat_error = ((np.sqrt((real_component / observed_amplitude * real_component_err) ** 2
                            + (imag_component / observed_amplitude * imag_component_err) ** 2))
                            * modulation_efficiency)
    amplitude_error = np.sqrt(amplitude_stat_error ** 2 +
                              (systematic_error * calibrated_amplitude)
                              .to(observed_amplitude.unit) ** 2)

    # Apply pixel correction
    phase += 46.1 * u.deg  # Center of large pixel in terms of phase of morie pattern
    # TODO add correction for small pixel

    # Grid correction factors
    grid_cal_file = Path(__file__).parent.parent / 'config' / 'data' / 'grid' / 'GridCorrection.csv'
    grid_cal = Table.read(grid_cal_file, header_start=2, data_start=3)
    grid_corr = grid_cal['Phase correction factor'] * u.deg

    # Apply grid correction
    phase += grid_corr

    # Phase correction
    phase_cal_file = Path(__file__).parent.parent / 'config' / 'data' / 'grid' / 'PhaseCorrFactors.csv'
    phase_cal = Table.read(phase_cal_file, header_start=3, data_start=4)
    phase_corr = phase_cal['Phase correction factor'] * u.deg

    # Apply grid correction
    phase += phase_corr

    # Compute error on phases
    phase_error = amplitude_error / calibrated_amplitude

    imaging_detector_indices = vis['isc'] - 1

    vis = SimpleNamespace(amplitude=calibrated_amplitude[imaging_detector_indices],
                          amplitude_error=amplitude_error[imaging_detector_indices],
                          phase=phase[imaging_detector_indices],
                          phase_error=phase_error[imaging_detector_indices],
                          **vis)

    return vis


def correct_phase_projection(vis, flare_xy):
    r"""
    Correct visibilities for projection of the grids onto the detectors.

    The rear grids are not in direct contact with the detectors so the phase of the visibilities
    needs to be project onto the grid and this is dependant on the position of the xray emission

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
    l1 = 550 * u.mm  # seperation of front rear grid
    l2 = 47 * u.mm  # seperation of rear grid and detector
    # TODO check how this works out to degress
    vis.phase -= (flare_xy[1].to_value('arcsec') * 360. * np.pi
                  / (180. * 3600. * 8.8 * u.mm) * (l2 + l1 / 2.0)) * u.deg

    vis.u *= -vis.phase_sense
    vis.v *= -vis.phase_sense

    # Compute real and imaginary part of the visibility
    vis.obsvis = vis.amplitude * (np.cos(vis.phase.to('rad')) + np.sin(vis.phase.to('rad')) * 1j)

    # Add phase factor for shifting the flare_xy
    map_center = flare_xy #- [26.1, 58.2] * u.arcsec
    # Subtract Frederic's mean shift values

    # TODO check the u, v why is flare_xy[1] used with u (prob stix vs spacecraft orientation
    phase_mapcenter = -2 * np.pi * (map_center[1] * vis.u - map_center[0] * vis.v) * u.rad
    vis.obsvis *= (np.cos(phase_mapcenter) + np.sin(phase_mapcenter) * 1j)

    return vis


def sas_map_center():
    # receter map at 0,0 taking account of mean or actual sas sol
    pass
