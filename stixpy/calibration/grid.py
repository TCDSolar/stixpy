 
from pathlib import Path
 
import numpy as np
 
import astropy.units as u
from astropy.table import Table
from roentgen.absorption import MassAttenuationCoefficient
 
from stixpy.coordinates.frames import STIXImaging
 
 
__all__ = ["get_grid_transmission", "stx_grid_transmission"]
 
 
# Subcollimator numbers, 0-indexed (detector = subcollimator - 1), matching the
# indexing convention used by callers of this module.
CFL_BKG_0 = [8, 9]            # sc 9 (CFL), sc 10 (BKG) -- no physical grid
FINEST_2ABC_0 = [11, 16, 18]  # sc 12 (2a), sc 17 (2c), sc 19 (2b) -> pitch/2, thick=0.2mm
FINEST_1ABC_0 = [10, 12, 17]  # sc 11 (1a), sc 13 (1b), sc 18 (1c) -> pitch/3, thick=0.133mm
FINEST_ALL_0 = FINEST_2ABC_0 + FINEST_1ABC_0
 
# Cache the roentgen coefficient object -- constructing it re-reads tabulated data.
_W_MASS_ATTEN = MassAttenuationCoefficient("W")
 
 
def _tungsten_path_length(ph_energy):
    """
    Photon energy [keV] -> attenuation path length L [mm] in tungsten, using
    roentgen's tabulated mass attenuation coefficient. Mirrors IDL's
    L = 1 / (mass_attenuation * density / 10), with density = 19.30 g/cm^3.
    """
    energy = np.atleast_1d(ph_energy).astype(float) * u.keV
    mass_atten = _W_MASS_ATTEN.func(energy)           # cm^2/g
    density_w = 19.30 * u.g / u.cm**3
    mu_linear = mass_atten * density_w                # cm^-1
    mu_mm = mu_linear.to(1 / u.mm).value              # mm^-1
    return 1.0 / mu_mm                                # mm
 
 
def stx_grid_transmission(pitch, slit, thickness, L, ds=5e-3, dh=5e-2):
    r"""
    Wedge-shape-model transmission for a single grid layer (front or rear),
    vectorized over photon energy.
 
    Parameters
    ----------
    pitch, slit, thickness : float
        Grid geometry for one subcollimator [mm].
    L : numpy.ndarray
        Tungsten attenuation path length per energy [mm], shape (n_energies,).
    ds, dh : float
        Wedge-model imperfection parameters. IDL sets both to 0 for the finest
        grids (1a/b/c, 2a/b/c), which disables the shadowing-correction term
        entirely rather than evaluating it at ds = dh = 0.
 
    Returns
    -------
    numpy.ndarray
        Transmission per energy, shape (n_energies,).
    """
    L = np.asarray(L, dtype=float)
 
    g0 = slit / pitch + (pitch - slit) / pitch * np.exp(-thickness / L)
 
    if ds == 0 and dh == 0:
        return g0
 
    ttt = L / dh * (1.0 - np.exp(-dh / L))
    g1 = 2.0 * ds / pitch * (ttt - np.exp(-thickness / L))
 
    return g0 + g1
 
 
def get_grid_transmission(ph_energy, detectors, flare_location: STIXImaging):
    r"""
    Return the grid transmission for the requested sub-collimators, corrected
    for internal shadowing, including the finest grids (1a/b/c, 2a/b/c) with
    their rescaled pitch/thickness and disabled shadowing term.
 
    Parameters
    ----------
    ph_energy : array_like
        Photon energies [keV].
    detectors : array_like
        0-indexed subcollimator numbers to compute transmission for
        (detector = subcollimator - 1, e.g. detector 0 = subcollimator 1).
    flare_location : STIXImaging, array-like, or None
        Location of the flare, in arcsec ([Tx, Ty]). If None, on-axis (0, 0)
        is assumed.
 
    Returns
    -------
    numpy.ndarray
        Transmission array of shape (n_energies, n_detectors_requested), in
        the same order as `detectors`.
    """
    root = Path(__file__).parent.parent
    grid_info = Path(root, *["config", "data", "grid"])
 
    column_names = ["sc", "p", "o", "phase", "slit", "grad", "rms", "thick", "bwidth", "bpitch"]
    front = Table.read(grid_info / "grid_param_front.txt", format="ascii", names=column_names)
    rear = Table.read(grid_info / "grid_param_rear.txt", format="ascii", names=column_names)
 
    nominal_transmission = Table.read(
        grid_info / "nom_grid_transmission.txt", format="ascii.no_header", comment="[;~]"
    )["col1"]
 
    # Current calibration table (post-March-2026 update): 6 columns, of which
    # IDL uses subc_n, subc_label, intercept, slope[1/deg] (skipping the two
    # error columns). This replaces the older CFL_subcoll_transmission.txt,
    # which carried stale coefficients for the finest grids specifically.
    calib_column_names = [
        "subc_n", "subc_label", "intercept", "intercept_error",
        "slope[1/deg]", "slope_error[1/deg]",
    ]
    calib = Table.read(
        grid_info / "stix_subcoll_transmission_10_15keV.csv",
        format="ascii.csv",
        header_start=0,
        data_start=1,
        names=calib_column_names,
    )
    subc_n_all = np.asarray(calib["subc_n"])
    intercept_all = np.asarray(calib["intercept"])
    slope_all = np.asarray(calib["slope[1/deg]"])
 
    ph_energy = np.atleast_1d(ph_energy)
    L = _tungsten_path_length(ph_energy)  # mm, shape (n_energies,)
    n_energies = len(L)
 
    if flare_location is not None:
        flare_loc_deg = np.asarray(flare_location, dtype=float) / 3600.0  # arcsec -> deg
    else:
        flare_loc_deg = np.array([0.0, 0.0])
 
    detectors = np.atleast_1d(detectors)
    n_det = len(detectors)
    subc_transm = np.zeros((n_energies, n_det))
 
    for j, det in enumerate(detectors):
        sc_num = det + 1  # convert 0-indexed detector -> 1-indexed subcollimator
 
        if det in CFL_BKG_0:
            # No physical grid: report the flat nominal value, same as before.
            subc_transm[:, j] = nominal_transmission[det]
            continue
 
        idx_front = np.where(front["sc"] == sc_num)[0]
        idx_rear = np.where(rear["sc"] == sc_num)[0]
 
        if det in FINEST_ALL_0:
            grid_orient_front = np.mean(front["o"][idx_front])
            grid_orient_rear = np.mean(rear["o"][idx_rear])
            pitch_front_raw = np.mean(front["p"][idx_front])
            pitch_rear_raw = np.mean(rear["p"][idx_rear])
 
            if det in FINEST_2ABC_0:
                pitch_front = pitch_front_raw / 2.0
                pitch_rear = pitch_rear_raw / 2.0
                thickness_front = 0.2
                thickness_rear = 0.2
            else:  # FINEST_1ABC_0
                pitch_front = pitch_front_raw / 3.0
                pitch_rear = pitch_rear_raw / 3.0
                thickness_front = 0.133
                thickness_rear = 0.133
 
            ds, dh = 0.0, 0.0
        else:
            grid_orient_front = front["o"][idx_front][0]
            grid_orient_rear = rear["o"][idx_rear][0]
            pitch_front = front["p"][idx_front][0]
            pitch_rear = rear["p"][idx_rear][0]
            thickness_front = front["thick"][idx_front][0]
            thickness_rear = rear["thick"][idx_rear][0]
            ds, dh = 5e-3, 5e-2
 
        grid_orient_avg = (grid_orient_front + grid_orient_rear) / 2.0
        theta = flare_loc_deg[0] * np.cos(np.deg2rad(grid_orient_avg)) + \
            flare_loc_deg[1] * np.sin(np.deg2rad(grid_orient_avg))
 
        idx_calib = np.where(subc_n_all == sc_num)[0]
        if len(idx_calib) == 0:
            raise ValueError(f"No calibration entry found for subcollimator {sc_num}.")
        intercept = intercept_all[idx_calib][0]
        slope = slope_all[idx_calib][0]
 
        subc_transm_low_e = intercept + slope * theta
        if subc_transm_low_e <= 0:
            raise ValueError(
                f"Transmission value for subcollimator {sc_num} is <= 0; "
                "check the provided flare location."
            )
 
        slit_to_pitch = np.sqrt(subc_transm_low_e)
        slit_front = slit_to_pitch * pitch_front
        slit_rear = slit_to_pitch * pitch_rear
 
        transm_front = stx_grid_transmission(pitch_front, slit_front, thickness_front, L, ds=ds, dh=dh)
        transm_rear = stx_grid_transmission(pitch_rear, slit_rear, thickness_rear, L, ds=ds, dh=dh)
 
        subc_transm[:, j] = transm_front * transm_rear
 
    return subc_transm
 
