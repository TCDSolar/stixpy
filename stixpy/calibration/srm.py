from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.table import Table

from stixpy.calibration.detector import tailing_matrix
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.transmission import Transmission
from stixpy.config.instrument import STIX_INSTRUMENT

__all__ = ["get_srm"]


def _match_idl_grid(drm, ph_full, ct_full, e_edges, epsilon=1e-4):
    """
    Put the saved DRM on the photon grid IDL builds for this product.

    IDL's grid is ``get_uniq([transmission_grid, ct_edges], epsilon=1e-4)``. The saved grid is the
    transmission grid plus all STIX count edges (``ct_full``), so dropping the ones this product
    doesn't have gives IDL's grid exactly, and each new bin is a run of whole saved bins.

    Parameters
    ----------
    drm : numpy.ndarray
        Saved DRM, (photon, count), counts / keV / photon.
    ph_full : numpy.ndarray
        Bin edges of the saved DRM in keV (same for both axes).
    ct_full : numpy.ndarray
        STIX count edges the saved grid contains, in keV.
    e_edges : numpy.ndarray
        This product's count edges, in keV.

    Returns
    -------
    drm_new : numpy.ndarray
        DRM on IDL's grid in counts per count bin per photon: count bins summed, photon bins
        averaged (weighted by width).
    ph_edges : numpy.ndarray
        IDL's photon grid for this product.
    """
    missing = ~np.isclose(e_edges[:, None], ph_full[None, :], atol=epsilon).any(axis=1)
    if missing.any():
        raise ValueError(f"Energy edges {e_edges[missing]} are not on the saved DRM grid.")

    is_count_edge = np.isclose(ph_full[:, None], ct_full[None, :], atol=epsilon).any(axis=1)
    in_product = np.isclose(ph_full[:, None], e_edges[None, :], atol=epsilon).any(axis=1)

    keep = ~is_count_edge | in_product  # = IDL's get_uniq([transmission_grid, e_edges])
    ph_edges = ph_full[keep]
    starts = np.flatnonzero(keep)[:-1]  # first saved bin in each new bin

    widths = np.diff(ph_full)
    summed = np.add.reduceat(drm * widths[:, None] * widths[None, :], starts, axis=0)
    summed = np.add.reduceat(summed, starts, axis=1)

    return summed / np.diff(ph_edges)[:, None], ph_edges


def get_srm(energies, flare_location, detector_indices_input, pixel_indices_input, rcr, srm_e_min=3.5 * u.keV):
    """
    Build the spectral response matrix (SRM) for a set of detectors and pixels.

    The detector response matrix (DRM) is read from the calibration file
    ``stx_detector_response_matrix.fits.gz``: ``drm.smatrix`` from STIX-GSW
    ``stx_build_drm``, without hole tailing, on a photon grid made of the STIX
    transmission-file grid plus all 31 STIX count edges, together with its bin
    edges, the count edges and the CdTe cross sections needed for the tailing.

    The steps follow STIX-GSW ``stx_build_pixel_drm``:

    1. The count edges are taken from the product's energy table, without the
       0 keV lower edge and the open top bin.
    2. The DRM is put on the photon grid IDL builds for these edges (see
       `_match_idl_grid`).
    3. The fine count bins are summed into the product's count bins.
    4. Hole tailing is applied along the photon axis on this grid (see
       `stixpy.calibration.detector.tailing_matrix`).
    5. Each photon row is multiplied by the detector transmission at the bin
       centre, averaged over the detectors, with the attenuator in if `rcr` is
       not 0, and by the grid transmission for `flare_location`, averaged over
       the detectors. For the background detector (index 9) on its own, the
       grid transmission is the mean over the selected pixels from
       ``real_bkg_grid_transmission.txt``.
    6. The result is divided by the count bin widths.

    Parameters
    ----------
    energies : astropy.table.Table
        The product's energy table, with ``e_low`` and ``e_high`` columns in keV.
    flare_location : array_like or None
        Flare position ``[Tx, Ty]`` in the STIX frame, passed to
        `get_grid_transmission`.
    detector_indices_input : int or array_like
        Detectors the response is for.
    pixel_indices_input : int or array_like
        Pixels the response is for.
    rcr : int or array_like
        RCR state, 0-7, as a number or a one-element array. Sets whether the
        attenuator is in and the fraction of the pixel area that is active.
    srm_e_min : astropy.units.Quantity, bool or None, optional
        Photon energies below this are cut from the SRM and the photon axis.
        True means 3.5 keV, and False or None keeps every photon energy.
        Default 3.5 keV.

    Returns
    -------
    dict
        With keys:

        - 'srm': the response matrix, one row per photon bin and one column per
          count bin, in counts per keV of count energy per photon.
        - 'ph_axis': photon bin edges in keV, as a plain array.
        - 'geo_area': geometric area in cm^2, the number of detectors times the
          area of the selected pixels, scaled by the fraction of the pixel area
          active in this RCR state.

    Raises
    ------
    ValueError
        If the CFL detector (index 8) is selected, if the BKG detector (index 9)
        is combined with other detectors, or if a count edge of the product is
        not on the DRM grid.
    """

    HERE = Path(__file__).parent
    ROOT = HERE.parent
    PATH_DRM = ROOT / "config" / "data" / "detector" / "stx_detector_response_matrix.fits.gz"
    PATH_BKG_TRANS = ROOT / "config" / "data" / "grid" / "real_bkg_grid_transmission.txt"

    drm = np.array(Table.read(PATH_DRM, hdu=1)["DRM"])
    ph_energies = np.array(Table.read(PATH_DRM, hdu=2)["DRM"])
    ct_energies = np.array(Table.read(PATH_DRM, hdu=3)["DRM"])
    xsec = Table.read(PATH_DRM, hdu=4)

    detector_indices_input = np.atleast_1d(detector_indices_input)
    pixel_indices_input = np.atleast_1d(pixel_indices_input)
    rcr = int(np.asarray(rcr).item())  # callers may pass a 1-element array

    # As in STIX-GSW stx_convert_spectrogram2ospex: the CFL (index 8) has no grid transmission,
    # and the BKG detector (index 9) has its own, so it can only be used on its own.
    if np.isin(8, detector_indices_input):
        raise ValueError("The CFL detector (index 8) can not be selected for spectral fitting.")
    if np.isin(9, detector_indices_input) and detector_indices_input.size > 1:
        raise ValueError(
            "The BKG detector (index 9) can not be selected together with imaging detectors for spectral fitting."
        )

    e_low = np.array(energies["e_low"])

    if e_low[0] == 0:
        e_low = e_low[1:]

    e_high = np.array(energies["e_high"])

    if e_high[-2] == 150:
        e_edges = e_low
        ct_e_diff = np.diff(e_edges)
    else:
        e_edges = np.concatenate([e_low, [e_high[-1]]])
        ct_e_diff = np.diff(e_edges)

    drm_clipped, ph_energies_clipped = _match_idl_grid(drm, ph_energies, ct_energies, e_edges)

    pixel_areas_full = STIX_INSTRUMENT.pixel_config["Area"].to("cm2")

    pixel_areas = pixel_areas_full[pixel_indices_input].value

    area_scale = np.size(detector_indices_input) * np.sum(pixel_areas)

    energy_widths = np.diff(ph_energies_clipped)

    e_mids = ph_energies_clipped[:-1] + (energy_widths / 2)

    trans = Transmission()

    if rcr == 0:
        tot_trans = trans.get_transmission(energies=e_mids * u.keV)
    else:
        tot_trans = trans.get_transmission(energies=e_mids * u.keV, attenuator=True)

    rcr_state_all = np.array([0.8096, 0.80961, 0.4048, 0.2024, 0.1012, 0.0396, 0.0198, 0.0099])
    pixel_indices_input_rcr = np.arange(0, 12, 1)

    rcr_state = rcr_state_all[int(rcr)]
    rcr_factor = rcr_state / np.sum(pixel_areas_full[pixel_indices_input_rcr].value)

    attenuation = np.mean([np.asarray(tot_trans[f"det-{det}"]) for det in detector_indices_input], axis=0)

    drm_new = []

    for j in range(np.shape(drm_clipped)[0]):
        working = []

        for i in range(len(e_edges) - 1):
            indices_sum = np.where((ph_energies_clipped >= e_edges[i]) & (ph_energies_clipped < e_edges[i + 1]))[0]

            tot = drm_clipped[j, indices_sum].sum(axis=0)

            working.append(tot)

        drm_new.append(working)

    drm_new = np.array(drm_new)

    tailing = tailing_matrix(
        ph_energies_clipped * u.keV, np.array(xsec["ENERGY"]) * u.keV, np.array(xsec["XSEC"]) / u.cm
    )
    drm_new = (tailing.T @ drm_new) * attenuation[:, None]

    grid_transmission = get_grid_transmission(e_mids * u.keV, detector_indices_input, flare_location)

    if detector_indices_input.size == 1 and detector_indices_input[0] == 9:
        bkg_transmission = Table.read(PATH_BKG_TRANS, format="ascii.no_header", comment="[;~]")["col1"]
        bkg_transmission_mean = np.nanmean(bkg_transmission[pixel_indices_input])
        grid_transmission = np.broadcast_to(bkg_transmission_mean, (np.shape(grid_transmission)[0], 1))

    grid_transmission = grid_transmission.mean(axis=1)

    srm = (drm_new * grid_transmission[:, None]) / ct_e_diff[None, :]

    if srm_e_min is True:
        srm_e_min = 3.5 * u.keV
    elif srm_e_min is False:
        srm_e_min = None

    if srm_e_min is not None:
        # ph_energies_clipped holds bin EDGES (one longer than the SRM's row
        # count), so find the cut on the lower edges and apply to both.
        lower = ph_energies_clipped[:-1] if ph_energies_clipped.size == srm.shape[0] + 1 else ph_energies_clipped
        i0 = int(np.searchsorted(lower, srm_e_min.value, side="left"))
        srm = srm[i0:]
        ph_energies_clipped = ph_energies_clipped[i0:]

    return {"srm": srm, "ph_axis": ph_energies_clipped, "geo_area": area_scale * rcr_factor}
