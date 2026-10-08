"""
Spectral response matrix (SRM) for STIX science data.
"""

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from stixpy.calibration.detector import tailing_matrix
from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.transmission import Transmission
from stixpy.config.instrument import RCR_AREA, STIX_INSTRUMENT

__all__ = ["SRM", "get_srm"]

_CONFIG_DATA = Path(__file__).parent.parent / "config" / "data"
DRM_PATH = _CONFIG_DATA / "detector" / "stx_detector_response_matrix.fits.gz"
BKG_GRID_TRANSMISSION_PATH = _CONFIG_DATA / "grid" / "real_bkg_grid_transmission.txt"

#: Tolerance in keV for matching count edges to the DRM grid, as IDL's ``get_uniq(..., epsilon=1e-4)``.
_EDGE_TOLERANCE = 1e-4

_CFL_DETECTOR = 8
_BKG_DETECTOR = 9


class SRM(NamedTuple):
    """
    A STIX spectral response matrix.
    """

    #: Response, (photon bin, count bin), in ct / (keV ph).
    srm: u.Quantity
    #: Photon bin edges.
    photon_edges: u.Quantity
    #: Count bin edges the response is for.
    count_edges: u.Quantity
    #: Geometric area the response is normalised to.
    geometric_area: u.Quantity


@lru_cache
def _read_drm():
    """
    Read the saved DRM, its bin edges, the STIX count edges and the CdTe cross sections, in keV and 1/cm.
    """
    drm = Table.read(DRM_PATH, hdu=1)["DRM"].quantity.to_value(u.ct / (u.keV * u.ph))
    ph_edges = Table.read(DRM_PATH, hdu=2)["DRM"].quantity.to_value(u.keV)
    ct_edges = Table.read(DRM_PATH, hdu=3)["DRM"].quantity.to_value(u.keV)
    xsec = Table.read(DRM_PATH, hdu=4)
    xsec_energy = xsec["ENERGY"].quantity.to_value(u.keV)
    xsec_value = xsec["XSEC"].quantity.to_value(1 / u.cm)
    arrays = (drm, ph_edges, ct_edges, xsec_energy, xsec_value)
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _match_idl_grid(drm, ph_full, ct_full, e_edges, epsilon=_EDGE_TOLERANCE):
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


def _count_edges_in_drm(count_edges, ct_full, epsilon=_EDGE_TOLERANCE):
    """
    Keep the count edges the DRM covers.

    STIX science channels run from 0 keV to an open top channel above 150 keV, which the DRM
    doesn't cover: its count edges run from 4 to 150 keV. Edges outside that range, or not finite,
    are dropped, so the bottom and open top channels are left out of the response.
    """
    count_edges = np.asarray(count_edges, dtype=float)
    finite = count_edges[np.isfinite(count_edges)]
    if np.any(np.diff(finite) <= 0):
        raise ValueError("count_edges must be increasing.")

    in_drm = (finite >= ct_full[0] - epsilon) & (finite <= ct_full[-1] + epsilon)
    e_edges = finite[in_drm]
    if e_edges.size < 2:
        raise ValueError(
            f"count_edges must contain at least two edges between {ct_full[0]} and {ct_full[-1]} keV, "
            f"the range the DRM covers."
        )
    return e_edges


@u.quantity_input
def get_srm(
    count_edges: u.Quantity[u.keV],
    detectors,
    pixels,
    rcr,
    flare_location: SkyCoord | u.Quantity | None = None,
    ph_energy_min: u.Quantity[u.keV] | None = 3.5 * u.keV,
):
    """
    Build the spectral response matrix (SRM) for a set of detectors and pixels.

    The detector response matrix (DRM) is read from the calibration file
    ``stx_detector_response_matrix.fits.gz``: ``drm.smatrix`` from STIX-GSW
    ``stx_build_drm``, without hole tailing, on a photon grid made of the STIX
    transmission-file grid plus all 31 STIX count edges, together with its bin
    edges, the count edges and the CdTe cross sections needed for the tailing.

    The steps follow STIX-GSW ``stx_build_pixel_drm``:

    1. Count edges outside the 4-150 keV range the DRM covers are dropped, which
       leaves out the 0-4 keV and open top channels of the science data.
    2. The DRM is put on the photon grid IDL builds for these edges (see
       ``_match_idl_grid``).
    3. The fine count bins are summed into the requested count bins.
    4. Hole tailing is applied along the photon axis on this grid (see
       `stixpy.calibration.detector.tailing_matrix`).
    5. Each photon row is multiplied by the detector transmission at the bin
       centre, averaged over the detectors, with the attenuator in if `rcr` is
       not 0, and by the grid transmission for `flare_location`, averaged over
       the detectors. For the background detector (index 9) on its own, the
       grid transmission is the mean over the selected pixels from
       ``real_bkg_grid_transmission.txt``.
    6. The result is divided by the count bin widths.
    7. Photon bins starting below `ph_energy_min` are dropped.

    Parameters
    ----------
    count_edges : `~astropy.units.Quantity`
        Count (science channel) bin edges, e.g.
        ``np.hstack([energies["e_low"], energies["e_high"][-1]])``. Every edge
        between 4 and 150 keV must be a STIX science channel edge.
    detectors : int or array_like
        0-indexed detectors the response is for.
    pixels : int or array_like
        0-indexed pixels the response is for.
    rcr : int
        Rate control regime (RCR) state, 0-7. Sets whether the attenuator is in
        and the fraction of the pixel area that is active (see
        ``stixpy.config.instrument.RCR_AREA``).
    flare_location : `~astropy.coordinates.SkyCoord` or `~astropy.units.Quantity`, optional
        Flare position, passed to `~stixpy.calibration.grid.get_grid_transmission`.
        If None, on-axis is assumed.
    ph_energy_min : `~astropy.units.Quantity`, optional
        Photon bins with a lower edge below this are dropped. None keeps every
        photon bin. Default 3.5 keV.

    Returns
    -------
    `SRM`
        The response in ct / (keV ph), one row per photon bin and one column per
        count bin, with the photon and count edges and the geometric area: the
        number of detectors times the area of the selected pixels, scaled by the
        fraction of the pixel area active in this RCR state.

    Raises
    ------
    ValueError
        If the CFL detector (index 8) is selected, if the BKG detector (index 9)
        is combined with other detectors, if `rcr` is not 0-7, or if a count edge
        is not on the DRM grid.
    """
    detectors = np.atleast_1d(detectors)
    pixels = np.atleast_1d(pixels)
    rcr = int(np.asarray(rcr).item())  # callers may pass a 1-element array

    # As in STIX-GSW stx_convert_spectrogram2ospex: the CFL has no grid transmission,
    # and the BKG detector has its own, so it can only be used on its own.
    if np.isin(_CFL_DETECTOR, detectors):
        raise ValueError("The CFL detector (index 8) can not be selected for spectral fitting.")
    bkg_only = detectors.size == 1 and detectors[0] == _BKG_DETECTOR
    if np.isin(_BKG_DETECTOR, detectors) and not bkg_only:
        raise ValueError(
            "The BKG detector (index 9) can not be selected together with imaging detectors for spectral fitting."
        )
    if not 0 <= rcr < len(RCR_AREA):
        raise ValueError(f"rcr must be between 0 and {len(RCR_AREA) - 1}, not {rcr}.")

    drm_full, ph_full, ct_full, xsec_energy, xsec = _read_drm()
    e_edges = _count_edges_in_drm(count_edges.to_value(u.keV), ct_full)

    drm, ph_edges = _match_idl_grid(drm_full, ph_full, ct_full, e_edges)

    # Sum the fine count bins into the requested count bins; each one is a run of whole fine bins.
    i_edges = np.searchsorted(ph_edges, e_edges - _EDGE_TOLERANCE)
    drm = np.add.reduceat(drm[:, i_edges[0] : i_edges[-1]], i_edges[:-1] - i_edges[0], axis=1)

    tailing = tailing_matrix(ph_edges * u.keV, xsec_energy * u.keV, xsec / u.cm)
    drm = tailing.T @ drm

    ph_mids = (ph_edges[:-1] + ph_edges[1:]) / 2 * u.keV
    transmission = Transmission().get_transmission(energies=ph_mids, attenuator=rcr > 0)
    attenuation = np.mean([np.asarray(transmission[f"det-{det}"]) for det in detectors], axis=0)

    if bkg_only:
        bkg_transmission = Table.read(BKG_GRID_TRANSMISSION_PATH, format="ascii.no_header", comment="[;~]")["col1"]
        grid_transmission = np.full(ph_mids.size, np.nanmean(bkg_transmission[pixels]))
    else:
        grid_transmission = get_grid_transmission(ph_mids, detectors, flare_location).mean(axis=1)

    srm = drm * (attenuation * grid_transmission)[:, None] / np.diff(e_edges)[None, :]

    if ph_energy_min is not None:
        i_min = np.searchsorted(ph_edges[:-1], ph_energy_min.to_value(u.keV))
        srm = srm[i_min:]
        ph_edges = ph_edges[i_min:]

    pixel_areas = STIX_INSTRUMENT.pixel_config["Area"].quantity.to(u.cm**2)
    geometric_area = detectors.size * pixel_areas[pixels].sum() * RCR_AREA[rcr] / pixel_areas.sum()

    return SRM(
        srm=srm * u.ct / (u.keV * u.ph),
        photon_edges=ph_edges * u.keV,
        count_edges=e_edges * u.keV,
        geometric_area=geometric_area,
    )
