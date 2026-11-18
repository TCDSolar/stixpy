"""
Spectral response matrix (SRM) for STIX science data.
"""

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.transmission import Transmission
from stixpy.config.instrument import RCR_AREA, STIX_INSTRUMENT

__all__ = ["SRM", "get_srm"]

_CONFIG_DATA = Path(__file__).parent.parent / "config" / "data"
DRM_PATH = _CONFIG_DATA / "detector" / "stix-drm-30ch.fits.gz"
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
    Read the IDL DRM in counts per photon and its energy edges in keV.
    """
    with fits.open(DRM_PATH) as hdus:
        arrays = tuple(np.array(hdus[key].data, dtype=float) for key in (0, "PHOTON_EDGES", "COUNT_EDGES"))
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _group_count_channels(drm, ct_full, e_edges, epsilon=_EDGE_TOLERANCE):
    """Sum whole science count channels while retaining every photon bin."""
    matches = np.isclose(e_edges[:, None], ct_full[None, :], atol=epsilon, rtol=0)
    missing = ~matches.any(axis=1)
    if missing.any():
        raise ValueError(f"Energy edges {e_edges[missing]} are not on the saved DRM grid.")

    indices = matches.argmax(axis=1)
    return np.add.reduceat(drm[:, indices[0] : indices[-1]], indices[:-1] - indices[0], axis=1)


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

    The detector response matrix (DRM) is read from ``stix-drm-30ch.fits.gz``.
    It contains the STIX-GSW v0.6.2 response with hole tailing applied in IDL
    on the full photon grid and the count axis integrated into the 30 finite
    science channels.

    Count edges outside 4–150 keV are dropped. Adjacent science channels are
    summed on the fixed photon grid.
    Detector and grid transmission are evaluated at the photon bin
    centres. For the background detector, grid transmission is averaged over
    the selected pixels. The response is divided by count bin widths and
    photon bins below `ph_energy_min` are dropped.

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

    drm_full, ph_edges, ct_full = _read_drm()
    e_edges = _count_edges_in_drm(count_edges.to_value(u.keV), ct_full)

    drm = _group_count_channels(drm_full, ct_full, e_edges)

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
