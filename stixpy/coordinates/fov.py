r"""
STIX field of view (FOV) and detector illumination.

The front and rear windows limit how much of each detector a source illuminates. The limits follow the nominal
geometry in Appendix C of Massa et al. (2023). As in `~stixpy.coordinates.frames.STIXImaging`, the four pixel columns
(A-D) lie along STIX X and the top (0-3) and bottom (4-7) pixel rows are separated along STIX Y, with +Y towards the
top row.

References
----------
Massa et al., 2023, The STIX Imaging Concept, Solar Physics, 298, https://doi.org/10.1007/s11207-023-02205-7
"""

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from stixpy.coordinates.frames import STIXImaging

__all__ = ["FOV_LIMITS", "get_illumination_status"]

_DISTANCE = 597 * u.mm  # front window to detector, d_sep + d_det (nominal 550 mm + 47 mm)
_WINDOW_X, _WINDOW_Y = 22 * u.mm, 20 * u.mm  # front window size along STIX X (columns) and Y (rows)
_DETECTOR_X, _DETECTOR_Y = 8.8 * u.mm, 9.2 * u.mm  # detector size along STIX X and Y


def _limit(offset):
    return np.arctan(offset / _DISTANCE).to(u.arcsec)


#: Half-widths (X, Y) of the STIX FOV boxes, in the STIX imaging frame.
FOV_LIMITS = {
    # all pixels fully illuminated
    "full": (_limit(_WINDOW_X / 2 - _DETECTOR_X / 2), _limit(_WINDOW_Y / 2 - _DETECTOR_Y / 2)),
    # one pixel row (top or bottom) partially or fully shadowed
    "single row": (_limit(_WINDOW_X / 2 - _DETECTOR_X / 2), _limit(_WINDOW_Y / 2)),
    # one row and two adjacent columns still illuminated (the maximal FOV of Massa et al. 2023)
    "partial": (_limit(_WINDOW_X / 2), _limit(_WINDOW_Y / 2)),
}


def get_illumination_status(coord):
    r"""
    Return the detector illumination status of a source and the pixels to use for it.

    Parameters
    ----------
    coord : `astropy.coordinates.SkyCoord`
        Source location(s). A coordinate not in the `~stixpy.coordinates.frames.STIXImaging` frame is transformed to
        it at the coordinate's ``obstime``.

    Returns
    -------
    status : `numpy.ndarray` of `str`
        One status per coordinate:

        * ``"full"``: all pixels fully illuminated, top and bottom rows see the same flux.
        * ``"single row"``: the top or bottom row is shadowed, use only the other row.
        * ``"partial"``: as ``"single row"``, and the outer pixel columns are partially shadowed.
        * ``"outside"``: outside the partial (maximal) FOV.
    pixels : `numpy.ndarray`
        The pixels to use, as accepted by the ``pixels`` argument of
        `~stixpy.calibration.visibility.create_meta_pixels`: ``"top+bot"`` for ``"full"``, ``"bot"`` for a source
        at +Y (top row shadowed), ``"top"`` for a source at -Y (bottom row shadowed) and `None` for ``"outside"``.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from stixpy.coordinates.frames import STIXImaging
    >>> from stixpy.coordinates.fov import get_illumination_status
    >>> coord = SkyCoord([0, 0, 3000, 0] * u.arcsec, [0, 2500, -3000, 4000] * u.arcsec, frame=STIXImaging)
    >>> status, pixels = get_illumination_status(coord)
    >>> status
    array(['full', 'single row', 'partial', 'outside'], dtype='<U10')
    >>> pixels
    array(['top+bot', 'bot', 'top', None], dtype=object)
    """
    if not isinstance(coord, SkyCoord):
        raise ValueError("coord must be a SkyCoord.")
    if not isinstance(coord.frame, STIXImaging):
        if coord.obstime is None:
            raise ValueError("coord must have an obstime to be transformed to the STIXImaging frame.")
        coord = coord.transform_to(STIXImaging(obstime=coord.obstime))
    x, y = np.abs(coord.Tx.to(u.arcsec)), coord.Ty.to(u.arcsec)
    inside = [(x <= lx) & (np.abs(y) <= ly) for lx, ly in FOV_LIMITS.values()]
    status = np.select(inside, list(FOV_LIMITS.keys()), default="outside")
    # the window shadow moves opposite to the source, so a source at +Y shadows the top row (+Y side)
    pixels = np.select([status == "full", status == "outside", y > 0], ["top+bot", None, "bot"], default="top")
    return status, pixels
