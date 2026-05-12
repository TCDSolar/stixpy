import warnings
from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.coordinates._ephemeris_fetcher import fetch_ephemeris_for_range
from stixpy.coordinates.frames import STIXImaging
from stixpy.product.product_factory import Product
from stixpy.product.sources.anc import Ephemeris
from stixpy.utils.logging import get_logger
from stixpy.utils.table_lru import TableLRUCache

STIX_X_SHIFT = 26.1 * u.arcsec  # fall back to this when non sas solution available
STIX_Y_SHIFT = 58.2 * u.arcsec  # fall back to this when non sas solution available
STIX_X_OFFSET = 60.0 * u.arcsec  # remaining offset after SAS solution
STIX_Y_OFFSET = 8.0 * u.arcsec  # remaining offset after SAS solution

# Bracket pad applied around cache range queries so np.interp doesn't clamp
# at endpoints. Intentionally not a clean 64 s multiple per the ephemeris doc.
EDGE_PAD = 80 * u.s

logger = get_logger(__name__)

__all__ = ["get_hpc_info", "stixim_to_hpc", "hpc_to_stixim", "STIX_EPHEMERIS_CACHE", "load_ephemeris_fits_to_cache"]

# Create a global cache for STIX ephemeris data
STIX_EPHEMERIS_CACHE = TableLRUCache("STIX_EPHEMERIS_CACHE", maxsize=300000, nominal_bin=64 * u.s)


def load_ephemeris_fits_to_cache(anc_file):
    """
    Load ephemeris data from a fits file into the ephemeris cache.
    """
    logger.info(f"Loading STIX ephemeris data from {anc_file}. into cache")
    anc_file = Path(anc_file)
    try:
        anc = Product(anc_file, data_only=True)
        if isinstance(anc, Ephemeris):
            STIX_EPHEMERIS_CACHE.put(anc.data, source=anc_file.name)
    except (OSError, ValueError) as e:
        logger.error(f"Error loading STIX ephemeris data from {anc_file}: {e}")


def _get_rotation_matrix_and_position(obstime, obstime_end=None):
    r"""
    Return rotation matrix STIX Imaging to SOLO HPC and position of SOLO in HEEQ.

    Parameters
    ----------
    obstime : `astropy.time.Time`
        Time
    Returns
    -------
    """
    roll, solo_position_heeq, spacecraft_pointing = get_hpc_info(obstime, obstime_end)

    # Generate the rotation matrix using the x-convention (see Goldstein)
    # Rotate +90 clockwise around the first axis
    # STIX is mounted on the -Y panel (SOLO_SRF) need to rotate to from STIX frame to SOLO_SRF
    rot_to_solo = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    logger.debug("Pointing: %s.", spacecraft_pointing)
    A = rotation_matrix(-1 * roll, "x")
    B = rotation_matrix(spacecraft_pointing[..., 1], "y")
    C = rotation_matrix(-1 * spacecraft_pointing[..., 0], "z")

    # Will be applied right to left
    rmatrix = A @ B @ C @ rot_to_solo

    return rmatrix, solo_position_heeq


def get_hpc_info(times, end_time=None, *, return_source=False):
    r"""
    Get STIX pointing and SO location from ANC aspect files.

    Parameters
    ----------
    times : `astropy.time.Time`
        Time, array of times, or start of a time interval.
    end_time : `astropy.time.Time`, optional
        End of a time interval; combined with a scalar ``times`` selects the
        range-averaging mode.
    return_source : list, optional
        If a list is passed, it is extended with the set of ANC FITS
        filenames that contributed to the result.

    Returns
    -------
    roll, solo_heeq, stix_pointing
        SOLO roll, SOLO HEEQ position, and the STIX pointing (best of the
        spacecraft and SAS solutions).
    """
    # Explicit dispatch: range mode is "scalar start + scalar end_time".
    # An array of times (with or without end_time) always uses interpolation.
    is_range = end_time is not None and times.isscalar

    aux = _get_ephemeris_data(times, end_time=end_time)

    if is_range:
        in_window = (aux["time"] >= times) & (aux["time"] <= end_time)
        window = aux[in_window]

        if len(window) >= 2:
            # Average across [times, end_time]
            roll, pitch, yaw = np.mean(window["roll_angle_rpy"], axis=0)
            solo_heeq = np.mean(window["solo_loc_heeq_zxy"], axis=0)

            good_sas = window[window["sas_ok"] == 1]
            if len(good_sas) == 0:
                warnings.warn(f"No good SAS solution found for time range: {times} to {end_time}.")
                sas_x = 0
                sas_y = 0
            else:
                sas_x = np.mean(good_sas["y_srf"])
                sas_y = np.mean(good_sas["z_srf"])
                sigma_x = np.std(good_sas["y_srf"])
                sigma_y = np.std(good_sas["z_srf"])
                tolerance = 3 * u.arcsec
                if sigma_x > tolerance or sigma_y > tolerance:
                    warnings.warn(f"Pointing unstable: StD(X) = {sigma_x}, StD(Y) = {sigma_y}.")
        else:
            # Fewer than 2 rows in [start, end]: degrade to a midpoint interpolation
            midpoint = times + (end_time - times) * 0.5
            roll, pitch, yaw, solo_heeq, sas_x, sas_y, good_sas = _interpolate_aux(midpoint, aux)
    else:
        # Point / array mode: interpolate each requested time
        roll, pitch, yaw, solo_heeq, sas_x, sas_y, good_sas = _interpolate_aux(times, aux)

    # Convert the spacecraft pointing to STIX frame
    rotated_yaw = -yaw * np.cos(roll) + pitch * np.sin(roll)
    rotated_pitch = yaw * np.sin(roll) + pitch * np.cos(roll)
    spacecraft_pointing = np.vstack([STIX_X_SHIFT + rotated_yaw, STIX_Y_SHIFT + rotated_pitch]).T
    stix_pointing = spacecraft_pointing
    sas_pointing = np.vstack([sas_x + STIX_X_OFFSET, -1 * sas_y + STIX_Y_OFFSET]).T

    pointing_diff = np.sqrt(np.sum((spacecraft_pointing - sas_pointing) ** 2, axis=0))
    if np.all(np.isfinite(sas_pointing)) and len(good_sas) > 0:
        if np.max(pointing_diff) < 200 * u.arcsec:
            logger.info(f"Using SAS pointing: {sas_pointing}")
            stix_pointing = np.where(pointing_diff < 200 * u.arcsec, sas_pointing, spacecraft_pointing)
        else:
            warnings.warn(
                f"Using spacecraft pointing: {spacecraft_pointing} large difference between SAS and spacecraft."
            )
    else:
        warnings.warn(f"SAS solution not available using spacecraft pointing: {stix_pointing}.")

    if is_range or times.size == 1:
        solo_heeq = solo_heeq.squeeze()
        stix_pointing = stix_pointing.squeeze()

    if isinstance(return_source, list) and "__source" in aux.colnames:
        return_source.extend(set(aux["__source"].value))

    return roll, solo_heeq, stix_pointing


def _interpolate_aux(at_times, aux):
    """
    Interpolate roll/pitch/yaw, SOLO HEEQ position, and SAS pointing from
    the cached ANC rows ``aux`` onto the requested ``at_times``.

    Returns ``(roll, pitch, yaw, solo_heeq, sas_x, sas_y, good_sas)``.
    """
    x = (at_times - aux["time"][0]).to_value(u.s)
    xp = (aux["time"] - aux["time"][0]).to_value(u.s)

    roll = np.interp(x, xp, aux["roll_angle_rpy"][:, 0].value) << aux["roll_angle_rpy"].unit
    pitch = np.interp(x, xp, aux["roll_angle_rpy"][:, 1].value) << aux["roll_angle_rpy"].unit
    yaw = np.interp(x, xp, aux["roll_angle_rpy"][:, 2].value) << aux["roll_angle_rpy"].unit

    solo_heeq = (
        np.vstack(
            [
                np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 0].value),
                np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 1].value),
                np.interp(x, xp, aux["solo_loc_heeq_zxy"][:, 2].value),
            ]
        ).T
        << aux["solo_loc_heeq_zxy"].unit
    )

    sas_x = np.interp(x, xp, aux["y_srf"].value) << aux["y_srf"].unit
    sas_y = np.interp(x, xp, aux["z_srf"].value) << aux["z_srf"].unit

    if np.ndim(x) == 0 or x.size == 1:
        good_sas = [True] if np.interp(x, xp, aux["sas_ok"]).astype(bool) else []
    else:
        sas_ok = np.interp(x, xp, aux["sas_ok"]).astype(bool)
        good_sas = sas_ok[sas_ok == True]  # noqa E712

    return roll, pitch, yaw, solo_heeq, sas_x, sas_y, good_sas


def _get_ephemeris_data(times, end_time=None, *, interpolate=True):
    r"""
    Return ANC ephemeris rows sufficient to interpolate or average over the
    requested time(s). Used by :func:`get_hpc_info`.

    Three input shapes are accepted and normalised to a single scalar
    range ``[qstart, qend]``:

    * scalar ``times`` (no ``end_time``)            → point at ``times``
    * scalar ``times`` + ``end_time``                → range ``[times, end_time]``
    * array ``times`` (with or without ``end_time``) → range covering the array

    Parameters
    ----------
    times : `astropy.time.Time`
        Scalar or array of times.
    end_time : `astropy.time.Time`, optional
        End of a time interval.
    interpolate : bool
        Legacy parameter, kept for back-compatibility; currently unused.

    Returns
    -------
    QTable
        Cache slice padded by ``EDGE_PAD`` on each side. Includes the
        ``__source`` column populated from the ANC filename(s).

    Raises
    ------
    ValueError
        If the requested range cannot be covered after fetching (real data
        gap larger than ``max_gap``).
    """
    if times.isscalar:
        qstart = times
        qend = end_time if end_time is not None else times
    else:
        qstart = times.min()
        qend = end_time if end_time is not None else times.max()

    logger.info(f"Resolving ANC ephemeris for {qstart} – {qend}")

    table = _query_cache(qstart, qend)
    if table is not None:
        return table

    fetched = fetch_ephemeris_for_range(qstart, qend)
    STIX_EPHEMERIS_CACHE.put(fetched, source=fetched["__source"].value)

    # _force=True so we still re-read the data we just put even if
    # bypass_cache_read is set (which only blocks initial reads).
    table = _query_cache(qstart, qend, _force=True)
    if table is None:
        raise ValueError(
            f"Ephemeris data for {qstart}–{qend} has gaps larger than "
            f"{STIX_EPHEMERIS_CACHE.max_gap} after fetch (real data gap)."
        )
    return table


def _query_cache(qstart, qend, *, _force=False):
    """
    Translate a normalised (qstart, qend) into a single cache call.

    Point queries (qstart == qend) widen the cache search window by
    ``EDGE_PAD`` so the cache can find bracketing rows for downstream
    interpolation. Range queries keep coverage strict on ``[qstart, qend]``
    and pad only the returned slice.
    """
    if qstart == qend:
        return STIX_EPHEMERIS_CACHE.get_range(qstart - EDGE_PAD, qend + EDGE_PAD, pad=0 * u.s, _force=_force)
    return STIX_EPHEMERIS_CACHE.get_range(qstart, qend, pad=EDGE_PAD, _force=_force)


@frame_transform_graph.transform(coord.FunctionTransform, STIXImaging, Helioprojective)
def stixim_to_hpc(stxcoord, hpcframe):
    r"""
    Transform STIX Imaging coordinates to given HPC frame
    """
    logger.debug("STIX: %s", stxcoord)

    rot_matrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxcoord.obstime, stxcoord.obstime_end)

    obstime = stxcoord.obstime
    if stxcoord.obstime_end is not None:
        obstime = stxcoord.obstime + (stxcoord.obstime_end - stxcoord.obstime) / 2

    solo_heeq = HeliographicStonyhurst(solo_pos_heeq.T, representation_type="cartesian", obstime=obstime)

    # Transform from STIX imaging to SOLO HPC
    newrepr = stxcoord.cartesian.transform(rot_matrix)

    # Create SOLO HPC
    solo_hpc = Helioprojective(
        newrepr,
        obstime=obstime,
        observer=solo_heeq.transform_to(HeliographicStonyhurst(obstime=obstime)),
    )
    logger.debug("SOLO HPC: %s", solo_hpc)

    # Transform from SOLO HPC to input HPC
    if hpcframe.observer is None:
        hpcframe = type(hpcframe)(obstime=hpcframe.obstime, observer=solo_heeq)
    hpc = solo_hpc.transform_to(hpcframe)
    logger.debug("Target HPC: %s", hpc)
    return hpc


@frame_transform_graph.transform(coord.FunctionTransform, Helioprojective, STIXImaging)
def hpc_to_stixim(hpccoord, stxframe):
    r"""
    Transform HPC coordinate to STIX Imaging frame.
    """
    logger.debug("Input HPC: %s", hpccoord)
    rmatrix, solo_pos_heeq = _get_rotation_matrix_and_position(stxframe.obstime, stxframe.obstime_end)

    obstime = stxframe.obstime
    if stxframe.obstime_end is not None:
        obstime = stxframe.obstime + (stxframe.obstime_end - stxframe.obstime) / 2
    solo_hgs = HeliographicStonyhurst(solo_pos_heeq.T, representation_type="cartesian", obstime=obstime)

    # Create SOLO HPC
    solo_hpc_frame = Helioprojective(obstime=obstime, observer=solo_hgs)
    # Transform input to SOLO HPC
    solo_hpc_coord = hpccoord.transform_to(solo_hpc_frame)
    logger.debug("SOLO HPC: %s", solo_hpc_coord)

    # Transform from SOLO HPC to STIX imaging
    newrepr = solo_hpc_coord.cartesian.transform(matrix_transpose(rmatrix))
    stx_coord = stxframe.realize_frame(newrepr)
    logger.debug("STIX: %s", newrepr)
    return stx_coord
