from typing import Optional
from pathlib import Path
from collections.abc import Sequence

import astropy.units as apu
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table.table import Table
from astropy.units import Quantity
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.time import TimeRange
from xrayvision.transform import generate_xy
from xrayvision.visibility import Visibilities

from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.utils.logging import get_logger

logger = get_logger(__name__, level="DEBUG")


def get_transmission_matrix(u, v, shape=[64, 64] * apu.pix, pixel_size=[4.0, 4.0] * apu.arcsec, *, center, pixel_sum):
    r"""

    Parameters
    ----------
    vis :
        Visibly object containing u, v sampling
    shape :
        Shape of the images
    pixel_size
        Size of the pixels

    Returns
    -------
    The transmission matrix
    """
    fact = 1.0
    if pixel_sum == 4:
        fact = 2.0
    effective_area = 1.0

    # Computation of the constant factors M0 and M1
    M0 = effective_area / 4.0
    M1 = effective_area * fact * 4.0 / (np.pi**3) * np.sin(np.pi / (4.0 * fact))

    # Apply pixel correction
    phase_cor = np.full(32, 46.1) * apu.deg  # Center of large pixel of phase of morie pattern
    # TODO add correction for small pixel

    # Grid correction factors
    grid_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "GridCorrection.csv"
    grid_cal = Table.read(grid_cal_file, header_start=2, data_start=3)
    grid_corr = grid_cal["Phase correction factor"] * apu.deg

    # Apply grid correction
    phase_cor += grid_corr

    # Phase correction
    phase_cal_file = Path(__file__).parent.parent / "config" / "data" / "grid" / "PhaseCorrFactors.csv"
    phase_cal = Table.read(phase_cal_file, header_start=3, data_start=4)
    phase_corr = phase_cal["Phase correction factor"] * apu.deg

    # Apply grid correction
    phase_cor += phase_corr

    idx = [6, 28, 0, 24, 4, 22, 5, 29, 1, 14, 26, 30, 23, 7, 27, 20, 25, 3, 15, 13, 31, 2, 19, 21]
    phase_cor = phase_cor[idx]

    x = generate_xy(shape[0], pixel_size=pixel_size[0], phase_center=center[0])
    y = generate_xy(shape[1], pixel_size=pixel_size[1], phase_center=center[1])

    x, y = np.meshgrid(x, y)
    # Check apu are correct for exp need to be dimensionless and then remove apu for speed
    if (u[0] * x[0, 0]).unit == apu.dimensionless_unscaled and (v[0] * y[0, 0]).unit == apu.dimensionless_unscaled:
        u = u.value
        v = v.value
        x = x.value
        y = y.value

        phase = 2 * np.pi * (
            x[..., np.newaxis] * u[np.newaxis, :] + y[..., np.newaxis] * v[np.newaxis, :]
        ) - phase_cor.to_value("rad")

        H = np.concatenate([-np.cos(phase), -np.sin(phase), np.cos(phase), np.sin(phase)], axis=-1)

        # Application of the correction factors (needed because the units of the flux are photons
        # s^-1 cm^-2 arcsec^-2)
        H = H * 2.0 * M1 + M0
        H = H * (pixel_size[0] * pixel_size[1])
        H = np.transpose(H, (2, 0, 1))
        return H.reshape(u.shape[0] * 4, -1)
    else:
        raise ValueError("Units dont match")


@apu.quantity_input
def em(
    countrates: dict,
    vis: Visibilities,
    *,
    shape: Quantity[apu.pix],
    pixel_size: Quantity[apu.arcsec / apu.pix],
    flare_location: SkyCoord,
    maxiter: int = 5000,
    tolerance: float = 0.001,
    idx: Optional[Sequence[int]] = None,
):
    r"""
    Count-based expectation maximisation imaging algorithm.

    Parameters
    ----------
    countrates : `dict`
        Count rate data
    vis :
        Visibilities use for meta data
    shape : `Quantity[apu.pix]`
        Shape of the image
    pixel_size : `Quantity[apu.arcsec/apu.pix]`
        Size of the pixels
    maxiter : `int`
        Maximum number of iterations.
    tolerance : `float`
        Tolerance for convergence
    flare_location : `SkyCoord`
        Center of image
    idx : `Sequence[int]`
        The indices to use defaults to 10-2
    Returns
    -------

    """

    # temp
    if idx is not None:
        idx = [6, 28, 0, 24, 4, 22, 5, 29, 1, 14, 26, 30, 23, 7, 27, 20, 25, 3, 15, 13, 31, 2, 19, 21]
    ii = np.array([np.argwhere(vis.meta["isc"] - 1 == i).ravel() for i in idx]).ravel()

    tr = TimeRange(vis.meta.time_range)

    if not isinstance(flare_location, STIXImaging) and flare_location.obstime != tr.center:
        roll, solo_heeq, stix_pointing = get_hpc_info(vis.meta.time_range[0], vis.meta.time_range[1])
        solo_coord = HeliographicStonyhurst(solo_heeq, representation_type="cartesian", obstime=tr.center)
        flare_location = flare_location.transform_to(STIXImaging(obstime=tr.center, observer=solo_coord))

    flare_xy = np.array([flare_location.Tx.value, flare_location.Ty.value]) * apu.arcsec

    H = get_transmission_matrix(vis.u[ii], vis.v[ii], shape=shape, pixel_size=pixel_size, center=flare_xy, pixel_sum=1)
    y = countrates[idx, ...]
    # Not quite sure why this is needed clearly related to ordering of counts and H matrix
    y = y.flatten(order="F")
    # EXPECTATION MAXIMIZATION ALGORITHM

    # Initialization
    x = np.ones_like(H[0, :].value) * countrates.unit / apu.arcsec**2  # add arcsec**2
    y_index = y > 0
    Ht1 = np.matmul(np.ones_like(y), H.reshape(96, -1))
    H2 = H**2

    # Loop of the algorithm
    for i in range(maxiter):
        Hx = np.matmul(H, x)
        # TODO support possibility to subtract background z = y / Hx + b
        z = y / Hx
        Hz = np.matmul(z, H)

        x = x * (Hz / Ht1).T

        cstat = (2 / y[y_index].size) * (
            y[y_index].value * (np.log((y[y_index] / Hx[y_index]).value)) + Hx[y_index].value - y[y_index].value
        ).sum()

        # Stopping rule
        if i > 10 and (i % 25) == 0:
            emp_back_res = ((x.value * (Ht1.value - Hz.value)) ** 2).sum()
            std_back_res = (x**2 * np.matmul(1 / Hx, H2)).sum()
            std_index = emp_back_res / std_back_res

            logger.info(f"Iteration: {i}, StdDeV: {std_index}, C-stat: {cstat}")

            if std_index < tolerance:
                break
    m, n = shape.to_value("pix").astype(int)
    # Note the order keyword here to match `y.flatten(order='F')` above at line 108
    x_im = x.reshape(m, n, order="F")
    return x_im.T
