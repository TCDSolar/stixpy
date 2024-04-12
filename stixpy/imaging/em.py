from pathlib import Path

import numpy as np
from xrayvision.transform import generate_xy

import astropy.units as apu
from astropy.table.table import Table

from stixpy.calibration.grid import get_grid_transmission
from stixpy.calibration.visibility import get_visibility_info_giordano
from stixpy.utils.logging import get_logger

logger = get_logger(__name__, level='DEBUG')


def get_transmission_matrix(u, v, shape=[64, 64]*apu.pix,
                            pixel_size=[4.0, 4.0]*apu.arcsec, *, center, pixel_sum):
    r"""

    Parameters
    ----------
    vis :
        Visibly object containg u, v sampling
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
    M0 = effective_area / 4.
    M1 = effective_area * fact * 4. / (np.pi**3) * np.sin(np.pi / (4. * fact))

    # Apply pixel correction
    phase_cor = np.full(32, 46.1) * apu.deg  # Center of large pixel of phase of morie pattern
    # TODO add correction for small pixel

    # Grid correction factors
    grid_cal_file = Path(__file__).parent.parent / 'config' / 'data' / 'grid' / 'GridCorrection.csv'
    grid_cal = Table.read(grid_cal_file, header_start=2, data_start=3)
    grid_corr = grid_cal['Phase correction factor'] * apu.deg

    # Apply grid correction
    phase_cor += grid_corr

    # Phase correction
    phase_cal_file = Path(__file__).parent.parent / 'config' / 'data' / 'grid' / 'PhaseCorrFactors.csv'
    phase_cal = Table.read(phase_cal_file, header_start=3, data_start=4)
    phase_corr = phase_cal['Phase correction factor'] * apu.deg

    # Apply grid correction
    phase_cor += phase_corr

    idx = [6, 28, 0, 24, 4, 22, 5, 29, 1, 14, 26, 30, 23, 7, 27, 20, 25, 3, 15, 13, 31, 2, 19, 21]
    phase_cor = phase_cor[idx]

    m, n = shape.to_value('pix')

    #center = center - [26.1, 58.2] * apu.arcsec

    y = generate_xy(m, pixel_size=pixel_size[1], center=center[1])
    x = generate_xy(n, pixel_size=pixel_size[0], center=center[0])
    x, y = np.meshgrid(x, y)
    # Check apu are correct for exp need to be dimensionless and then remove apu for speed
    if (u[0] * x[0, 0]).unit == apu.dimensionless_unscaled and \
            (v[0] * y[0, 0]).unit == apu.dimensionless_unscaled:
        u = u.value
        v = v.value
        x = x.value
        y = y.value

        phase = (2 * np.pi * (x[..., np.newaxis] * u[np.newaxis, :]
                              + y[..., np.newaxis] * v[np.newaxis, :])
                 - phase_cor.to_value('rad'))

        H = np.concatenate([-np.cos(phase), -np.sin(phase), np.cos(phase), np.sin(phase)], axis=-1)

        # Application of the correction factors (needed because the units of the flux are photons
        # s^-1 cm^-2 arcsec^-2)
        H = H * 2.0 * M1 + M0
        H = H * (pixel_size[0] * pixel_size[1])
        H = np.transpose(H, (2, 0, 1))
        return H.reshape(u.shape[0]*4, -1)
    else:
        raise ValueError('Units dont match')


def em(countrates, vis, shape, pixel_size, maxiter=5000, tolerance=0.001, *, flare_xy, idx):
    r"""
    Count-based expectation maximisation imaging algorithm.


    Parameters
    ----------
    countrates :
        Meta pixel counts rates
    vis
        Visiblty
    shape

    pixel_size
    maxiter
    tolerance
    flare_xy
    idx

    Returns
    -------

    """

    # temp
    idx = [6, 28, 0, 24, 4, 22, 5, 29, 1, 14, 26, 30, 23, 7, 27, 20, 25, 3, 15, 13, 31, 2, 19, 21]
    ii = np.array([np.argwhere(vis.isc - 1 == i).ravel() for i in idx]).ravel()

    H = get_transmission_matrix(vis.u[ii], vis.v[ii], shape=shape, pixel_size=pixel_size,
                                center=flare_xy, pixel_sum=1)
    y = countrates[idx, ...]
    # Not quite sure why this is needed clearly related to ordering of counts and H matrix
    y = y.flatten(order='F')
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

        cstat = (2 / y[y_index].size) * (y[y_index].value * (np.log((y[y_index]/Hx[y_index]).value))
                                         + Hx[y_index].value - y[y_index].value).sum()

        # Stopping rule
        if i > 10 and (i % 25) == 0:
            emp_back_res = ((x.value * (Ht1.value - Hz.value))**2).sum()
            std_back_res = (x**2 * np.matmul(1/Hx, H2)).sum()
            std_index = emp_back_res / std_back_res

            logger.info(f'Iteration: {i}, StdDeV: {std_index}, C-stat: {cstat}')

            if std_index < tolerance:
                break
    m, n = shape.to_value('pix').astype(int)
    # Note the order keyword here to match `y.flatten(order='F')` above at line 108
    # TODO investigate why .T is needed probably because of fortran odering
    x_im = x.reshape(m, n, order='F').T
    return x_im
