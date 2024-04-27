from unittest import mock

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.coordinates.frames import STIXImaging


def assert_quantity_allclose(x, param):
    pass


@pytest.mark.skip(reason="Test data maybe incorrect")
@pytest.mark.remote_data
def test_transforms_with_know_values():
    obstime = Time("2021-05-22T02:52:00")
    earth_coord = Helioprojective(-395 * u.arcsec, 330 * u.arcsec, observer="earth", obstime=obstime)
    earth_to_stix = earth_coord.transform_to(STIXImaging(obstime=obstime))

    stix_coord = STIXImaging(earth_to_stix.x, earth_to_stix.y, obstime=obstime)
    stix_to_earth = stix_coord.transform_to(Helioprojective(observer="Earth", obstime=obstime))

    # assert_quantity_allclose(earth_to_stix.x, 410.747992 * u.arcsec)
    # assert_quantity_allclose(earth_to_stix.y, 840.925692*u.arcsec)
    assert_quantity_allclose(earth_to_stix.x, 367.30422293 * u.arcsec)
    assert_quantity_allclose(earth_to_stix.y, 731.37631103 * u.arcsec)
    assert_quantity_allclose(earth_coord.Tx, stix_to_earth.Tx)
    assert_quantity_allclose(earth_coord.Ty, stix_to_earth.Ty)


@mock.patch("stixpy.coordinates.transforms._get_aux_data")
def test_hpc_to_stx(mock):
    # fake some data to make check easier
    obstime = Time("2021-05-22T02:52:00")
    roll, yaw, pitch, sas_x, sas_y = [0, 0, 0, 0, 0] * u.arcsec
    solo_pos = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=obstime)
    mock.return_value = (roll, yaw, pitch, sas_x, sas_y, solo_pos.cartesian)
    solo_hpc_coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, observer=solo_pos, obstime=obstime)
    stix_frame = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=obstime)
    stix_coord = solo_hpc_coord.transform_to(stix_frame)
    # should match the offset -8, 60
    assert_quantity_allclose(stix_coord.Tx, -8 * u.arcsec)
    assert_quantity_allclose(stix_coord.Ty, 60 * u.arcsec)


@mock.patch("stixpy.coordinates.transforms._get_aux_data")
def test_hpc_to_stx_no_sas(mock):
    # fake some data to make check easier
    obstime = Time("2021-05-22T02:52:00")
    roll, yaw, pitch, sas_x, sas_y = [0, 10, 10, np.nan, np.nan] * u.arcsec
    solo_pos = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=obstime)
    mock.return_value = (roll, yaw, pitch, sas_x, sas_y, solo_pos.cartesian)
    solo_hpc_coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, observer=solo_pos, obstime=obstime)
    stix_frame = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=obstime)
    with pytest.warns(Warning, match="SAS solution not"):
        stix_coord = solo_hpc_coord.transform_to(stix_frame)
        # should match the offset -8, 60 added to yaw and pitch 10, 10
        assert_quantity_allclose(stix_coord.Tx, (10 - 8) * u.arcsec)
        assert_quantity_allclose(stix_coord.Ty, (10 + 60) * u.arcsec)
<<<<<<< HEAD


def test_stix_wcs_to_frame(stix_wcs):
    frame = stix_wcs_to_frame(stix_wcs)
    assert isinstance(frame, STIXImaging)


def test_stix_wcs_to_frame_none():
    w = WCS(naxis=2)
    w.wcs.ctype = ['ham', 'cheese']
    frame = stix_wcs_to_frame(w)

    assert frame is None


def test_stix_frame_to_wcs(stix_frame):
    wcs = stix_frame_to_wcs(stix_frame)

    assert isinstance(wcs, WCS)
    assert wcs.wcs.ctype[0] == 'SXLN-TAN'
    assert wcs.wcs.cunit[0] == 'arcsec'
    assert wcs.wcs.dateobs == '2024-01-01 00:00:00.000'

    assert wcs.wcs.aux.rsun_ref == stix_frame.rsun.to_value(u.m)
    assert wcs.wcs.aux.dsun_obs == 1.5e11
    assert wcs.wcs.aux.hgln_obs == 10
    assert wcs.wcs.aux.hglt_obs == 20


def test_stix_frame_to_wcs_none():
    wcs = stix_frame_to_wcs(Helioprojective())
    assert wcs is None
=======
>>>>>>> 8f2c7e0 (Working end to end)
