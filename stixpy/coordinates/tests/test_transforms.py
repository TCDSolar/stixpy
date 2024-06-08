from unittest import mock

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from numpy.testing import assert_allclose
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import _get_aux_data, get_hpc_info


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


@mock.patch("stixpy.coordinates.transforms.get_hpc_info")
def test_hpc_to_stx(mock):
    # fake some data to make check easier
    obstime = Time("2021-05-22T02:52:00")
    roll = 0 * u.deg
    sas = [10, 15] * u.arcsec
    solo_pos = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=obstime)
    mock.return_value = (roll, solo_pos.cartesian, sas)
    solo_hpc_coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, observer=solo_pos, obstime=obstime)
    stix_frame = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=obstime)
    stix_coord = solo_hpc_coord.transform_to(stix_frame)
    assert_quantity_allclose(stix_coord.Tx, -sas[1])
    assert_quantity_allclose(stix_coord.Ty, sas[0])


@pytest.mark.remote_data
def test_get_aux_data():
    with pytest.raises(ValueError, match="No STIX pointing data found for time range"):
        _get_aux_data(Time("2015-06-06"))  # Before the mission started

    aux_data = _get_aux_data(Time("2022-08-28T16:02:00"))
    assert len(aux_data) == 1341

    aux_data = _get_aux_data(Time("2022-08-28T16:02:00"), end_time=Time("2022-08-28T16:04:00"))
    assert len(aux_data) == 1341

    aux_data = _get_aux_data(Time("2022-08-28T23:58:00"), end_time=Time("2022-08-29T00:02:00"))
    assert len(aux_data) == 2691


@pytest.mark.remote_data
def test_get_hpc_info():
    with pytest.warns(match="SAS solution not available.*"):
        roll, solo_heeq, stix_pointing = get_hpc_info(Time(["2022-08-28T16:00:16", "2022-08-28T16:00:17.000"]))

    assert_quantity_allclose(-3.3733306 * u.deg, roll)
    assert_quantity_allclose([25.61525646, 57.914266] * u.arcsec, stix_pointing)
    assert_quantity_allclose([-97868816.0, 62984852.0, -5531986.5] * u.km, solo_heeq)

    roll, solo_heeq, stix_pointing = get_hpc_info(Time("2022-08-28T16:00:00"), end_time=Time("2022-08-28T16:03:59.575"))
    assert_quantity_allclose(-3.3732104 * u.deg, roll)
    assert_quantity_allclose([25.923784, 58.179676] * u.arcsec, stix_pointing)
    assert_quantity_allclose([-9.7867768e07, 6.2983744e07, -5532067.5] * u.km, solo_heeq)

    roll, solo_heeq, stix_pointing = get_hpc_info(Time("2022-08-28T21:00:00"), end_time=Time("2022-08-28T21:03:59.575"))
    assert_quantity_allclose(-3.3619127 * u.deg, roll)
    assert_quantity_allclose([-26.070198, 173.48871] * u.arcsec, stix_pointing)
    assert_quantity_allclose([-9.7671984e07, 6.2774768e07, -5547166.0] * u.km, solo_heeq)


@pytest.mark.remote_data
def test_get_hpc_info_shapes():
    t = Time("2022-08-28T16:00:00") + np.arange(10) * u.min
    roll1, solo_heeq1, stix_pointing1 = get_hpc_info(t)

    roll2, solo_heeq2, stix_pointing2 = get_hpc_info(t[[0, -1]])

    roll3, solo_heeq3, stix_pointing3 = get_hpc_info(t[5])

    assert_allclose(np.mean(roll1, axis=0), roll2, rtol=1e-6)
    assert_allclose(np.mean(roll1, axis=0), roll3, rtol=5e-5)

    assert_allclose(np.mean(solo_heeq1, axis=0), solo_heeq2.value, rtol=1e-6)
    assert_allclose(np.mean(solo_heeq1, axis=0), solo_heeq3[0, :], rtol=1e-5)
