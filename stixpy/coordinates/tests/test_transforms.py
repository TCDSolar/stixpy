from unittest import mock

import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import STIX_EPHEMERIS_CACHE, _get_ephemeris_data, get_hpc_info


@pytest.fixture(autouse=True)
def _clear_ephemeris_cache():
    """Isolate every test from cache pollution carried over from earlier tests."""
    STIX_EPHEMERIS_CACHE.clear()
    yield
    STIX_EPHEMERIS_CACHE.clear()


def _fake_ephemeris(start, *, n_rows=2000, dt=64 * u.s):
    """Synthesize an ANC-shaped ephemeris table for warm-cache tests."""
    times = start + np.arange(n_rows) * dt
    return QTable(
        {
            "time": times,
            "time_end": times + dt,
            "timedel": np.full(n_rows, 64.0) * u.s,
            "roll_angle_rpy": np.zeros((n_rows, 3)) * u.deg,
            "solo_loc_heeq_zxy": np.tile([1.0e8, 0.0, 0.0], (n_rows, 1)) * u.km,
            "y_srf": np.zeros(n_rows) * u.arcsec,
            "z_srf": np.zeros(n_rows) * u.arcsec,
            "sas_ok": np.ones(n_rows, dtype=int),
        }
    )


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
    roll = [0] * u.deg
    sas = [[10, 15]] * u.arcsec
    solo_pos = HeliographicStonyhurst(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime=obstime)
    mock.return_value = (roll, solo_pos.cartesian, sas)
    solo_hpc_coord = Helioprojective(0 * u.arcsec, 0 * u.arcsec, observer=solo_pos, obstime=obstime)
    stix_frame = STIXImaging(0 * u.arcsec, 0 * u.arcsec, obstime=obstime)
    stix_coord = solo_hpc_coord.transform_to(stix_frame)
    assert_quantity_allclose(stix_coord.Tx, -sas[0, 1])
    assert_quantity_allclose(stix_coord.Ty, sas[0, 0])


@pytest.mark.remote_data
def test_get_hpc_info_interp():
    t1 = Time("2022-03-28T18:43:00")
    t2 = Time("2022-03-28T18:44:00")
    t3 = Time("2022-03-28T18:45:00")
    t4 = Time("2022-03-28T18:46:00")

    roll1, solo_xyz1, ptg_1 = get_hpc_info(t1, t2)
    roll2, solo_xyz2, ptg_2 = get_hpc_info(t2, t3)
    roll3, solo_xyz3, ptg_3 = get_hpc_info(t3, t4)

    assert_quantity_allclose(ptg_1, ptg_2, rtol=0.03)
    assert_quantity_allclose(ptg_2, ptg_3, rtol=0.03)


@pytest.mark.remote_data
def test_stx_to_hpc_times():
    times = Time("2023-01-01") + np.arange(10) * u.min
    stix_coord = SkyCoord([0] * 10 * u.deg, [0] * 10 * u.deg, obstime=times, frame=STIXImaging)
    hp = stix_coord.transform_to(Helioprojective(obstime=times))
    stix_coord_rt = hp.transform_to(STIXImaging(obstime=times))
    assert_quantity_allclose(0 * u.deg, stix_coord.separation(stix_coord_rt), atol=1e-17 * u.deg)
    assert np.all(stix_coord.obstime.isclose(stix_coord_rt.obstime))


@pytest.mark.remote_data
def test_stx_to_hpc_obstime_end():
    times = Time("2023-01-01") + [0, 2] * u.min
    time_avg = np.mean(times)
    stix_coord = SkyCoord(0 * u.deg, 0 * u.deg, frame=STIXImaging(obstime=times[0], obstime_end=times[-1]))
    hp = stix_coord.transform_to(Helioprojective(obstime=time_avg))
    stix_coord_rt = hp.transform_to(STIXImaging(obstime=times[0], obstime_end=times[-1]))

    stix_coord_rt_interp = hp.transform_to(STIXImaging(obstime=times[1]))  # noqa: F841
    assert_quantity_allclose(0 * u.deg, stix_coord.separation(stix_coord_rt), atol=1e-9 * u.deg)
    assert np.all(stix_coord.obstime.isclose(stix_coord_rt.obstime))
    assert np.all(stix_coord.obstime_end.isclose(stix_coord_rt.obstime_end))


def test_stx_to_hpc_obstime_end_x2():
    # strange error that the call the second times crashes
    test_stx_to_hpc_obstime_end()
    test_stx_to_hpc_obstime_end()


@pytest.mark.remote_data
def test_get_aux_data():
    with pytest.raises(ValueError, match="No STIX pointing data found for time range"):
        _get_ephemeris_data(Time("2015-06-06"))  # Before the mission started

    t1 = Time("2022-08-28T16:02:00")
    aux_data = _get_ephemeris_data(t1)
    assert len(aux_data) > 0
    assert aux_data["time"].min() <= t1 <= aux_data["time_end"].max()

    t2 = Time("2022-08-28T16:04:00")
    aux_data = _get_ephemeris_data(t1, end_time=t2)
    assert len(aux_data) > 0
    # Cache returns a padded slice so min may be slightly before t1.
    assert aux_data["time"].min() <= t1 < t2 <= aux_data["time_end"].max()

    t1 = Time("2022-08-28T23:58:00")
    t2 = Time("2022-08-29T00:02:00")
    aux_data = _get_ephemeris_data(t1, end_time=t2)
    assert len(aux_data) > 0
    assert aux_data["time"].min() <= t1 < t2 <= aux_data["time_end"].max()


@pytest.mark.remote_data
def test_get_hpc_info():
    with pytest.warns(match="SAS solution not available.*"):
        roll, solo_heeq, stix_pointing = get_hpc_info(
            Time("2022-08-28T16:00:17"), end_time=Time("2022-08-28T16:00:18.000")
        )

    assert_quantity_allclose(roll, -3.3733292 * u.deg)
    assert_quantity_allclose(stix_pointing, [25.622768, 57.920166] * u.arcsec)
    assert_quantity_allclose(solo_heeq, [-97868803.82, 62984839.12, -5531987.445] * u.km)

    roll, solo_heeq, stix_pointing = get_hpc_info(Time("2022-08-28T16:00:00"), end_time=Time("2022-08-28T16:03:59.575"))
    assert_quantity_allclose(
        roll,
        -3.3732104 * u.deg,
    )
    assert_quantity_allclose(
        stix_pointing,
        [25.923784, 58.179676] * u.arcsec,
    )
    assert_quantity_allclose(solo_heeq, [-9.7867768e07, 6.2983744e07, -5532067.5] * u.km)

    roll, solo_heeq, stix_pointing = get_hpc_info(Time("2022-08-28T21:00:00"), end_time=Time("2022-08-28T21:03:59.575"))
    assert_quantity_allclose(roll, -3.3619127 * u.deg)
    assert_quantity_allclose(stix_pointing, [-26.070198, 173.48871] * u.arcsec)
    assert_quantity_allclose(solo_heeq, [-9.7671984e07, 6.2774768e07, -5547166.0] * u.km)


def test_get_hpc_info_array_times_hits_warm_cache():
    """Regression: array `times` against a warm cache must not blow up.

    Before the rework, _get_ephemeris_data passed the multi-element Time
    straight to TableLRUCache.get, broadcasting an N-row column against an
    M-element array. The bug was latent because the array case usually
    arrives with a cold cache and falls into the Fido path.
    """
    STIX_EPHEMERIS_CACHE.clear()
    base = Time("2023-06-15T00:00:00")
    STIX_EPHEMERIS_CACHE.put(_fake_ephemeris(base), source="fake.fits")

    times = base + 6 * u.h + np.arange(10) * u.min
    roll, solo_heeq, stix_pointing = get_hpc_info(times)

    assert roll.shape == (10,)
    assert solo_heeq.shape == (10, 3)
    assert stix_pointing.shape == (10, 2)


def test_get_hpc_info_single_time_hits_warm_cache():
    """Regression: scalar point query must hit the warm cache.

    Before the rework, the cache's strict time-equality check made point
    queries almost always miss (64 s ANC cadence vs. arbitrary timestamps),
    so every call re-ran Fido. With nearest-row lookup and range padding,
    a scalar query inside the cached day must return without fetching.
    """
    STIX_EPHEMERIS_CACHE.clear()
    base = Time("2023-06-15T00:00:00")
    STIX_EPHEMERIS_CACHE.put(_fake_ephemeris(base), source="fake.fits")

    # Pick an instant deliberately not aligned to the 64 s grid.
    t = base + 6 * u.h + 17.3 * u.s
    roll, solo_heeq, stix_pointing = get_hpc_info(t)

    assert np.isscalar(roll.value) or roll.size == 1
    assert solo_heeq.shape == (3,)


@pytest.mark.remote_data
def test_get_hpc_info_shapes():
    t = Time("2022-08-28T16:00:00") + np.arange(10) * u.min
    roll1, solo_heeq1, stix_pointing1 = get_hpc_info(t)
    roll2, solo_heeq2, stix_pointing2 = get_hpc_info(t[5:6])
    roll3, solo_heeq3, stix_pointing3 = get_hpc_info(t[5])

    assert_quantity_allclose(roll1[5], roll2)
    assert_quantity_allclose(solo_heeq1[5, :], solo_heeq2)
    assert_quantity_allclose(stix_pointing1[5, :], stix_pointing2)

    assert_quantity_allclose(roll3, roll2[0])
    assert_quantity_allclose(solo_heeq3, solo_heeq2)
    assert_quantity_allclose(stix_pointing3, stix_pointing2)
