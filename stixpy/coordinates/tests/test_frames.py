import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import get_horizons_coord
from sunpy.coordinates.frames import HeliographicStonyhurst, Helioprojective
from sunpy.map import Map, make_fitswcs_header
from sunpy.map.mapbase import SpatialPair

from stixpy.coordinates.frames import STIXImaging, stix_frame_to_wcs, stix_wcs_to_frame
from stixpy.coordinates.transforms import hpc_to_stixim, stixim_to_hpc  # noqa
from stixpy.map.stix import STIXMap


@pytest.fixture
def stix_wcs():
    w = WCS(naxis=2)

    w.wcs.datebeg = "2024-01-01 00:00:00.000"
    w.wcs.dateavg = "2024-01-01 00:00:01.000"
    w.wcs.dateend = "2024-01-01 00:00:02.000"
    w.wcs.crpix = [10, 20]
    w.wcs.cdelt = np.array([2, 2])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["SXLN-TAN", "SXLT-TAN"]

    w.wcs.aux.hgln_obs = 10
    w.wcs.aux.hglt_obs = 20
    w.wcs.aux.dsun_obs = 1.5e11

    return w


@pytest.fixture
def stix_frame():
    obstime = "2024-01-01"
    obstime_end = "2024-01-01 00:00:02.000"
    observer = HeliographicStonyhurst(10 * u.deg, 20 * u.deg, 1.5e11 * u.m, obstime=obstime)

    frame_args = {"obstime": obstime, "observer": observer, "obstime_end": obstime_end, "rsun": 695_700_000 * u.m}

    frame = STIXImaging(**frame_args)
    return frame


def test_stix_wcs_to_frame(stix_wcs):
    frame = stix_wcs_to_frame(stix_wcs)
    assert isinstance(frame, STIXImaging)

    assert frame.obstime.isot == "2024-01-01T00:00:00.000"
    assert frame.rsun == 695700 * u.km
    assert frame.observer == HeliographicStonyhurst(
        10 * u.deg, 20 * u.deg, 1.5e11 * u.m, obstime="2024-01-01T00:00:01.000"
    )


def test_stix_wcs_to_frame_none():
    w = WCS(naxis=2)
    w.wcs.ctype = ["ham", "cheese"]
    frame = stix_wcs_to_frame(w)

    assert frame is None


def test_stix_frame_to_wcs(stix_frame):
    wcs = stix_frame_to_wcs(stix_frame)

    assert isinstance(wcs, WCS)
    assert wcs.wcs.ctype[0] == "SXLN-TAN"
    assert wcs.wcs.cunit[0] == "arcsec"
    assert wcs.wcs.datebeg == "2024-01-01 00:00:00.000"
    assert wcs.wcs.dateavg == "2024-01-01 00:00:01.000"
    assert wcs.wcs.dateend == "2024-01-01 00:00:02.000"

    assert wcs.wcs.aux.rsun_ref == stix_frame.rsun.to_value(u.m)
    assert wcs.wcs.aux.dsun_obs == 1.5e11
    assert wcs.wcs.aux.hgln_obs == 10
    assert wcs.wcs.aux.hglt_obs == 20


def test_stix_frame_to_wcs_none():
    wcs = stix_frame_to_wcs(Helioprojective())
    assert wcs is None


@pytest.mark.remote_data
def test_stix_frame_map():
    data = np.random.rand(512, 512)
    obstime = "2023-01-01 12:00:00"
    obstime_end = "2023-01-01 12:02:00"
    solo = get_horizons_coord("solo", time=obstime)
    coord = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, obstime=obstime, obstime_end=obstime_end, observer=solo, frame=STIXImaging
    )
    header = make_fitswcs_header(
        data, coord, scale=[8, 8] * u.arcsec / u.pix, telescope="STIX", instrument="STIX", observatory="Solar Orbiter"
    )
    smap = Map((data, header))
    assert isinstance(smap, STIXMap)
    assert smap.coordinate_system == SpatialPair(axis1="SXLN-TAN", axis2="SXLT-TAN")
    assert isinstance(smap.coordinate_frame, STIXImaging)
    smap.plot()
    smap.draw_limb()
    smap.draw_grid()
