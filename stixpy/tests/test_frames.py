import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from sunpy.coordinates.frames import Helioprojective, HeliographicStonyhurst

from stixpy.frames import STIXImagingFrame


@pytest.mark.remote_data
def test_stix_imaging_to_hpc():
    solo_heeq_pos = [58137120.0, 18048598.0, -4397405.5] * u.km
    solo_heeq_coord = HeliographicStonyhurst(*solo_heeq_pos, representation='cartesian',
                                             obstime='2022-03-14T12:00:00')
    solo_hpc_frame = Helioprojective(observer=solo_heeq_coord.transform_to(HeliographicStonyhurst),
                                     obstime='2022-03-14T12:00:00')
    stx = STIXImagingFrame(0 * u.deg, 0 * u.deg, obstime='2022-03-14T12:00:00')
    hpc_res = stx.transform_to(solo_hpc_frame)
    stx_rt = hpc_res.transform_to(stx)

    assert np.allclose(stx.x, stx_rt.x)
    assert np.allclose(stx.y, stx_rt.y)
    assert np.allclose(stx.distance, stx_rt.distance)


@pytest.mark.remote_data
def test_test():
    obstime = Time('2021-05-22T02:52:00')
    earth_coord = Helioprojective(-395 * u.arcsec, 330 * u.arcsec, observer='earth', obstime=obstime)
    earth_to_stix = earth_coord.transform_to(STIXImagingFrame(obstime=obstime))

    stix_coord = STIXImagingFrame(earth_to_stix.x, earth_to_stix.y, obstime=obstime)
    stix_to_earth = stix_coord.transform_to(Helioprojective(observer='Earth', obstime=obstime))
    print(1)
