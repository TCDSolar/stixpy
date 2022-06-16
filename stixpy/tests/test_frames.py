import astropy.units as u
import numpy as np
import pytest

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
