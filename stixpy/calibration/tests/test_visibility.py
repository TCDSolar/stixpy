import astropy.units as u

from stixpy.calibration.visibility import get_uv_points_data


def test_get_uv_points_data():
    uv_data = get_uv_points_data()
    assert uv_data['u'][0]     == -0.03333102271709666 / u.arcsec
    assert uv_data['v'][0]     == 0.005908224739704219 / u.arcsec
    assert uv_data['isc'][0]   == 1
    assert uv_data['label'][0] == '3c'
