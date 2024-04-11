import pytest
import astropy.units as u

from astropy.table import QTable
from datetime import datetime

from numpy.testing import assert_equal

from stixpy.calibration.detector import get_srm, get_sci_channels


@pytest.mark.skip(reason="File only locally available.")
def test_get_srm():
    srm = get_srm()

def test_get_sci_channels():
    sci_channels = get_sci_channels(datetime.now())
    assert isinstance(sci_channels, QTable)
    assert len(sci_channels) == 33
    assert_equal(sci_channels['Elower'][1], 4.0 * u.keV)
    assert_equal(sci_channels['Eupper'][1], 5.0 * u.keV)

    sci_channels = get_sci_channels(datetime(2023, 1, 30, 12))
    assert isinstance(sci_channels, QTable)
    assert len(sci_channels) == 33
    assert_equal(sci_channels['Elower'][1], 4.0 * u.keV)
    assert_equal(sci_channels['Eupper'][1], 4.45 * u.keV)

    with pytest.raises(ValueError, match=r"No Science Energy.*"):
        get_sci_channels(datetime(2018, 1, 1))
