import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from stixpy.calibration.grid import get_grid_transmission
from stixpy.coordinates.frames import STIXImaging


# Output values taken from IDL routine
@pytest.mark.parametrize(
    "input,out",
    [
        (
            SkyCoord(*[0, 0] * u.arcsec, frame=STIXImaging),
            [
                [
                    0.38359416,
                    0.24363935,
                    0.26036045,
                    0.26388368,
                    0.28507271,
                    0.24874112,
                    0.41114664,
                    0.25676244,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    0.26332209,
                    0.24405889,
                    0.26135680,
                    1.00000000,
                    1.00000000,
                    1.00000000,
                    0.26654890,
                    0.25451636,
                    0.25651491,
                    0.27999377,
                    0.23954011,
                    0.29099935,
                    0.25652689,
                    0.23173763,
                    0.27111799,
                    0.39068285,
                    0.26008865,
                    0.26436430,
                    0.26104531,
                ]
            ],
        ),
        (
            SkyCoord(*[0, 500.0] * u.arcsec, frame=STIXImaging),
            [
                0.38119361,
                0.23916472,
                0.25980136,
                0.26185310,
                0.28049889,
                0.24295497,
                0.39690757,
                0.25528145,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                1.00000000,
                0.26191920,
                0.24099952,
                0.26109657,
                1.00000000,
                1.00000000,
                1.00000000,
                0.26544300,
                0.25416556,
                0.25598019,
                0.27559373,
                0.23802824,
                0.28210551,
                0.25501686,
                0.23110151,
                0.26802292,
                0.38006619,
                0.25902179,
                0.26052520,
                0.25992528,
            ],
        ),
    ],
)
def test_grid_transmission(input, out):
    grid_transmission = get_grid_transmission(input)
    assert np.allclose(grid_transmission, np.array(out))
