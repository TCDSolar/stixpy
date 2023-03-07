
import pytest
import astropy.units as u
import numpy as np

from astropy.time import Time
from numpy.testing import assert_equal

from stixpy.utils.time import times_to_indices

@pytest.mark.parametrize('duration,expected_indices', [(5*u.s,  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                       (10*u.s, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                                                       (15*u.s, [0, 1, 3, 4, 6, 7, 9]),
                                                       (20*u.s, [0, 2, 4, 6, 8, 9]),
                                                       (200*u.s, [0,9])])
def test_times_to_indices_scalar(duration, expected_indices):
    observed_times = Time('2021-12-20T00:00:00.0') + np.arange(10) * 10*u.s
    indices = times_to_indices(duration, observed_times)
    assert_equal(indices, np.array(expected_indices))

def test_times_to_indices_1d_times():
    observed_times = Time.now() + np.arange(10) * 10 * u.s
    request_times = observed_times[:]
    indices = times_to_indices(request_times, observed_times)
    assert_equal(indices, np.arange(10))
    indices1 = times_to_indices(request_times - 5.0 * u.s, observed_times)
    assert_equal(indices1, np.arange(9))
    indices2 = times_to_indices(request_times + 5 * u.s, observed_times)
    assert_equal(indices2, np.arange(10))

    uneven_obs_time = observed_times[0] + [5,  15,  30,  50,  70,  85,  95, 100] * u.s
    request_times = observed_times[0] + np.cumsum(np.arange(8)*10) * u.s
    indices3 = times_to_indices(request_times, uneven_obs_time)
    assert_equal(indices3, [0, 2, 3, 7])

def test_times_to_indices_2d_times():
    observed_times = Time.now() + np.arange(10) * 10 * u.s
    tmp = np.arange(10)*10
    requested_times = (observed_times[0]+[0, 90]*u.s).reshape(2, 1)
    indices = times_to_indices(requested_times, observed_times)
    assert_equal(indices, np.array([[0, 9]]))
    requested_times = observed_times[0] + np.vstack((tmp[0:-1], tmp[1:]))*u.s

    indices = times_to_indices(requested_times, observed_times)
    assert_equal(indices,  np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                                  [5, 6], [6, 7], [7, 8], [8, 9]]))
    requested_times = observed_times[0] + [[0, 18], [18, 27], [27, 30], [30, 50], [50, 62],[62, 110]]*u.s
    indices2 = times_to_indices(requested_times, observed_times)
    assert_equal(indices2, np.array([[0, 2], [2, 3], [3, 3], [3, 5], [5, 6], [6, 9]]))
