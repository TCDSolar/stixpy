import numpy as np
import pytest
from numpy.testing import assert_array_equal

import astropy.units as u

from stixpy.product.sources._rcr import (
    RCR_JUMP_THRESHOLD,
    RCR_SIGNAL_DETECTORS,
    _handle_nested_pairs,
    _rcr_error,
    _rcr_shift,
    _rcr_signal,
    _rcr_warning,
)

HIGH = 3 * RCR_JUMP_THRESHOLD
LOW = 0.0


def rcr_change(at, n=10, before=0, after=1):
    """Reported RCR state that changes from `before` to `after` at bin `at`."""
    return np.r_[[before] * at, [after] * (n - at)]


def insert_and_remove(n=20, insert=5, remove=15):
    """Counts that drop at `insert` and recover at `remove`, so the first and last bins match."""
    signal = np.full(n, HIGH)
    signal[insert:remove] = LOW
    return signal


def test_rcr_signal_sums_all_but_time_and_scales():
    counts = np.ones((3, 4, 12, 5)) * u.ct
    counts[1, 0, 0, 0] = np.nan
    signal = _rcr_signal(counts, n_detectors=4)
    expected = np.array([240.0, 239.0, 240.0]) * RCR_SIGNAL_DETECTORS / 4
    assert_array_equal(signal, expected)


def test_rcr_signal_zero_detectors():
    assert_array_equal(_rcr_signal(np.ones((2, 3)), n_detectors=0), [3 * RCR_SIGNAL_DETECTORS] * 2)


def test_rcr_shift_no_change_is_identity():
    rcr = np.full(10, 2)
    assert_array_equal(_rcr_shift(rcr, insert_and_remove(10, 3, 7)), rcr)


def test_rcr_shift_max_zero_returns_input():
    rcr = np.zeros(10, dtype=int)
    signal = insert_and_remove(10, 3, 7)
    assert_array_equal(_rcr_shift(rcr, signal), rcr)


def test_rcr_shift_single_change_aligned_with_jump():
    # The steps in counts are between bins 4 and 5 and between 14 and 15, so the
    # jumps are at bins 4 and 14 and, as in GSW, those are the first bins in the new state.
    signal = insert_and_remove(insert=5, remove=15)
    rcr = np.r_[[0] * 4, [1] * 10, [0] * 6]
    assert_array_equal(_rcr_shift(rcr, signal), rcr)


@pytest.mark.parametrize("reported", [3, 5, 6])
def test_rcr_shift_change_off_jump_moves_onto_it(reported):
    # One or two bins either side of the jump at 4 (bin 2 would tie with the bin-0 jump GSW puts in front).
    signal = insert_and_remove(insert=5, remove=15)
    rcr = np.r_[[0] * reported, [1] * (15 - reported), [0] * 5]
    expected = np.r_[[0] * 4, [1] * 10, [0] * 6]
    assert_array_equal(_rcr_shift(rcr, signal), expected)


def test_rcr_shift_close_jumps_keep_last():
    # The attenuator moving during bin 4 gives jumps at bins 3 and 4; the last one is kept.
    signal = np.r_[[HIGH] * 4, [HIGH / 2], [LOW] * 10, [HIGH] * 5]
    jumps = np.flatnonzero(np.abs(signal - np.roll(signal, -1)) > RCR_JUMP_THRESHOLD)
    assert_array_equal(jumps, [3, 4, 14])
    rcr = np.r_[[0] * 3, [1] * 12, [0] * 5]
    expected = np.r_[[0] * 4, [1] * 10, [0] * 6]
    assert_array_equal(_rcr_shift(rcr, signal), expected)


def test_rcr_shift_jump_below_threshold_is_ignored():
    signal = insert_and_remove()
    signal[5:15] = HIGH - RCR_JUMP_THRESHOLD
    rcr = rcr_change(6, n=20)
    # No jump, so every bin takes the last state (see the next test).
    assert_array_equal(_rcr_shift(rcr, signal), np.ones(20))


def test_rcr_shift_no_jump_takes_last_state():
    # IDL: where() gives -1, jumps = [0, -1], no jump is kept, jumps_use = jumps[-1] = -1
    # and every change moves to -1, clipped to 0.
    rcr = np.r_[[0] * 3, [1] * 3, [2] * 4]
    assert_array_equal(_rcr_shift(rcr, np.zeros(10)), np.full(10, 2))


def test_rcr_shift_wrap_around_adds_jump_at_last_bin():
    # shift() wraps, so the last bin is compared with the first: counts that
    # drop and stay low give a jump at the last bin as well as at the drop.
    signal = np.r_[[HIGH] * 5, [LOW] * 5]
    jumps = np.flatnonzero(np.abs(signal - np.roll(signal, -1)) > RCR_JUMP_THRESHOLD)
    assert_array_equal(jumps, [4, 9])
    # A change reported at bin 7 is closer to the wrap-around jump at 9 than to the real one at 4.
    assert_array_equal(_rcr_shift(rcr_change(7), signal), rcr_change(9))
    # One reported at bin 6 still moves to the real jump.
    assert_array_equal(_rcr_shift(rcr_change(6), signal), rcr_change(4))


def test_rcr_shift_changes_on_same_jump_collapse():
    signal = insert_and_remove(insert=5, remove=15)
    rcr = np.r_[[0] * 4, [1], [2] * 10, [0] * 5]
    expected = np.r_[[0] * 4, [2] * 10, [0] * 6]
    assert_array_equal(_rcr_shift(rcr, signal), expected)


def test_rcr_error():
    rcr = [0, 0, 1, 1]
    _rcr_error([], rcr)
    _rcr_error([0, 1], rcr)
    _rcr_error([2, 3], rcr)
    with pytest.raises(ValueError, match="index 1 has RCR=0, index 2 has RCR=1"):
        _rcr_error([1, 2, 3], rcr)


def test_rcr_warning():
    rcr = [0, 0, 1, 1]
    with pytest.warns(UserWarning, match="RCR state change detected") as record:
        _rcr_warning([0, 1, 2, 3], rcr)
    messages = [str(w.message) for w in record]
    assert len(messages) == 2
    assert "index 0 has RCR=0, index 2 has RCR=1" in messages[0]
    assert "index 0 has RCR=0, index 3 has RCR=1" in messages[1]


def test_handle_nested_pairs():
    rcr = [0, 0, 1, 1, 0, 0]
    pairs = [[0, 1], [4, 5]]
    assert _handle_nested_pairs(pairs, rcr) is pairs

    with pytest.warns(UserWarning, match="RCR state differs across nested pairs"):
        assert _handle_nested_pairs([[0, 1], [2, 3]], rcr) == [[0, 1], [2, 3]]

    with pytest.raises(ValueError, match="RCR state change detected"):
        _handle_nested_pairs([[0, 1], [1, 2]], rcr)
