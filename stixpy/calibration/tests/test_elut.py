from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from stixpy.calibration.elut import ELUTCorrection, get_elut, get_elut_correction


def test_get_elut():
    with pytest.raises(ValueError, match=r"No ELUT for for date.*"):
        get_elut(datetime(2019, 1, 1))

    elut = get_elut(datetime(2020, 6, 7))
    assert elut.file == "elut_table_20200519.csv"

    elut = get_elut(datetime(2024, 4, 9))
    assert elut.file == "elut_table_20240205.csv"
    assert elut.e_actual.shape == (32, 12, 31)


@pytest.fixture
def pixel_data():
    energy_mask = np.ones(32, dtype=int)
    energy_mask[[0, -1]] = 0  # drop the 0-4 keV and open-top channels
    return SimpleNamespace(
        energy_masks=SimpleNamespace(energy_mask=energy_mask),
        # A datetime centre (``TimeRange.center`` is a ``Time``) avoids an ErfaWarning from comparing ``Time`` with
        # the far-future end of the open ELUT index interval.
        time_range=SimpleNamespace(center=datetime(2024, 4, 9, 0, 30)),
    )


def test_get_elut_correction(pixel_data):
    e_ind = np.arange(5, 10)
    cor = get_elut_correction(e_ind, pixel_data)

    assert isinstance(cor, ELUTCorrection)
    assert cor._fields == ("e_cor_high", "e_cor_low", "bins", "bins_actual")
    e_cor_high, e_cor_low, bins, bins_actual = cor
    assert e_cor_high is cor.e_cor_high

    n_energy = 30
    assert cor.e_cor_low.shape == (32, 12)
    assert cor.e_cor_high.shape == (32, 12)
    assert cor.bins.shape == (1, 32, 12, n_energy)
    assert cor.bins_actual.shape == (1, 32, 12, n_energy)

    elut = get_elut(pixel_data.time_range.center)
    sci_edges = elut.e.value[1:-1]  # edges of the 30 unmasked science channels
    assert_allclose(cor.bins[0, 3, 7], np.diff(sci_edges))
    assert_allclose(cor.bins_actual[0], np.diff(elut.e_actual, axis=-1))

    # Fraction of the actual bin that falls inside the science range. Masked index i is science channel i + 1,
    # which has actual edges e_actual[..., i] and e_actual[..., i + 1] and science edges sci_edges[i : i + 2].
    lo, hi = e_ind[0], e_ind[-1]
    width_lo = elut.e_actual[..., lo + 1] - elut.e_actual[..., lo]
    width_hi = elut.e_actual[..., hi + 1] - elut.e_actual[..., hi]
    assert_allclose(cor.e_cor_low, (elut.e_actual[..., lo + 1] - sci_edges[lo]) / width_lo)
    assert_allclose(cor.e_cor_high, (sci_edges[hi + 1] - elut.e_actual[..., hi]) / width_hi)
    assert np.all((cor.e_cor_low > 0.5) & (cor.e_cor_low < 1.5))
    assert np.all((cor.e_cor_high > 0.5) & (cor.e_cor_high < 1.5))
