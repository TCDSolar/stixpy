import numpy as np
import pytest

from stixpy.science import *


@pytest.mark.parametrize("input_file,output",
    [('data/solo_L1_stix-sci-xray-l0-87031824_20200508T011207_V01.fits', (XrayLeveL0, 5, 10, 12, 32))])
def test_science_l0(input_file, output):
    type_, num_times, num_detectors, num_pixels, num_energies = output
    res = ScienceData.from_fits(input_file)
    assert isinstance(res, type_)
    assert len(res.data) == num_times
    assert res.detectors.sum() == num_detectors
    assert res.pixels.sum() == num_pixels
    assert len(res.energies) == num_energies


@pytest.mark.parametrize("input_file,output",
    [('data/solo_L1_stix-sci-xray-l1-1167338000_20200518T165743_V01.fits', (XrayLeveL1, 5, 10, 12, 32))])
def test_science_l1(input_file, output):
    type_, num_times, num_detectors, num_pixels, num_energies = output
    res = ScienceData.from_fits(input_file)
    assert isinstance(res, type_)
    assert len(res.data) == num_times
    assert res.detectors.sum() == num_detectors
    assert res.pixels.sum() == num_pixels
    assert len(res.energies) == num_energies


@pytest.mark.parametrize("input_file,output",
    [('data/solo_L1_stix-sci-xray-l3-87031827_20200508T011207_V01.fits', (XrayVisibity, 5, 10, 12, 32))])
def test_science_l3(input_file, output):
    type_, num_times, num_detectors, num_pixels, num_energies = output
    res = ScienceData.from_fits(input_file)
    assert isinstance(res, type_)
    assert len(res.data) == num_times
    assert res.detectors.sum() == num_detectors
    assert np.array_equal(res.pixels.sum(axis=0), np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    assert len(res.energies) == num_energies
