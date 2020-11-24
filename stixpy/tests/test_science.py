from pathlib import Path

import numpy as np
import pytest

from stixpy.data import test
from stixpy.science import *


def test_science_l0():
    type_, num_times, num_detectors, num_pixels, num_energies = XrayLeveL0, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_L0)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_science_l1():
    type_, num_times, num_detectors, num_pixels, num_energies = XrayLeveL1, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_L1)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.eye(num_pixels)[None, ...])
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_science_l2():
    type_, num_times, num_detectors, num_pixels, num_energies = XrayLeveL2, 5, 32, 4, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_L2)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    #assert np.array_equal(res.pixels.masks.astype(int), np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_science_l3():
    type_, num_times, num_detectors, num_pixels, num_energies = XrayVisibility, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_XRAY_L3)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    #assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))


def test_spectrogram():
    type_, num_times, num_detectors, num_pixels, num_energies = XraySpectrogram, 5, 32, 12, 32
    res = ScienceData.from_fits(test.STIX_SCI_SPECTROGRAM)
    assert isinstance(res, type_)
    assert len(res.times) == num_times
    assert np.array_equal(res.detectors.masks, np.ones((1, num_detectors)))
    assert np.array_equal(res.pixels.masks, np.ones((1, num_pixels)))
    assert np.array_equal(res.energy_masks.masks, np.ones((1, num_energies)))
