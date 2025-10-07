import numpy as np

from stixpy.calibration.livetime import get_livetime_fraction, pileup_correction_factor


def test_pileup_correction_factor():
    pileup_factor = pileup_correction_factor()
    # IDL value
    assert pileup_factor == 0.9405910326086957


def test_get_livetime():
    eta = 2.5e-6
    tau = 12.5e-6
    beta = 0.94059104
    ph_in = np.arange(1000) * 1e3

    # Simulate normal trigger process
    trig1 = ph_in / (1 + ph_in * (tau + eta))
    # Two photon contribution
    trig2 = trig1 * np.exp(-1 * beta * ph_in * eta)

    livetime, two_photon, ph_out = get_livetime_fraction(trig1, eta=eta, tau=tau)

    assert np.allclose(ph_in, ph_out)
    assert np.allclose(ph_in, trig2 / livetime)
    assert np.allclose((trig2 / trig1)[1:], two_photon[1:])
