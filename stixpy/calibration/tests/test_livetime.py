import numpy as np

from stixpy.calibration.livetime import get_livetime_fraction


def test_get_livetime():
    eta = 2.5e-6
    tau = 12.5e-6
    beta = 0.94059104
    ph_in = np.arange(1000) * 1e3

    # Simulate normal trigger process
    trig1 = ph_in / (1 + ph_in * (tau + eta))
    # Two photon contribution
    trig2 = trig1 * np.exp(-1 * beta * ph_in * eta)

    livetime_fraction1, livetime_fraction2, ph_out = get_livetime_fraction(trig1, eta=eta, tau=tau)

    assert np.allclose(ph_in, ph_out)
    assert np.allclose(ph_in, trig1 / livetime_fraction1)
    assert np.allclose(ph_in, trig2 / livetime_fraction2)
