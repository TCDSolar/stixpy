"""
One overriding requirement for STIX is that the incoming photon rate can be determined with high
accuracy from the measured count rate. As such the dead time or equivalently the live time must be
accurately determined from the trigger rates. The live time fraction is defined as the ratio of the
event trigger rate to the photon rate :math:`R_{trigger}/R_{photon}`. The dead time is primarily
controlled by two timescales the readout time :math:`\\tau` and latency :math:`\\eta`.

These timescales correspond to two distinct contributions to the effective live time of a group of
ASICs. First, photons arriving during the read-out sequence of a previously triggered ASIC of that
group will not be registered as the trigger circuit is inactive during the readout. Second, of those
events that did trigger, a certain fraction will suffer from multiple hits and will be discarded,
if one or more additional triggers occur during the latency time before the read-out. The live time
fraction the ratio of the event rate to the photon rate is thus given by

.. math::

    R_{photon} = \\frac{R_{trigger}}{1 -  R_{trigger}(\\tau+\\eta)}

    \\text{lifetime fraction} = \\frac{e^{-\\eta R_{photon}}}{ 1 + R_{photon}(\\tau+\\eta)}

where :math:`R_{trigger}` is the observed trigger rate and :math:`R_{photon}` is the derived
incident photon rate.

The plot below shows the behaviour of the trigger rate taking these effects into account.

.. plot::

    import numpy as np

    from matplotlib import pyplot as plt

    from stixpy.calibration.livetime import get_livetime_fraction

    eta = 2.5e-6
    tau = 12.5e-6
    ph_in = np.logspace(3, 6, 50)
    trig1 = ph_in / (1 + ph_in*(tau+eta) )
    trig2 = trig1*np.exp(-1*ph_in*eta)

    livetime_fraction1, livetime_fraction2, ph_out = get_livetime_fraction(trig1, eta=eta, tau=tau)

    plt.plot(ph_in, ph_in,  label='Photons In')
    plt.loglog()
    plt.xlim(1e3, 1e6)
    plt.ylim(1e3, 1e6)
    plt.plot(ph_in, trig1, '-.', label='Triggers')
    plt.plot(ph_in, trig2, ':',  label='Triggers (inc two photon)')
    plt.plot(ph_in, trig2/livetime_fraction2, '+',
             label='Photons derived from Triggers (inc two photon)')
    plt.xlabel('Incoming Photon Rate')
    plt.ylabel('Trigger Rate')
    plt.title(rf'$\\tau =$ {tau} s, $\\eta =$ {eta} s')
    plt.legend()
    plt.show()

References
----------

STIX-TN-0015-ETH_I1R0_Caliste_Rates
"""

import astropy.units as u
import numpy as np

__all__ = ["pileup_correction_factor", "get_livetime_fraction","livetime_counts_corr"]

from stixpy.io.readers import read_subc_params


def pileup_correction_factor():
    r"""
    Estimate the fractional area of a single large pixel compared with the rest of the detector group.

    This is used in estimating the probability that when two photons are detected in the same group if
    the first hits a given large pixel that the second hits a different pixel resulting in anti-coincidence rejection.
    Otherwise if both hit the same pixel pileup will occur.

    Notes
    -----
    This assumes uniform illumination over the detector pair
    This ignores the possibility of the first hit being a small pixel
    """

    subc_str = read_subc_params()
    large_pixel_area = (subc_str["L Pixel Xsize"] * subc_str["L Pixel Ysize"])[0]
    small_pixel_area = (subc_str["S Pixel Xsize"] * subc_str["S Pixel Ysize"])[0]
    # half a small pixel overlaps each big pixel
    large_pixel_area_corrected = large_pixel_area - 0.5 * small_pixel_area
    detector_area = (subc_str["Detect Xsize"] * subc_str["Detect Ysize"])[0]
    big_pixel_fraction = large_pixel_area_corrected / detector_area
    prob_diff_pix = (2 / big_pixel_fraction - 1) / (2 / big_pixel_fraction)

    return prob_diff_pix


def get_livetime_fraction(trigger_rate,*, eta=1.1e-6 *u.s, tau=10.1e-6 * u.s):
    """
    Return the live time fraction for the given trigger rate.

    Parameters
    ----------
    trigger_rate : `float`
        Event trigger rate
    eta : `float`
        Latency per event
    tau : `float`
        Readout time per event

    Returns
    -------
    `float`, `float`, `float`:
        The live time fraction
    """
    beta = 0.94059104  # pileup_correction_factor()

    # tau = tau / time_del
    # eta = eta / time_del

    photons_in = trigger_rate / (1.0 - trigger_rate * (tau + eta))
    livetime_fraction = 1 / (1.0 + (tau + eta) * photons_in)
    two_photon = np.exp(-eta * beta * photons_in) * livetime_fraction


    return livetime_fraction, two_photon, photons_in

     

    
