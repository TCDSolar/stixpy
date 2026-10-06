from pathlib import Path

import numpy as np

import astropy.units as u

from stixpy.io.readers import read_energy_channel_index, read_sci_energy_channels

__all__ = ["get_sci_channels", "tailing_matrix", "tailing_matrix_corrected"]

SCI_INDEX = None
SCI_CHANNELS = {}


def get_sci_channels(date):
    r"""
    Get the science energy channels for given date.

    Parameters
    ----------
    date : `datetime.datetime`
        Date to lookup science energy channels.

    Returns
    -------
    `astropy.table.QTable`
        Science Energy Channels
    """
    global SCI_INDEX, SCI_CHANNELS

    # Cache index
    if SCI_INDEX is None:
        root = Path(__file__).parent.parent
        sci_chan_index_file = Path(root, *["config", "data", "detector", "science_echan_index.csv"])
        sci_chan_index = read_energy_channel_index(sci_chan_index_file)
        SCI_INDEX = sci_chan_index

    sci_info = SCI_INDEX.at(date)
    if len(sci_info) == 0:
        raise ValueError(f"No Science Energy Channel file found for date {date}")
    elif len(sci_info) > 1:
        raise ValueError(f"Multiple Science Energy Channel file for date {date}")
    start_date, end_date, sci_echan_file = list(sci_info)[0]

    # Cache sci channels
    if sci_echan_file.name in SCI_CHANNELS:
        sci_echan_table = SCI_CHANNELS[sci_echan_file.name]
    else:
        sci_echan_table = read_sci_energy_channels(sci_echan_file)
        SCI_CHANNELS[sci_echan_file.name] = sci_echan_table

    return sci_echan_table


@u.quantity_input
def tailing_matrix(
    ph_edges: u.Quantity[u.keV],
    xsec_energy: u.Quantity[u.keV],
    xsec: u.Quantity[1 / u.cm],
    depth: u.Quantity[u.cm] = 1 * u.mm,
    trap_length_h: u.Quantity[u.cm] = 3.6 * u.mm,
    trap_length_e: u.Quantity[u.cm] = 240 * u.mm,
    damage_layer_depth: u.Quantity[u.cm] = 0.44 * u.um,
    r0: float = 0.8,
    n_layers: int = 1000,
):
    r"""
    Hole-tailing matrix, a port of STIX-GSW ``stx_tailing_matrix.pro`` as ``stx_build_drm`` calls it.

    IDL builds it on the photon bin means and applies it along the photon axis
    (``eloss_mat # tailing_matrix``), so it depends on the photon grid and is rebuilt here for
    each product's grid.

    The detector of thickness :math:`d` is split into depth layers, with a finer grid over the front
    damage layer. A photon absorbed at depth :math:`x` (measured from the cathode) produces a signal
    reduced by the charge collection efficiency given by the Hecht relation for hole and electron
    trapping, multiplied by a surface damage term:

    .. math::

        \eta(x) = \left[\frac{\lambda_h}{d}\left(1 - e^{-x/\lambda_h}\right)
                  + \frac{\lambda_e}{d}\left(1 - e^{-(d - x)/\lambda_e}\right)\right]
                  \left(1 - r_0\,e^{-x/\delta}\right)

    where :math:`\lambda_h` and :math:`\lambda_e` are the hole and electron trapping lengths,
    :math:`r_0` the fractional charge loss at the front surface and :math:`\delta` the damage layer depth.
    The probability that a photon of energy :math:`E` that interacts in the detector does so in
    layer :math:`i`, centred at :math:`\bar{x}_i` with thickness :math:`\Delta x_i`, is

    .. math::

        P_i(E) = \frac{e^{-\mu(E)\,\bar{x}_i}\left(1 - e^{-\mu(E)\,\Delta x_i}\right)}{1 - e^{-\mu(E)\,d}}

    with :math:`\mu(E)` the linear attenuation coefficient, evaluated midway between adjacent photon
    bin means. For each layer, a fraction :math:`P_i(E_j)` of source bin :math:`j` is mapped onto the
    interval :math:`[\eta(\bar{x}_i) E_j, \eta(\bar{x}_i) E_{j+1}]` and split between the bins
    containing its two ends in proportion to the overlap; any part falling below the lowest bin is
    lost. Since :math:`\eta \le 1`, counts are only lost or shifted to lower energies.

    Parameters
    ----------
    ph_edges : `~astropy.units.Quantity`
        Photon bin edges (the product's grid), length ``n + 1`` for ``n`` photon bins. Energy units.
    xsec_energy : `~astropy.units.Quantity`
        Energies at which ``xsec`` is tabulated. Energy units.
    xsec : `~astropy.units.Quantity`
        CdTe photoelectric + incoherent linear attenuation coefficient :math:`\mu` (``det_xsec`` 'PE' + 'SI'),
        interpolated log-log. Units of inverse length.
    depth : `~astropy.units.Quantity`, optional
        Detector thickness :math:`d`. Default 1 mm.
    trap_length_h : `~astropy.units.Quantity`, optional
        Hole trapping length :math:`\lambda_h` (mean drift length before trapping). Default 3.6 mm.
    trap_length_e : `~astropy.units.Quantity`, optional
        Electron trapping length :math:`\lambda_e`. Default 240 mm.
    damage_layer_depth : `~astropy.units.Quantity`, optional
        Characteristic depth :math:`\delta` of the front surface damage layer. Default 0.44 µm.
    r0 : float, optional
        Fractional charge loss at the front surface :math:`r_0` (dimensionless, 0 to 1). Default 0.8.
    n_layers : int, optional
        Number of depth layers across the full detector thickness. The damage region (the front
        :math:`10\,\delta`) is sampled with ``2 * n_layers`` finer layers. Default 1000.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, n)``, ``T[dest, src]`` over photon bins; apply to a (photon, count) matrix
        as ``T.T @ drm``.

    See Also
    --------
    tailing_matrix_corrected : The same model with the discretisation errors below fixed.

    Notes
    -----
    This reproduces the IDL implementation, including its discretisation errors, so results can be
    compared directly with STIX-GSW:

    * :math:`P_i` uses the attenuation to the middle of the layer, :math:`e^{-\mu \bar{x}_i}`, rather than
      to its top, :math:`e^{-\mu x_i}`, which loses up to about 2% of the counts at energies absorbed in the
      bulk layers. The last layer, :math:`[d - d/n, d]`, is also omitted.
    * Bins are built from the bin means rather than the edges, so each source bin is spread over the
      interval between its own mean and that of the next bin. This shifts the response by half a bin,
      leaves the highest photon bin with no entries (its column is zero), and gives errors of up to
      about 0.1 in individual entries on grids with uneven bin widths.
    * If the interval spans more than two bins, the bins in between receive nothing.
    """
    ph_edges = ph_edges.to_value(u.keV)
    energy = 0.5 * (ph_edges[1:] + ph_edges[:-1])  # IDL passes the photon bin means
    nen = energy.size
    tm = np.zeros((nen, nen))  # tm[src, dest], as in IDL

    # detector layers, with the finer damage layer at the front
    # lengths in um, as in IDL
    d = depth.to_value(u.um)
    dl = damage_layer_depth.to_value(u.um)
    trap_length_h = trap_length_h.to_value(u.um)
    trap_length_e = trap_length_e.to_value(u.um)
    x = d * np.arange(n_layers) / n_layers
    t = 10 * dl * np.arange(2 * n_layers) / (2 * n_layers)
    x = np.concatenate([t, x[x >= 10 * dl]])
    h = (trap_length_h * (1 - np.exp(-x / trap_length_h)) + trap_length_e * (1 - np.exp(-(d - x) / trap_length_e))) / d
    h = h * (1 - r0 * np.exp(-x / dl))  # charge collection efficiency per layer

    emin = 0.5 * (energy[1:] + energy[:-1])
    log_xsec = np.interp(np.log(emin), np.log(xsec_energy.to_value(u.keV)), np.log(xsec.to_value(1 / u.um)))
    stot = np.exp(log_xsec)  # 1/um
    mx, dx = 0.5 * (x[1:] + x[:-1]), np.diff(x)

    j = np.arange(nen - 1)
    for i in range(x.size - 1):
        f = energy * h[i]
        pslice = np.exp(-stot * mx[i]) * (1 - np.exp(-stot * dx[i])) / (1 - np.exp(-stot * d))
        g0 = np.searchsorted(energy, f[:-1], side="right") - 1  # IDL value_locate
        g1 = np.searchsorted(energy, f[1:], side="right") - 1
        width = f[1:] - f[:-1]

        same = (g0 == g1) & (g0 >= 0)
        tm[j[same], g0[same]] += pslice[same]

        low = (g0 != g1) & (g0 < 0)
        tm[j[low], g1[low]] += np.abs((f[1:][low] - energy[g1[low]]) / width[low]) * pslice[low]

        split = (g0 != g1) & (g0 >= 0)
        tm[j[split], g0[split]] += np.abs((f[:-1][split] - energy[g1[split]]) / width[split]) * pslice[split]
        tm[j[split], g1[split]] += np.abs((f[1:][split] - energy[g1[split]]) / width[split]) * pslice[split]

    return tm.T


@u.quantity_input
def tailing_matrix_corrected(
    ph_edges: u.Quantity[u.keV],
    xsec_energy: u.Quantity[u.keV],
    xsec: u.Quantity[1 / u.cm],
    depth: u.Quantity[u.cm] = 1 * u.mm,
    trap_length_h: u.Quantity[u.cm] = 3.6 * u.mm,
    trap_length_e: u.Quantity[u.cm] = 240 * u.mm,
    damage_layer_depth: u.Quantity[u.cm] = 0.44 * u.um,
    r0: float = 0.8,
    n_layers: int = 1000,
):
    r"""
    Hole-tailing matrix, with the discretisation errors of `tailing_matrix` fixed.

    Uses the same physical model as `tailing_matrix` (and STIX-GSW ``stx_tailing_matrix.pro``). The
    detector of thickness :math:`d` is split into layers with edges :math:`x_i`, covering the full
    thickness, with a finer grid over the front damage layer. A photon absorbed at depth :math:`x`
    (measured from the cathode) produces a signal reduced by the charge collection efficiency given by
    the Hecht relation for hole and electron trapping, multiplied by a surface damage term:

    .. math::

        \eta(x) = \left[\frac{\lambda_h}{d}\left(1 - e^{-x/\lambda_h}\right)
                  + \frac{\lambda_e}{d}\left(1 - e^{-(d - x)/\lambda_e}\right)\right]
                  \left(1 - r_0\,e^{-x/\delta}\right)

    where :math:`\lambda_h` and :math:`\lambda_e` are the hole and electron trapping lengths,
    :math:`r_0` the fractional charge loss at the front surface and :math:`\delta` the damage layer depth.
    The probability that a photon of energy :math:`E` that interacts in the detector does so in
    layer :math:`i`, between :math:`x_i` and :math:`x_{i+1} = x_i + \Delta x_i`, is

    .. math::

        P_i(E) = \frac{e^{-\mu(E)\,x_i}\left(1 - e^{-\mu(E)\,\Delta x_i}\right)}{1 - e^{-\mu(E)\,d}}

    with :math:`\mu(E)` the linear attenuation coefficient at the photon bin mean, so that
    :math:`\sum_i P_i = 1`. For each layer, a fraction :math:`P_i(E_j)` of photon bin :math:`j`, with edges
    :math:`[e_j, e_{j+1}]`, is spread uniformly over :math:`[\eta_i e_j, \eta_i e_{j+1}]`, where
    :math:`\eta_i = \eta(\bar{x}_i)` at the layer centre, and shared between the bins it overlaps:

    .. math::

        T_{kj} = \sum_i P_i(E_j)\,
                 \frac{\max\left(0, \min(e_{k+1}, \eta_i e_{j+1}) - \max(e_k, \eta_i e_j)\right)}
                      {\eta_i (e_{j+1} - e_j)}

    Only the part falling below the lowest edge is lost. Since :math:`\eta \le 1`, counts are only lost
    or shifted to lower energies.

    Parameters
    ----------
    ph_edges : `~astropy.units.Quantity`
        Photon bin edges (the product's grid), length ``n + 1`` for ``n`` photon bins. Energy units.
    xsec_energy : `~astropy.units.Quantity`
        Energies at which ``xsec`` is tabulated. Energy units.
    xsec : `~astropy.units.Quantity`
        CdTe photoelectric + incoherent linear attenuation coefficient :math:`\mu` (``det_xsec`` 'PE' + 'SI'),
        interpolated log-log. Units of inverse length.
    depth : `~astropy.units.Quantity`, optional
        Detector thickness :math:`d`. Default 1 mm.
    trap_length_h : `~astropy.units.Quantity`, optional
        Hole trapping length :math:`\lambda_h` (mean drift length before trapping). Default 3.6 mm.
    trap_length_e : `~astropy.units.Quantity`, optional
        Electron trapping length :math:`\lambda_e`. Default 240 mm.
    damage_layer_depth : `~astropy.units.Quantity`, optional
        Characteristic depth :math:`\delta` of the front surface damage layer. Default 0.44 µm.
    r0 : float, optional
        Fractional charge loss at the front surface :math:`r_0` (dimensionless, 0 to 1). Default 0.8.
    n_layers : int, optional
        Number of depth layers across the full detector thickness. The damage region (the front
        :math:`10\,\delta`) is sampled with ``2 * n_layers`` finer layers. Default 1000.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, n)``, ``T[dest, src]`` over photon bins; apply to a (photon, count) matrix
        as ``T.T @ drm``.

    See Also
    --------
    tailing_matrix : Direct port of the IDL implementation, for comparison with STIX-GSW.
    """
    edges = ph_edges.to_value(u.keV)
    lo, hi = edges[:-1], edges[1:]
    energy = 0.5 * (lo + hi)
    nen = energy.size
    src = np.arange(nen)
    tm = np.zeros((nen, nen))  # tm[dest, src]

    # layer edges in um through the full thickness, with the finer damage layer at the front
    d = depth.to_value(u.um)
    dl = damage_layer_depth.to_value(u.um)
    trap_length_h = trap_length_h.to_value(u.um)
    trap_length_e = trap_length_e.to_value(u.um)
    x = d * np.arange(n_layers + 1) / n_layers
    t = 10 * dl * np.arange(2 * n_layers) / (2 * n_layers)
    x = np.concatenate([t, x[x >= 10 * dl]])
    mx, dx = 0.5 * (x[1:] + x[:-1]), np.diff(x)
    h = (
        trap_length_h * (1 - np.exp(-mx / trap_length_h)) + trap_length_e * (1 - np.exp(-(d - mx) / trap_length_e))
    ) / d
    h = h * (1 - r0 * np.exp(-mx / dl))  # charge collection efficiency at each layer centre

    log_xsec = np.interp(np.log(energy), np.log(xsec_energy.to_value(u.keV)), np.log(xsec.to_value(1 / u.um)))
    stot = np.exp(log_xsec)  # 1/um

    for i in range(dx.size):
        pslice = np.exp(-stot * x[i]) * (1 - np.exp(-stot * dx[i])) / (1 - np.exp(-stot * d))
        a, b = h[i] * lo, h[i] * hi
        k0 = np.searchsorted(edges, a, side="right") - 1  # bin containing the low end, -1 if below
        k1 = np.searchsorted(edges, b, side="right") - 1  # bin containing the high end
        for m in range((k1 - k0).max() + 1):
            k = k0 + m
            ok = (k >= 0) & (k <= k1) & (k < nen)
            kk = k[ok]
            overlap = np.minimum(b[ok], hi[kk]) - np.maximum(a[ok], lo[kk])
            tm[kk, src[ok]] += overlap / (b[ok] - a[ok]) * pslice[ok]

    return tm
