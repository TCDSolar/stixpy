import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective

from stixpy.coordinates.fov import FOV_LIMITS, get_illumination_status
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.visualisation.plotters import plot_fov


@pytest.fixture(autouse=True)
def _close_figures():
    # plot_fov draws on the current pyplot figure, close it so later tests (e.g. Map.plot) do not reuse it
    yield
    plt.close("all")


def test_fov_limits_match_paper():
    # Massa et al. 2023, Appendix C: columns 0.633 deg and 1.056 deg, rows 0.518 deg and 0.960 deg
    assert u.allclose(FOV_LIMITS["full"], [0.633, 0.518] * u.deg, atol=0.001 * u.deg)
    assert u.allclose(FOV_LIMITS["partial"], [1.056, 0.960] * u.deg, atol=0.001 * u.deg)


def test_get_illumination_status():
    x = [0, 2000, 0, 3000, 4000, 0] * u.arcsec
    y = [0, -1900, 2500, 3000, 0, -4000] * u.arcsec
    status, pixels = get_illumination_status(SkyCoord(x, y, frame=STIXImaging))
    assert np.all(status == ["full", "single row", "single row", "partial", "outside", "outside"])
    assert pixels.tolist() == ["top+bot", "top", "bot", "bot", None, None]


def test_get_illumination_status_needs_obstime():
    with pytest.raises(ValueError, match="obstime"):
        get_illumination_status(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=Helioprojective))


def test_plot_fov_needs_obstime():
    # the plot needs Solar Orbiter's position, which needs a time
    with pytest.raises(ValueError, match="obstime"):
        plot_fov(SkyCoord(100 * u.arcsec, 200 * u.arcsec, frame=STIXImaging))


def _world(ax, label):
    # plot_coord lines hold world data with their own transform, go through pixels to the axes WCS
    (line,) = [line for line in ax.lines if line.get_label() == label]
    pix = ax.transData.inverted().transform(line.get_transform().transform(line.get_xydata()))
    return ax.wcs.pixel_to_world(pix[:, 0], pix[:, 1])


@pytest.mark.remote_data
def test_plot_fov_hpc():
    # 2022-10-09 east limb flare, seen by STIX at about STIX (1231, 2947) arcsec
    obstime = Time("2022-10-09T14:10:25.668")
    _, solo, _ = get_hpc_info(obstime)
    observer = HeliographicStonyhurst(solo, representation_type="cartesian", obstime=obstime)
    hpc = Helioprojective(observer=observer, obstime=obstime)
    flare_hpc = SkyCoord(-3027.57 * u.arcsec, 1024.27 * u.arcsec, frame=hpc)

    status, pixels = get_illumination_status(flare_hpc)
    assert status == "single row"
    assert pixels == "bot"

    # HPC input is transformed to and plotted in the STIX frame, with sunpy's limb and grid
    ax = plot_fov(flare_hpc, draw_limb=True, draw_grid=True)
    assert [t.get_text() for t in ax.texts] == ["single row (bot)"]
    stx = _world(ax, "Source")
    assert u.allclose([stx.Tx[0], stx.Ty[0]], [1230.7, 2946.5] * u.arcsec, atol=1 * u.arcsec)

    # with frame the boxes and source are plotted in HPC, off-disk box corners included
    ax = plot_fov(flare_hpc, frame=hpc, draw_limb=True)
    hpc_source = _world(ax, "Source")
    assert u.allclose([hpc_source.Tx[0], hpc_source.Ty[0]], [-3027.57, 1024.27] * u.arcsec, atol=1e-3 * u.arcsec)
    for name in ["Full FOV", "Single row FOV", "Partial FOV"]:
        assert not np.isnan(_world(ax, name).Tx).any()

    # a frame with another observer or time cannot hold the off-disk box corners
    with pytest.raises(ValueError, match="Solar Orbiter"):
        plot_fov(flare_hpc, frame=Helioprojective(observer="earth", obstime=obstime))
    with pytest.raises(ValueError, match="Solar Orbiter"):
        plot_fov(flare_hpc, frame=Helioprojective(observer=observer, obstime=obstime + 1 * u.s))
