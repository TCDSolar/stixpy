import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from stixpy.data import test
from stixpy.product import Product
from stixpy.product.sources.flarelist import FlareList


@pytest.fixture
def flarelist_sep():
    return Product(test.STIX_L3_FLARELIST_SEP2024)


@pytest.fixture
def flarelist_oct():
    return Product(test.STIX_L3_FLARELIST_OCT2024)


def test_flarelist_type(flarelist_sep):
    assert isinstance(flarelist_sep, FlareList)


def test_flarelist_len(flarelist_sep, flarelist_oct):
    assert len(flarelist_sep) == 2360
    assert len(flarelist_oct) == 2516


def test_flarelist_time_range(flarelist_sep):
    tr = flarelist_sep.time_range
    assert tr.start < tr.end
    assert tr.start.scale == "utc"


def test_flarelist_start_end_peak_time(flarelist_sep):
    assert isinstance(flarelist_sep.start_time, Time)
    assert isinstance(flarelist_sep.end_time, Time)
    assert isinstance(flarelist_sep.peak_time, Time)
    assert len(flarelist_sep.start_time) == len(flarelist_sep)
    assert np.all(flarelist_sep.start_time <= flarelist_sep.peak_time)
    assert np.all(flarelist_sep.peak_time <= flarelist_sep.end_time)


def test_flarelist_duration(flarelist_sep):
    dur = flarelist_sep.duration
    assert dur.unit.is_equivalent(u.s)
    assert len(dur) == len(flarelist_sep)
    assert np.all(dur > 0 * u.s)


def test_flarelist_flare_id(flarelist_sep):
    ids = flarelist_sep.flare_id
    assert len(ids) == len(flarelist_sep)
    assert ids[0] == 2409010007


def test_flarelist_goes_class(flarelist_sep):
    gc = flarelist_sep.goes_class
    assert len(gc) == len(flarelist_sep)
    assert gc[0] == "C5.4"


def test_flarelist_goes_flux(flarelist_sep):
    flux = flarelist_sep.goes_flux
    assert flux.unit.is_equivalent(u.Unit("W m-2"))
    assert len(flux) == len(flarelist_sep)


def test_flarelist_attenuator_in(flarelist_sep):
    att = flarelist_sep.attenuator_in
    assert att.dtype == bool
    assert len(att) == len(flarelist_sep)


def test_flarelist_visible_from_earth(flarelist_sep):
    vis = flarelist_sep.visible_from_earth
    assert vis.dtype == bool
    assert len(vis) == len(flarelist_sep)


def test_flarelist_flare_location(flarelist_sep):
    loc = flarelist_sep.flare_location
    assert isinstance(loc, SkyCoord)
    assert loc.frame.name == "icrs"
    assert len(loc) == len(flarelist_sep)
    assert np.isnan(loc[0].ra.deg)
    assert not np.isnan(loc[1].ra.deg)


def test_flarelist_flare_location_hgs(flarelist_sep):
    loc = flarelist_sep.flare_location_hgs
    assert isinstance(loc, SkyCoord)
    assert loc.frame.name == "heliographic_stonyhurst"
    assert len(loc) == len(flarelist_sep)
    assert np.isnan(loc[0].lon.deg)
    assert not np.isnan(loc[1].lon.deg)


def test_flarelist_solo_location_hgs(flarelist_sep):
    loc = flarelist_sep.solo_location_hgs
    assert isinstance(loc, SkyCoord)
    assert loc.frame.name == "heliographic_stonyhurst"
    assert len(loc) == len(flarelist_sep)


def test_flarelist_energies(flarelist_sep):
    energies = flarelist_sep.energies
    assert energies is not None
    assert "e_low" in energies.colnames
    assert "e_high" in energies.colnames


def test_flarelist_repr(flarelist_sep):
    r = repr(flarelist_sep)
    assert "FlareList" in r
    assert "2360 flares" in r


def test_flarelist_level(flarelist_sep):
    assert flarelist_sep.level.strip() == "L3"


def test_flarelist_getitem_mask(flarelist_sep):
    mask = np.zeros(len(flarelist_sep), dtype=bool)
    mask[:5] = True
    sub = flarelist_sep[mask]
    assert isinstance(sub, FlareList)
    assert len(sub) == 5
    assert sub.flare_id[0] == flarelist_sep.flare_id[0]


def test_flarelist_getitem_slice(flarelist_sep):
    sub = flarelist_sep[10:20]
    assert isinstance(sub, FlareList)
    assert len(sub) == 10
    assert sub.flare_id[0] == flarelist_sep.flare_id[10]


def test_flarelist_concatenate(flarelist_sep, flarelist_oct):
    combined = flarelist_sep.concatenate(flarelist_oct)
    assert isinstance(combined, FlareList)
    assert len(combined) == len(flarelist_sep) + len(flarelist_oct)
    assert combined.time_range.start == flarelist_sep.time_range.start
    assert combined.time_range.end == flarelist_oct.time_range.end


def test_flarelist_concatenate_type_error(flarelist_sep):
    with pytest.raises(TypeError):
        flarelist_sep.concatenate("not a flarelist")


def test_flarelist_plots(flarelist_sep, tmp_path):
    import matplotlib.pyplot as plt

    collections = flarelist_sep.plot_timeseries()
    assert len(collections) == len(flarelist_sep.energies)

    out = tmp_path / "flarelist_timeseries.png"
    plt.savefig(out)
    plt.close()
    assert out.stat().st_size > 10

    for observer in ("hgs", "earth", "solo", "carrington"):
        ax = flarelist_sep.plot_locations(observer=observer)
        assert ax is not None

        out = tmp_path / f"flarelist_locations_{observer}.png"
        plt.savefig(out)
        plt.close()
        assert out.stat().st_size > 10

    # Carrington-specific checks: axis limits cover the whole rotating Sun and
    # the scatter is non-empty so users actually see something on the test data.
    ax = flarelist_sep.plot_locations(observer="carrington")
    assert ax.get_xlim() == (0, 360)
    assert ax.get_ylim() == (-90, 90)
    assert any(c.get_offsets().shape[0] > 0 for c in ax.collections)
    plt.close()


def test_flarelist_plot_locations_invalid_observer(flarelist_sep):
    with pytest.raises(ValueError, match="observer must be"):
        flarelist_sep.plot_locations(observer="invalid")
