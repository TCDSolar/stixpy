import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import ConciseDateFormatter

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from astropy.time import Time
from astropy.visualization import quantity_support

from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, get_earth
from sunpy.time import TimeRange

from stixpy.product.product import GenericProduct

__all__ = ["FlareList"]

quantity_support()

_GOES_LETTER_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
_GOES_RE = re.compile(r"([ABCMX])(\d+\.?\d*)", re.IGNORECASE)
_GOES_NORM = LogNorm(vmin=1e-8, vmax=1e-3)


def _goes_class_to_flux(goes_class_arr):
    """Convert GOES class strings (e.g. ``'C5.4'``, ``'M1'``) to W m⁻² flux values."""
    result = np.full(len(goes_class_arr), np.nan)
    for i, cls in enumerate(goes_class_arr):
        m = _GOES_RE.match(str(cls).strip())
        if m:
            result[i] = _GOES_LETTER_BASE[m.group(1).upper()] * float(m.group(2))
    return result


class FlareList(GenericProduct):
    """
    STIX L3 Flare List product with reconstructed source locations.

    Examples
    --------
    >>> from stixpy.data import test
    >>> from stixpy.product import Product
    >>> fl = Product(test.STIX_L3_FLARELIST_SEP2024)
    >>> fl
    FlareList
       <sunpy.time.timerange.TimeRange object at ...
        Start: 2024-09-01 00:06:44
        End:   2024-10-01 00:47:10
        Center:2024-09-16 00:26:57
        Duration:30.028079074074075 days or
               720.6738977777778 hours or
               43240.433866666666 minutes or
               2594426.032 seconds
    <BLANKLINE>
        2360 flares
    """

    def __init__(self, *, meta, control, data, energies=None, idb_versions=None):
        super().__init__(meta=meta, control=control, data=data, energies=energies, idb_versions=idb_versions)
        self._add_heliographic_coordinates()

    def _add_heliographic_coordinates(self):
        if "location_time_UTC" not in self.data.colnames:
            return
        obstime = self.data["location_time_UTC"]
        if "location_icrs" in self.data.colnames:
            self.data["location_hgs"] = self.data["location_icrs"].transform_to(HeliographicStonyhurst(obstime=obstime))
        if "solo_location_icrs" in self.data.colnames:
            self.data["solo_location_hgs"] = self.data["solo_location_icrs"].transform_to(
                HeliographicStonyhurst(obstime=obstime)
            )

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        level = meta.get("LEVEL", "").strip()
        service_subservice_ssid = tuple(meta.get(name, -1) for name in ["STYPE", "SSTYPE", "SSID"])
        return level == "L3" and service_subservice_ssid == (0, 0, 3)

    @property
    def start_time(self) -> Time:
        return self.data["start_UTC"]

    @property
    def end_time(self) -> Time:
        return self.data["end_UTC"]

    @property
    def peak_time(self) -> Time:
        return self.data["peak_UTC"]

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(self.start_time[0], self.end_time[-1])

    @property
    def duration(self) -> u.Quantity:
        return self.data["duration"]

    @property
    def flare_id(self):
        return self.data["flare_id"]

    @property
    def goes_class(self):
        return self.data["GOES_class"]

    @property
    def goes_flux(self) -> u.Quantity:
        return self.data["GOES_flux"]

    @property
    def attenuator_in(self):
        return self.data["att_in"]

    @property
    def visible_from_earth(self):
        return self.data["visible_from_earth"]

    @property
    def flare_location(self) -> SkyCoord:
        """Flare location in ICRS. Entries without a solution have NaN coordinates."""
        return self.data["location_icrs"]

    @property
    def flare_location_hgs(self) -> SkyCoord:
        """Flare location in HeliographicStonyhurst. Entries without a solution have NaN coordinates."""
        return self.data["location_hgs"]

    @property
    def solo_location_hgs(self) -> SkyCoord:
        """Solar Orbiter location in HeliographicStonyhurst at the time of location determination."""
        return self.data["solo_location_hgs"]

    @property
    def energies(self):
        return self._energies

    def concatenate(self, others):
        """
        Concatenate two or more FlareList products into one.

        Parameters
        ----------
        others : `FlareList` or list of `FlareList`

        Returns
        -------
        `FlareList`
        """
        others = others if isinstance(others, list) else [others]
        if not all(isinstance(o, type(self)) for o in others):
            raise TypeError(f"Can only concatenate {type(self).__name__} with objects of the same type")

        tables = []
        for obj in [self] + others:
            t = obj.data.copy()
            for key in ("DATASUM", "CHECKSUM"):
                t.meta.pop(key, None)
            # Strip derived HGS columns — vstack can't reconcile per-row obstime
            # across tables of different lengths; __init__ recomputes them from
            # the combined location_icrs / solo_location_icrs columns.
            for col in ("location_hgs", "solo_location_hgs"):
                if col in t.colnames:
                    t.remove_column(col)
            tables.append(t)
        combined = vstack(tables)

        return type(self)(meta=self.meta, control=None, data=combined, energies=self.energies)

    def plot_locations(self, observer="earth", axes=None, **scatter_kwargs):
        """
        Plot flare locations on sky.

        Parameters
        ----------
        observer : {"earth", "solo", "hgs"}
            Reference frame / point of view:

            * ``"hgs"``   – Heliographic Stonyhurst lon/lat
            * ``"earth"`` – Helioprojective (Tx/Ty) as seen from Earth
            * ``"solo"``  – Helioprojective (Tx/Ty) as seen from Solar Orbiter
        axes : `matplotlib.axes.Axes`, optional
        **scatter_kwargs :
            Passed to `~matplotlib.axes.Axes.scatter`.

        Returns
        -------
        `matplotlib.axes.Axes`
        """
        if axes is None:
            _, axes = plt.subplots()

        theta = np.linspace(0, 2 * np.pi, 300)
        scatter_kwargs.setdefault("s", 8)
        scatter_kwargs.setdefault("alpha", 0.5)
        scatter_kwargs.pop("color", None)  # c= takes precedence; drop colour kwarg if set

        goes_flux = _goes_class_to_flux(self.data["goes_max_class_est"])

        if observer == "hgs":
            loc = self.flare_location_hgs
            lon = loc.lon.deg
            lat = loc.lat.deg
            valid = np.isfinite(lon) & np.isfinite(lat)
            sc = axes.scatter(
                lon[valid], lat[valid], c=goes_flux[valid], norm=_GOES_NORM, cmap="plasma", **scatter_kwargs
            )
            for x in (-90, 90):
                axes.axvline(
                    x, color="k", ls="--", lw=0.8, alpha=0.4, label="Earth-facing limb (±90° lon)" if x == 90 else None
                )
            axes.set_xlabel("HGS longitude [deg]")
            axes.set_ylabel("HGS latitude [deg]")
            axes.set_xlim(-180, 180)
            axes.set_ylim(-90, 90)

        elif observer in ("earth", "solo"):
            loc_hgs = self.flare_location_hgs
            valid = np.isfinite(loc_hgs.lon.deg) & np.isfinite(loc_hgs.lat.deg)

            if observer == "solo":
                solo_hgs = self.solo_location_hgs
                valid &= np.isfinite(solo_hgs.lon.deg)

            ot = self.data["location_time_UTC"][valid]

            if observer == "earth":
                obs = get_earth(ot)
                rsun_arcsec = 960.0
            else:
                obs = solo_hgs[valid]
                rsun_arcsec = float(np.nanmedian(self.data["sun_disc_size"][valid].to_value(u.arcsec)))

            hpc = loc_hgs[valid].transform_to(Helioprojective(observer=obs, obstime=ot))
            tx = hpc.Tx.arcsec
            ty = hpc.Ty.arcsec

            sc = axes.scatter(tx, ty, c=goes_flux[valid], norm=_GOES_NORM, cmap="plasma", **scatter_kwargs)
            axes.plot(
                rsun_arcsec * np.cos(theta),
                rsun_arcsec * np.sin(theta),
                "k-",
                lw=1,
                alpha=0.5,
                label="Solar limb",
            )
            axes.set_xlabel("Tx [arcsec]")
            axes.set_ylabel("Ty [arcsec]")
            lim = max(1500.0, rsun_arcsec * 1.6)
            axes.set_xlim(-lim, lim)
            axes.set_ylim(-lim, lim)

        else:
            raise ValueError(f"observer must be 'earth', 'solo', or 'hgs', got {observer!r}")

        cbar = plt.colorbar(sc, ax=axes)
        cbar.set_label("GOES max class [W m⁻²]")
        cbar.set_ticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
        cbar.set_ticklabels(["A", "B", "C", "M", "X"])

        axes.set_aspect("equal")
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)
        axes.set_title(f"Flare locations (n={valid.sum()}) — {observer}")

        return axes

    def plot_timeseries(self, axes=None, **scatter_kwargs):
        """
        Scatter plot of per-flare peak counts per energy band vs peak time.

        A scatter (rather than line) plot is used because flares are discrete
        events — connecting them with a line would imply a continuity that
        doesn't exist.

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`, optional
        **scatter_kwargs :
            Passed to `~matplotlib.axes.Axes.scatter`.

        Returns
        -------
        list of `matplotlib.collections.PathCollection`
        """
        if axes is None:
            _, axes = plt.subplots()

        scatter_kwargs.setdefault("s", 10)
        scatter_kwargs.setdefault("alpha", 0.6)

        lc_peak = self.data["lc_peak"]
        times = self.peak_time.to_datetime()

        collections = []
        for i, row in enumerate(self.energies):
            label = f"{row['e_low'].value:.0f}–{row['e_high'].value:.0f} keV"
            sc = axes.scatter(times, lc_peak[:, i], label=label, **scatter_kwargs)
            collections.append(sc)

        axes.set_yscale("log")
        axes.set_ylabel(f"Peak counts [{lc_peak.unit}]")
        axes.legend()
        axes.xaxis.set_major_formatter(ConciseDateFormatter(axes.xaxis.get_major_locator()))

        return collections

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return type(self)(
            meta=self.meta,
            control=None,
            data=self.data[item],
            energies=self.energies,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}\n    {len(self)} flares"
