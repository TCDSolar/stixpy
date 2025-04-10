import astropy.units as u
from astropy.time import Time
from astropy.units import Quantity
from sunpy.time import TimeRange

from stixpy.product.product import L1Product

__all__ = ["ANCProduct", "Ephemeris"]


class ANCProduct(L1Product):
    """
    Basic ANC
    """

    @property
    def time(self) -> Time:
        return self.data["time"]

    @property
    def exposure_time(self) -> Quantity[u.s]:
        return self.data["timedel"].to(u.s)

    @property
    def time_range(self) -> TimeRange:
        """
        A `sunpy.time.TimeRange` for the data.
        """
        return TimeRange(self.time[0] - self.exposure_time[0] / 2, self.time[-1] + self.exposure_time[-1] / 2)


class Ephemeris(ANCProduct):
    """
    Ephemeris data in daily files normal with 64s time resolution
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO remove this when we have a better solution
        # Shift ANC data by half a time bin (starting time vs. bin centre)
        self.data["time_end"] = self.data["time"] + 32 * u.s
        self.data["time"] = self.data["time"] - 32 * u.s
        self.data["timedel"] = 64 * u.s
        self.data["time_utc"] = [Time(t, format="isot", scale="utc") for t in self.data["time_utc"]]

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data meach Raw Pixel Data"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (0, 0, 1) and level == "ANC":
            return True

    def __repr__(self):
        return f"{self.__class__.__name__}\n" f"    {self.time_range}"
