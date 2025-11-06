import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from numpy.typing import NDArray
from sunpy.time import TimeRange

from stixpy.coordinates.frames import STIXImaging
from stixpy.product.product import L1Product

THERMAL_INDEX_MAP = np.array(["None", "Minor", "Small", "Moderate", "Major"])
FLUX_INDEX_MAP = np.array(["None", "Weak", "Significant"])
LOCATION_INDEX_MAP = np.array(["None", "Previous", "New"])
TM_PROCESSING_STATUS = {
    0x01: "Imaging Started",
    0x02: "Imaging Ended",
    0x04: "Spectroscopy Started",
    0x08: "Spectroscopy Ended",
    0x10: "Publication Started",
    0x20: "Publication Ended",
    0x40: "Cancelled",
    0x80: "New Entry",
}

__all__ = ["QuickLookProduct", "QLLightCurve", "QLBackground", "QLVariance", "QLFlareFlag", "QLLightCurve"]


class QuickLookProduct(L1Product):
    """
    Basic science data class
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


class QLLightCurve(QuickLookProduct):
    """
    Quicklook Lightcurves nominal in 5 energy bins every 4s
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data match"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 30) and level == "L1":
            return True

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}"


class QLBackground(QuickLookProduct):
    """
    Quicklook Background nominal in 5 energy bins every 8s
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data match"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 31) and level == "L1":
            return True

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}"


class QLVariance(QuickLookProduct):
    """
    Quicklook variance nominally in 5 energy bins every 8s
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data match"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 33) and level == "L1":
            return True

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}"


class QLFlareFlag(QuickLookProduct):
    """
    Quicklook Flare flag and location
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data match"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 34) and level == "L1":
            return True

    @property
    def thermal_index(self):
        r"""
        Flare thermal index - amount of thermal emission
        """
        return THERMAL_INDEX_MAP[self.data["thermal_index"]]

    @property
    def flare_location(self) -> SkyCoord:
        r"""
        Flare location as`

        Notes
        -----
        From STIX-TN-0109-FHNW_I3R3 `YLOS = -Ysc = -Yint` and `ZLOS = Zsc = Xint`

        """
        return SkyCoord(
            self.data["loc_z"] * u.arcmin, -self.data["loc_y"] * u.arcmin, frame=STIXImaging(obstime=self.data["time"])
        )

    @property
    def non_thermal_index(self) -> NDArray[str]:
        r"""
        Flare non-thermal index - significant non-thermal emission
        """
        return THERMAL_INDEX_MAP[self.data["non_thermal_index"]]

    @property
    def location_flag(self) -> NDArray[str]:
        r"""
        Flare location flag
        """
        return THERMAL_INDEX_MAP[self.data["location_status"]]

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}"


class QLTMStatusFlareList(QuickLookProduct):
    """
    Quicklook variance nominally in 5 energy bins every 8s
    """

    @classmethod
    def is_datasource_for(cls, *, meta, **kwargs):
        """Determines if meta data match"""
        service_subservice_ssid = tuple(meta[name] for name in ["STYPE", "SSTYPE", "SSID"])
        level = meta["level"]
        if service_subservice_ssid == (21, 6, 43) and level == "L1":
            return True

    @property
    def processing_status(self):
        r"""
        Processing status
        """
        return [TM_PROCESSING_STATUS[status] for status in self.data["processing_mask"]]

    def __repr__(self):
        return f"{self.__class__.__name__}\n    {self.time_range}"
