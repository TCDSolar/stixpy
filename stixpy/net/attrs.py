import re

from sunpy.net.attr import SimpleAttr

__all__ = ["DataType", "DataProduct"]


class DataType(SimpleAttr):
    """
    Data type Quicklook (QL), science (SCI), or housekeeping (HK)
    """


class Version(SimpleAttr):
    """
    Exact version of the data file
    """

    PATTERN = re.compile(r"V(\d{2})([a-zA-Z]?)")

    def __init__(self, version: int):
        super().__init__(version)
        self.allow_uncompleted = False
        self.operator = int.__eq__

    def matches(self, ver: str) -> bool:
        match = Version.PATTERN.match(ver)
        if match is None:
            return False
        v = int(match.group(1))
        u = match.group(2)

        ver_res = self.operator(v, self.value)
        u_res = u in ["", "U"] if self.allow_uncompleted else u == ""

        return ver_res and u_res


class VersionU(Version):
    """
    min version of the data file
    """

    def __init__(self, version: int):
        super().__init__(version)
        self.allow_uncompleted = True
        self.operator = int.__eq__


class MinVersion(Version):
    """
    min version of the data file
    """

    def __init__(self, version: int):
        super().__init__(version)
        self.operator = int.__ge__


class MinVersionU(VersionU):
    """
    min version of the data file
    """

    def __init__(self, version: int):
        super().__init__(version)
        self.operator = int.__ge__


class MaxVersion(Version):
    """
    max version of the data file
    """

    def __init__(self, version: int):
        super().__init__(version)
        self.operator = int.__lt__


class MaxVersionU(VersionU):
    """
    max version of the data file
    """

    def __init__(self, version: int):
        super().__init__(version)
        self.operator = int.__lt__


class DataProduct(SimpleAttr):
    """
    Data product
    """
