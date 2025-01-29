import re

import numpy as np
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever.client import QueryResponse

__all__ = ["DataType", "DataProduct"]


class DataType(SimpleAttr):
    """
    Data type Quicklook (QL), science (SCI), or housekeeping (HK)
    """


class Version(SimpleAttr):
    """
    exact version of the data file
    """

    PATTERN = re.compile(r"V([0-9][0-9])([a-zA-Z]?)")

    def __init__(self, version: int, allow_uncompleted=True):
        super().__init__(version)
        self.allow_uncompleted = allow_uncompleted
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


class MinVersion(Version):
    """
    min version of the data file
    """

    def __init__(self, version: int, allow_uncompleted=True):
        super().__init__(version, allow_uncompleted=allow_uncompleted)
        self.operator = int.__ge__


class MaxVersion(Version):
    """
    max version of the data file
    """

    def __init__(self, version: int, allow_uncompleted=True):
        super().__init__(version, allow_uncompleted=allow_uncompleted)
        self.operator = int.__lt__


class LatestVersion:
    """
    latest version of the data file
    """

    def __init__(self, *, allow_uncompleted=True):
        self.allow_uncompleted = allow_uncompleted

    def filter(self, res_table: QueryResponse):
        res_table["_num_version"] = 0
        for i, row in enumerate(res_table):
            match = Version.PATTERN.match(row["Ver"])
            if match is None:
                res_table.remove_row(i)
                continue
            v = int(match.group(1))
            u = match.group(2)
            if u not in ["", "U"] if self.allow_uncompleted else u != "":
                res_table.remove_row(i)
                continue
            row["_num_version"] = v
        grouped_res = res_table.group_by(
            ["Start Time", "End Time", "Instrument", "Level", "DataType", "DataProduct", "Request ID"]
        )
        maxv = grouped_res["_num_version"].groups.aggregate(np.argmax)
        res_table.remove_rows(range(len(res_table)))
        for key, group in zip(maxv, grouped_res.groups):
            res_table.add_row(group[key])
        res_table.remove_column("_num_version")
        return res_table


class DataProduct(SimpleAttr):
    """
    Data product
    """
