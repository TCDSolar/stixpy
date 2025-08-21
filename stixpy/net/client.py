import astropy.units as u
import numpy as np
from astropy.time import Time
from sunpy.net import attrs as a
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever import GenericClient
from sunpy.net.dataretriever.client import QueryResponse
from sunpy.time import TimeRange

from stixpy.net.attrs import MaxVersion, MaxVersionU, MinVersion, MinVersionU, Version, VersionU

try:
    from sunpy.net.scraper import Scraper
except ModuleNotFoundError:
    from sunpy.util.scraper import Scraper

__all__ = ["STIXClient", "StixQueryResponse"]


class StixQueryResponse(QueryResponse):
    def filter_for_latest_version(self, allow_uncompleted=False):
        r"""
        Filter the response to only include the most recent versions of results.

        Parameters
        ----------
        allow_uncompleted
            Include incomplete version (e.g. V02U)

        """
        if len(self) > 0 and "Start Time" in self.columns:
            self["_tidx"] = range(len(self))
            grouped_res = self.group_by(
                ["Start Time", "End Time", "Instrument", "Level", "DataType", "DataProduct", "Request ID"]
            )
            keep = np.zeros(len(self), dtype=bool)
            for key, group in zip(grouped_res.groups.keys, grouped_res.groups):
                group.sort("Ver")
                if not allow_uncompleted:
                    incomplete = np.char.endswith(group["Ver"].data, "U")
                    keep[group[~incomplete][-1]["_tidx"]] = True
                else:
                    keep[group[-1]["_tidx"]] = True
            self.remove_column("_tidx")
            self.remove_rows(np.where(~keep))


class STIXClient(GenericClient):
    """
    A Fido client to search and download STIX data from the STIX instrument archive

    Examples
    --------
    >>> from sunpy.net import Fido, attrs as a
    >>> from stixpy.net.client import STIXClient
    >>> query = Fido.search(a.Time('2020-06-05', '2020-06-07'), a.Instrument.stix,
    ...                     a.stix.DataProduct.ql_lightcurve)  #doctest: +REMOTE_DATA
    >>> query  #doctest: +REMOTE_DATA
    <sunpy.net.fido_factory.UnifiedResponse object at ...>
    Results from 1 Provider:
    <BLANKLINE>
    3 Results from the STIXClient:
           Start Time               End Time        Instrument ... Ver Request ID
    ----------------------- ----------------------- ---------- ... --- ----------
    2020-06-05 00:00:00.000 2020-06-05 23:59:59.999       STIX ... V02          -
    2020-06-06 00:00:00.000 2020-06-06 23:59:59.999       STIX ... V02          -
    2020-06-07 00:00:00.000 2020-06-07 23:59:59.999       STIX ... V02          -
    <BLANKLINE>
    <BLANKLINE>
    """

    baseurl = r"https://pub099.cs.technik.fhnw.ch/data/fits"
    base_pattern = r"/{Level}/{{year:4d}}/{{month:2d}}/{{day:2d}}/{DataType}/"
    ql_pattern = r"solo_{Level}_{{descriptor}}_{{time}}_{{Ver}}.fits"
    sci_pattern = r"solo_{Level}_{{descriptor}}_{{start}}-{{end}}_{{Ver}}_{{Request}}-{{tc}}.fits"

    required = {a.Time, a.Instrument}

    def __init__(self, *, source=None) -> None:
        """Creates a Fido client to search and download STIX data from the STIX instrument archive

        Parameters
        ----------
        source : str, optional
            a url like path to alternative data source. You can provide a local filesystem path here. by default "https://pub099.cs.technik.fhnw.ch/data/fits/"
        """
        super().__init__()
        source = source if source else self.baseurl
        self.pattern = source + self.base_pattern

    def search(self, *args, **kwargs) -> StixQueryResponse:
        """
        Query this client for a list of results.

        Parameters
        ----------
        *args: `tuple`
         `sunpy.net.attrs` objects representing the query.
        **kwargs: `dict`
          Any extra keywords to refine the search.

        Returns
        -------
        A `QueryResponse` instance containing the query result.
        """
        matchdict = self._get_match_dict(*args, **kwargs)

        # Try to minimise requests by update possible type and product using the given attributes
        types_from_product = set([prod.split("_")[0].casefold() for prod in matchdict["DataProduct"]])
        datatypes = list(types_from_product.intersection([t.casefold() for t in matchdict["DataType"]]))
        products = [prod for prod in matchdict["DataProduct"] if prod.split("_")[0].casefold() in datatypes]
        matchdict.update({"DataType": datatypes, "DataProduct": products})
        levels = matchdict["Level"]

        versions = []
        for versionAttrType in [Version, VersionU, MinVersion, MinVersionU, MaxVersion, MaxVersionU]:
            if versionAttrType.__name__ in matchdict:
                for version in matchdict[versionAttrType.__name__]:
                    versions.append(versionAttrType(int(version)))

        tr = TimeRange(matchdict["Start Time"], matchdict["End Time"])

        # Because of the way the data is organised on the server products which start on one day but end on the next
        # will only be present in the "path" for the start date the longest request we practically make are ~6 hours
        # so need to extend the paths we search by ~6 hours but not alter the actual request search range
        path_tr = TimeRange(matchdict["Start Time"], matchdict["End Time"])
        if "sci" in (t.casefold() for t in matchdict["DataType"]):
            path_tr = TimeRange(Time(matchdict["Start Time"]) - 6.5 * u.h, matchdict["End Time"])

        metalist = []

        # for date in path_tr.get_dates():
        for level in levels:
            for datatype in matchdict["DataType"]:
                if datatype.lower() == "sci":
                    pattern = self.pattern + self.sci_pattern
                else:
                    pattern = self.pattern + self.ql_pattern

                pattern = pattern.format(
                    Level=level.upper(),
                    DataType=datatype.upper(),
                )
                pattern = pattern.replace(r"{", r"{{").replace(r"}", r"}}")
                scraper = Scraper(format=pattern)
                try:
                    filesmeta = scraper._extract_files_meta(path_tr)

                    products = [p.replace("_", "-") for p in matchdict["DataProduct"] if p.startswith(datatype.lower())]
                    if len(products) > 0:
                        for meta in filesmeta:
                            current_matchdict = {
                                **matchdict,
                                **{"DataType": [datatype], "DataProduct": products, "Level": [level]},
                            }
                            rowdict = self.post_search_hook(meta, current_matchdict)

                            versionTest = True
                            for versionAttr in versions:
                                versionTest &= versionAttr.matches(rowdict["Ver"])
                                if not versionTest:
                                    break
                            if not versionTest:
                                continue

                            if rowdict["DataProduct"].replace("-", "_") not in matchdict["DataProduct"]:
                                continue

                            file_tr = rowdict.pop("tr", TimeRange(rowdict["Start Time"], rowdict["End Time"]))
                            # 4 cases file time full in, fully our start in or end in
                            if file_tr.start >= tr.start and file_tr.end <= tr.end:
                                metalist.append(rowdict)
                            elif tr.start <= file_tr.start and tr.end >= file_tr.end:
                                metalist.append(rowdict)
                            elif file_tr.start <= tr.start <= file_tr.end:
                                metalist.append(rowdict)
                            elif file_tr.start <= tr.end <= file_tr.end:
                                metalist.append(rowdict)

                except FileNotFoundError:
                    continue
        return StixQueryResponse(metalist, client=self)

    @classmethod
    def _can_handle_query(cls, *query):
        """
        Method the
        `sunpy.net.fido_factory.UnifiedDownloaderFactory`
        class uses to dispatch queries to this Client.
        """
        regattrs_dict = cls.register_values()
        optional = {k for k in regattrs_dict.keys()} - cls.required
        if not cls.check_attr_types_in_query(query, cls.required, optional):
            return False
        for key in regattrs_dict:
            all_vals = [i[0].lower() for i in regattrs_dict[key]]
            for x in query:
                if isinstance(x, key) and issubclass(key, SimpleAttr) and str(x.value).lower() not in all_vals:
                    return False
        return True

    def post_search_hook(self, exdict, matchdict):
        rowdict = super().post_search_hook(exdict, matchdict)
        product = rowdict.pop("descriptor")[5:]  # Strip 'sci-' from product name
        rowdict["DataProduct"] = product

        if rowdict.get("DataType") == "SCI":
            rowdict["Request ID"] = int(rowdict["Request"])
            ts = rowdict.pop("start")
            te = rowdict.pop("end")
            tr = TimeRange(ts, te)
            rowdict["tr"] = tr
            rowdict["Start Time"] = tr.start
            rowdict["End Time"] = tr.end
            rowdict.pop("tc")
            rowdict.pop("Request")
        else:
            rowdict["Request ID"] = "-"
            rowdict.pop("time")

        return rowdict

    @classmethod
    def _attrs_module(cls):
        return "stix", "stixpy.net.attrs"

    @classmethod
    def register_values(cls):
        from sunpy.net import attrs

        adict = {
            attrs.Instrument: [("STIX", "Spectrometer/Telescope for Imaging X-rays")],
            attrs.Level: [
                ("L0", "STIX: commutated, uncompressed, uncalibrated data."),
                ("L1", "STIX: Engineering and UTC time conversion ."),
                ("L2", "STIX: Calibrated data."),
                ("ANC", "STIX: Ancillary data."),
            ],
            attrs.stix.DataType: [
                ("QL", "Quick Look"),
                ("SCI", "Science Data"),
                ("CAL", "Calibration"),
                ("ASP", "Aspect"),
                ("HK", "House Keeping"),
            ],
            attrs.stix.DataProduct: [
                ("hk_maxi", "House Keeping Maxi Report"),
                ("cal_energy", "Energy Calibration"),
                ("ql_lightcurve", "Quick look light curve"),
                ("ql_background", "Quick look background light curve"),
                ("ql_variance", "Quick look variance curve"),
                ("ql_spectra", "Quick look spectra"),
                ("ql_calibration_spectrum", "Quick look energy calibration spectrum"),
                ("ql_flareflag", "Quick look flare flag including location"),
                ("ql_tmstatusflarelist", "Quick look TM Status and flare list"),
                ("sci_xray_rpd", "Raw Pixel Data"),
                ("sci_xray_cpd", "Compressed Pixel Data"),
                ("sci_xray_scpd", "Summed Compressed Pixel Data"),
                ("sci_xray_vis", "Visibilities"),
                ("sci_xray_spec", "Spectrogram"),
                ("asp_ephemeris", "Aspect Solution and Ephemeris data"),
            ],
        }
        return adict
