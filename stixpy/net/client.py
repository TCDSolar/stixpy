import numpy as np
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
        self["_num_version"] = 0
        for i, row in enumerate(self):
            match = Version.PATTERN.match(row["Ver"])
            if match is None:
                self.remove_row(i)
                continue
            v = int(match.group(1))
            u = match.group(2)
            if u not in ["", "U"] if allow_uncompleted else u != "":
                self.remove_row(i)
                continue
            row["_num_version"] = v
        grouped_res = self.group_by(
            ["Start Time", "End Time", "Instrument", "Level", "DataType", "DataProduct", "Request ID"]
        )
        maxv = grouped_res["_num_version"].groups.aggregate(np.argmax)
        self.remove_rows(range(len(self)))
        for key, group in zip(maxv, grouped_res.groups):
            self.add_row(group[key])
        self.remove_column("_num_version")


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

    baseurl = r"https://pub099.cs.technik.fhnw.ch/data/fits/"
    datapath = r"{level}/{year:4d}/{month:02d}/{day:02d}/{datatype}/"

    ql_filename = r"solo_{level}_stix-{product}_[0-9]{{8}}_V[0-9]{{2}}[a-zA-Z]?[.]fits"
    sci_filename = (
        r"solo_{level}_stix-{product}_" r"[0-9]{{8}}T[0-9]{{6}}-[0-9]{{8}}T[0-9]{{6}}_V[0-9]{{2}}[a-zA-Z]?_.*[.]fits"
    )

    base_pattern = r"{}/{Level}/{year:4d}/{month:02d}/{day:02d}/{DataType}/"
    ql_pattern = r"solo_{Level}_{descriptor}_{time}_{Ver}.fits"
    sci_pattern = r"solo_{Level}_{descriptor}_{start}-{end}_{Ver}_{Request}-{tc}.fits"

    required = {a.Time, a.Instrument}

    def __init__(self, *, source="https://pub099.cs.technik.fhnw.ch/data/fits/") -> None:
        """Creates a Fido client to search and download STIX data from the STIX instrument archive

        Parameters
        ----------
        source : str, optional
            a url like path to alternative data source. You can provide a local filesystem path here. by default "https://pub099.cs.technik.fhnw.ch/data/fits/"
        """
        super().__init__()
        self.baseurl = source + r"{level}/{year:4d}/{month:02d}/{day:02d}/{datatype}/"

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
        levels = matchdict["Level"]

        versions = []
        for versionAttrType in [Version, VersionU, MinVersion, MinVersionU, MaxVersion, MaxVersionU]:
            if versionAttrType.__name__ in matchdict:
                for version in matchdict[versionAttrType.__name__]:
                    versions.append(versionAttrType(int(version)))

        metalist = []
        tr = TimeRange(matchdict["Start Time"], matchdict["End Time"])
        for date in tr.get_dates():
            year = date.datetime.year
            month = date.datetime.month
            day = date.datetime.day
            for level in levels:
                for datatype in matchdict["DataType"]:
                    products = [p for p in matchdict["DataProduct"] if p.startswith(datatype.lower())]
                    for product in products:
                        if datatype.lower() == "ql" and product.startswith("ql"):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        if datatype.lower() == "hk" and product.startswith("hk"):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        elif datatype.lower() == "sci" and product.startswith("sci"):
                            url = self.baseurl + self.sci_filename
                            pattern = self.base_pattern + self.sci_pattern
                        elif datatype.lower() == "cal" and product.startswith("cal"):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        elif datatype.lower() in ["asp", "aux"] and product.endswith("ephemeris"):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern

                        url = url.format(
                            level=level.upper(),
                            year=year,
                            month=month,
                            day=day,
                            datatype=datatype.upper(),
                            product=product.replace("_", "-"),
                        )

                        scraper = Scraper(url, regex=True)
                        try:
                            filesmeta = scraper._extract_files_meta(tr, extractor=pattern)

                            for i in filesmeta:
                                rowdict = self.post_search_hook(i, matchdict)

                                versionTest = True
                                for versionAttr in versions:
                                    versionTest &= versionAttr.matches(rowdict["Ver"])
                                    if not versionTest:
                                        break
                                if not versionTest:
                                    continue

                                file_tr = rowdict.pop("tr", None)
                                if file_tr is not None:
                                    # 4 cases file time full in, fully our start in or end in
                                    if file_tr.start >= tr.start and file_tr.end <= tr.end:
                                        metalist.append(rowdict)
                                    elif tr.start <= file_tr.start and tr.end >= file_tr.end:
                                        metalist.append(rowdict)
                                    elif file_tr.start <= tr.start <= file_tr.end:
                                        metalist.append(rowdict)
                                    elif file_tr.start <= tr.end <= file_tr.end:
                                        metalist.append(rowdict)
                                else:
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
            rowdict["Start Time"] = tr.start.iso
            rowdict["End Time"] = tr.end.iso
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
                ("ANC", "STIX: Ancillary Data like aspect."),
                ("L2", "STIX: Calibrated data."),
                ("ANC", "STIX: Ancillary data."),
            ],
            attrs.stix.DataType: [
                ("QL", "Quick Look"),
                ("SCI", "Science Data"),
                ("CAL", "Calibration"),
                ("ASP", "Aspect"),
                ("AUX", "will be removed when ANC is ready"),
                ("HK", "House Keeping"),
            ],
            attrs.stix.DataProduct: [
                ("hk_maxi", "House Keeping Maxi Report"),
                ("cal_energy", "Energy Calibration"),
                ("ql_lightcurve", "Quick look light curve"),
                ("ql_background", "Quick look background light curve"),
                ("ql_variance", "Quick look variance curve"),
                ("ql_spectra", "Quick look spectra"),
                ("ql_calibration_spectrum", "Quick look energy " "calibration spectrum"),
                ("ql_flareflag", "Quick look flare flag including location"),
                ("ql_tmstatusflarelist", "Quick look TM Status and flare list"),
                ("sci_xray_rpd", "Raw Pixel Data"),
                ("sci_xray_cpd", "Compressed Pixel Data"),
                ("sci_xray_scpd", "Summed Compressed Pixel Data"),
                ("sci_xray_vis", "Visibilities"),
                ("sci_xray_spec", "Spectrogram"),
                ("asp-ephemeris", "Aspect Solution and Ephemeris data"),
            ],
        }
        return adict
