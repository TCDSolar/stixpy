from sunpy.net import attrs as a
from sunpy.net.dataretriever import GenericClient
from sunpy.net.dataretriever.client import QueryResponse
from sunpy.time import TimeRange
try:
    from sunpy.net.scraper import Scraper
except ModuleNotFoundError:
    from sunpy.util.scraper import Scraper

__all__ = ["STIXClient"]


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
    baseurl = (r'https://pub099.cs.technik.fhnw.ch/data/fits/'
               r'{level}/{year:4d}/{month:02d}/{day:02d}/{datatype}/')
    ql_filename = r'solo_{level}_stix-{product}_\d{{8}}_V\d{{2}}\D?.fits'
    sci_filename = (r'solo_{level}_stix-{product}_'
                    r'\d{{8}}T\d{{6}}-\d{{8}}T\d{{6}}_V\d{{2}}\D?_.*.fits')

    base_pattern = r'{}/{Level}/{year:4d}/{month:02d}/{day:02d}/{DataType}/'
    ql_pattern = r'solo_{Level}_{descriptor}_{time}_{Ver}.fits'
    sci_pattern = r'solo_{Level}_{descriptor}_{start}-{end}_{Ver}_{Request}-{tc}.fits'

    required = {a.Time, a.Instrument}

    def search(self, *args, **kwargs):
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
        levels = matchdict['Level']

        metalist = []
        tr = TimeRange(matchdict['Start Time'], matchdict['End Time'])
        for date in tr.get_dates():
            year = date.datetime.year
            month = date.datetime.month
            day = date.datetime.day
            for level in levels:
                for datatype in matchdict['DataType']:
                    products = [p for p in matchdict['DataProduct']
                                if p.startswith(datatype.lower())]
                    for product in products:
                        if datatype.lower() == 'ql' and product.startswith('ql'):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        if datatype.lower() == 'hk' and product.startswith('hk'):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        elif datatype.lower() == 'sci' and product.startswith('sci'):
                            url = self.baseurl + self.sci_filename
                            pattern = self.base_pattern + self.sci_pattern
                        elif datatype.lower() == 'cal' and product.startswith('cal'):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern
                        elif datatype.lower() == 'aux' and product.startswith('aux'):
                            url = self.baseurl + self.ql_filename
                            pattern = self.base_pattern + self.ql_pattern

                        url = url.format(level=level.upper(), year=year, month=month, day=day,
                                         datatype=datatype.upper(),
                                         product=product.replace('_', '-'))

                        scraper = Scraper(url, regex=True)
                        filesmeta = scraper._extract_files_meta(tr, extractor=pattern)
                        for i in filesmeta:
                            rowdict = self.post_search_hook(i, matchdict)
                            file_tr = rowdict.pop('tr', None)
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

        return QueryResponse(metalist, client=self)

    def post_search_hook(self, exdict, matchdict):
        rowdict = super().post_search_hook(exdict, matchdict)
        product = rowdict.pop('descriptor')[5:]  # Strip 'sci-' from product name
        rowdict['DataProduct'] = product

        if rowdict.get('DataType') == 'SCI':
            rowdict['Request ID'] = int(rowdict['Request'])
            ts = rowdict.pop('start')
            te = rowdict.pop('end')
            tr = TimeRange(ts, te)
            rowdict['tr'] = tr
            rowdict['Start Time'] = tr.start.iso
            rowdict['End Time'] = tr.end.iso
            rowdict.pop('tc')
            rowdict.pop('Request')
        else:
            rowdict['Request ID'] = '-'
            rowdict.pop('time')

        return rowdict

    @classmethod
    def _attrs_module(cls):
        return 'stix', 'stixpy.net.attrs'

    @classmethod
    def register_values(cls):
        from sunpy.net import attrs
        adict = {attrs.Instrument: [('STIX', 'Spectrometer/Telescope for Imaging X-rays')],
                 attrs.Level: [('L0', 'STIX: Decommutated, uncompressed, uncalibrated data.'),
                               ('L1', 'STIX: Engineering and UTC time conversion .'),
                               ('L2', 'STIX: Calibrated data.')],
                 attrs.stix.DataType: [('QL', 'Quick Look'), ('SCI', 'Science Data'),
                                       ('CAL', 'Calibration'), ('AUX', 'Auxiliary'),
                                       ('HK', 'House Keeping')],
                 attrs.stix.DataProduct: [('hk_maxi', 'House Keeping Maxi Report'),
                                          ('cal_energy', 'Energy Calibration'),
                                          ('ql_lightcurve', 'Quick look light curve'),
                                          ('ql_background', 'Quick look background light curve'),
                                          ('ql_variance', 'Quick look variance curve'),
                                          ('ql_spectra', 'Quick look spectra'),
                                          ('ql_calibration_spectrum', 'Quick look energy '
                                                                      'calibration spectrum'),
                                          ('ql_flareflag',
                                           'Quick look energy calibration spectrum'),
                                          ('sci_xray_rpd', 'Raw Pixel Data'),
                                          ('sci_xray_cpd', 'Compressed Pixel Data'),
                                          ('sci_xray_scpd', 'Summed Compressed Pixel Data'),
                                          ('sci_xray_vis', 'Visibilities'),
                                          ('sci_xray_spec', 'Spectrogram'),
                                          ('aux_ephemeris', 'Auxiliary ephemeris data')]
                 }
        return adict
