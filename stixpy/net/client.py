from datetime import datetime

import astropy.units as u

from astropy.time.core import TimeDelta
from sunpy.net.dataretriever import GenericClient
from sunpy.net import attrs as a
from sunpy.net.dataretriever.client import QueryResponse
from sunpy.time import TimeRange, parse_time
from sunpy.util.scraper import Scraper

__all__ = ["STIXClient"]


class STIXClient(GenericClient):
    baseurl = (r'http://pub023.cs.technik.fhnw.ch/data/new/'
               r'{level}/{year:4d}/{month:02d}/{day:02d}/{datatype}/')
    ql_filename = r'solo_{level}_stix-{product}_\d{{8}}_V\d{{2}}.fits'
    sci_filename = (r'solo_{level}_stix-{product}-\d{{8}}_'
                    r'\d{{8}}T\d{{6}}-\d{{8}}T\d{{6}}_V\d{{2}}_\d{{5}}.fits')

    base_pattern = r'{}/{Level}/{year:4d}/{month:02d}/{day:02d}/{DataType}/'
    ql_pattern = r'solo_{Level}_{descriptor}_{time}_{ver}.fits'
    sci_pattern = r'solo_{Level}_{descriptor}_{start}-{end}_{ver}_{tc}.fits'

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
        level = matchdict['Level'][0]

        metalist = []
        for date in matchdict['Time'].get_dates():
            year = date.datetime.year
            month = date.datetime.month
            day = date.datetime.day
            for datatype in matchdict['DataType']:
                products = [p for p in matchdict['DataProduct'] if p.startswith(datatype.lower())]
                for product in products:
                    if datatype == 'QL' and product.startswith('ql'):
                        url = self.baseurl + self.ql_filename
                        pattern = self.base_pattern + self.ql_pattern
                    elif datatype == 'SCI' and product.startswith('sci'):
                        url = self.baseurl + self.sci_filename
                        pattern = self.base_pattern + self.sci_pattern

                    url = url.format(level=level, year=year, month=month, day=day,
                                     datatype=datatype, product=product.replace('_', '-'))

                    scraper = Scraper(url, regex=True)
                    filesmeta = scraper._extract_files_meta(matchdict['Time'], extractor=pattern)
                    for i in filesmeta:
                        rowdict = self.post_search_hook(i, matchdict)
                        metalist.append(rowdict)

        return QueryResponse(metalist, client=self)

    def post_search_hook(self, exdict, matchdict):
        rowdict = super().post_search_hook(exdict, matchdict)
        if rowdict.get('DataType') == 'QL':
            product = rowdict.pop('descriptor').strip('stix-')
            rowdict['DataProduct'] = product
            rowdict['Request ID'] = '-'
            rowdict.pop('ver')
            rowdict.pop('time')
        elif rowdict.get('DataType') == 'SCI':
            descriptor = rowdict.pop('descriptor')
            *product, req_id = descriptor.split('-')
            product = '-'.join(product[1:])
            rowdict['DataProduct'] = product
            rowdict['Request ID'] = int(req_id)
            ts = rowdict.pop('start')
            te = rowdict.pop('end')
            tr = TimeRange(ts, te)
            rowdict['Time'] = tr
            rowdict['Start Time'] = tr.start.iso
            rowdict['End Time'] = tr.end.iso
            rowdict.pop('ver')
            rowdict.pop('tc')
        return rowdict

    @classmethod
    def _attrs_module(cls):
        return 'stix', 'stixpy.net.attrs'

    @classmethod
    def register_values(cls):
        from sunpy.net import attrs
        adict = {attrs.Instrument: [('STIX', 'Spectrometer/Telescope for Imaging X-rays')],
                 attrs.Level: [('L1', 'STIX: Decommutated, uncompressed, uncalibrated data.')],
                 attrs.stix.DataType: [('QL', 'Quick Look'), ('SCI', 'Science Data')],
                 attrs.stix.DataProduct: [('ql_lightcurve', 'Quick look light curve'),
                                          ('ql_background', 'Quick look background light curve'),
                                          ('ql_variance', 'Quick look variance curve'),
                                          ('ql_spectra', 'Quick look spectra'),
                                          ('ql_calibration_spectrum', 'Quick look energy '
                                                                      'calibration spectrum'),
                                          ('ql_flareflag',
                                           'Quick look energy calibration spectrum'),
                                          ('sci_xray_l0', 'Uncompressed pixel counts (12)'),
                                          ('sci_xray_l1', 'Compressed pixel counts (12)'),
                                          ('sci_xray_l2', 'Compressed summed pixel counts (4)'),
                                          ('sci_xray_l3', 'Compressed visibilities'),
                                          ('sci_spectrogram', 'spectrogram')]
                 }
        return adict
