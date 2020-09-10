import astropy.units as u

from astropy.time.core import TimeDelta
from sunpy.net.dataretriever import GenericClient
from sunpy.time import TimeRange
from sunpy.util.scraper import Scraper

__all__ = ["STIXClient"]

BASE_PATTERN = 'http://pub023.cs.technik.fhnw.ch/data/new/' \
               '{level}/%Y/%m/%d/{type}/solo_{level}_stix-{product}_%Y%m%d_V01.fits'


class STIXClient(GenericClient):
    """
    Provides access to SolarOrbite STIX data

    Searches data hosted by the `FHNW <http://http://pub023.cs.technik.fhnw.ch/data/>`__.

    Examples
    --------

    """

    def _get_url_for_timerange(self, timerange, **kwargs):
        """
        Returns list of URLS corresponding to value of input timerange.
        Parameters
        ----------
        timerange: `sunpy.time.TimeRange`
            time range for which data is to be downloaded.
        Returns
        -------
        urls : list
            list of URLs corresponding to the requested time range
        """
        # TODO hard coded for now, not sure how to handel unknow url for L0 and L1 science data
        stix = Scraper(BASE_PATTERN, level='L1', type='QL', product='ql-background')
        return stix.filelist(timerange)

    def _get_time_for_url(self, urls):
        freq = urls[0].split('/')[-1][0:3]  # extract the frequency label
        crawler = Scraper(BASE_PATTERN, level='L1', type='QL', product='ql-background')
        times = list()
        for url in urls:
            t0 = crawler._extractDateURL(url)
            # hard coded full day as that's the normal.
            times.append(TimeRange(t0, t0 + TimeDelta(1*u.day)))
        return times

    def _makeimap(self):
        """
        Helper Function used to hold information about source.
        """
        self.map_['source'] = 'FHNW'
        self.map_['provider'] = 'FHNW'
        self.map_['instrument'] = 'STIX'
        self.map_['physobs'] = 'X-ray'

    @classmethod
    def _can_handle_query(cls, *query):
        """
        Answers whether client can service the query.
        Parameters
        ----------
        query : list of query objects
        Returns
        -------
        boolean
            answer as to whether client can service the query
        """
        # Import here to prevent circular imports
        from sunpy.net import attrs as a

        required = {a.Time, a.Instrument}
        optional = {a.Level,}
        if not cls.check_attr_types_in_query(query, required, optional):
            return False

    @classmethod
    def register_values(cls):
        from sunpy.net import attrs
        adict = {attrs.Instrument: [('STIX',
                                     ('Spectrometer/Telescope for Imaging X-rays'))]}
        return adict