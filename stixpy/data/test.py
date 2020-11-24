import os
import sys
import tarfile
from pathlib import Path

import stixpy


package_dir = Path(os.path.dirname(stixpy.__file__))
root_dir = package_dir.joinpath("data")


def _unzip_test_data():
    data_tar = root_dir / 'test_data.tar.gz'
    with tarfile.open(data_tar.as_posix(), 'r:gz') as tar:
        tar.extractall(root_dir.absolute())


_unzip_test_data()

_TEST_DATA = {
    'STIX_QL_BACKGROUND_TIMESERIES': 'solo_L1_stix-ql-background_20200505_V01.fits',
    'STIX_QL_CALIBRATION': 'solo_L1_stix-ql-calibration-spectrum_20200505_V01.fits',
    'STIX_QL_FLAREFLAG':  'solo_L1_stix-ql-flareflag_20200505_V01.fits',
    'STIX_QL_LIGHTCURVE_TIMESERIES': 'solo_L1_stix-ql-lightcurve_20200505_V01.fits',
    'STIX_QL_SPECTRA': 'solo_L1_stix-ql-spectra_20200505_V01.fits',
    'STIX_QL_VARIANCE_TIMESERIES': 'solo_L1_stix-ql-variance_20200505_V01.fits',
    'STIX_SCI_SPECTROGRAM':
        'solo_L1_stix-sci-spectrogram-87031812_20200505T235958-20200506T000054_V01_51092.fits',
    'STIX_SCI_XRAY_L0':
        'solo_L1_stix-sci-xray-l0-87031808_20200505T235958-20200506T000018_V01_50882.fits',
    'STIX_SCI_XRAY_L1':
        'solo_L1_stix-sci-xray-l1-87031809_20200505T235958-20200506T000018_V01_50883.fits',
    'STIX_SCI_XRAY_L2':
        'solo_L1_stix-sci-xray-l2-87031810_20200505T235958-20200506T000018_V01_50884.fits',
    'STIX_SCI_XRAY_L3':
        'solo_L1_stix-sci-xray-l3-87031811_20200505T235958-20200510T000014_V01_50885.fits'
}

for k, v in _TEST_DATA.items():
    p = root_dir / v
    if not p.exists():
        raise ValueError(f'Test data missing please try manually running _unzip_test_date()')
    setattr(sys.modules[__name__], k, str(p))

__all__ = [_TEST_DATA.values()]
