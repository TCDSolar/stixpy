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
        def is_within_directory(directory, target):

            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)

            prefix = os.path.commonprefix([abs_directory, abs_target])

            return prefix == abs_directory

        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")

            tar.extractall(path, members, numeric_owner=numeric_owner)

        safe_extract(tar, root_dir.absolute())


_unzip_test_data()

_TEST_DATA = {
    'STIX_HK_MAXI_TIMESERIES': 'solo_L1_stix-hk-maxi_20200501_V01.fits',
    'STIX_QL_BACKGROUND_TIMESERIES': 'solo_L1_stix-ql-background_20200506_V01.fits',
    'STIX_QL_CALIBRATION': 'solo_L1_stix-cal-energy_20200506_V01.fits',
    'STIX_QL_FLAREFLAG':  'solo_L1_stix-ql-ql-tmstatusflarelist_20200506_V01.fits',
    'STIX_QL_LIGHTCURVE_TIMESERIES': 'solo_L1_stix-ql-lightcurve_20200506_V01.fits',
    'STIX_QL_SPECTRA': 'solo_L1_stix-ql-spectra_20200506_V01.fits',
    'STIX_QL_VARIANCE_TIMESERIES': 'solo_L1_stix-ql-variance_20200506_V01.fits',
    'STIX_SCI_XRAY_SPEC':
        'solo_L1_stix-sci-xray-spec_20200505T235959-20200506T000019_V01_0087031812-50886.fits',
    'STIX_SCI_XRAY_RPD':
        'solo_L1_stix-sci-xray-rpd_20200505T235959-20200506T000019_V01_0087031808-50882.fits',
    'STIX_SCI_XRAY_CPD':
        'solo_L1_stix-sci-xray-cpd_20200505T235959-20200506T000019_V01_0087031809-50883.fits',
    'STIX_SCI_XRAY_SCPD':
        'solo_L1_stix-sci-xray-scpd_20200505T235959-20200506T000019_V01_0087031810-50884.fits',
    # 'STIX_SCI_XRAY_VIS':
    #     'solo_L1_stix-sci-xray-vis-87031811_20200505T235958-20200510T000014_V01_50885.fits'
}

__doc__ = ''
for k, v in _TEST_DATA.items():
    p = root_dir / v
    if not p.exists():
        raise ValueError(f'Test data {k}, {v} missing please try manually running _unzip_test_date()')
    setattr(sys.modules[__name__], k, str(p))
    __doc__ = __doc__ + f'   - ``{k}``\n'

__all__ = [*_TEST_DATA.keys()]
