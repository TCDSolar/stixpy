from astropy.table import QTable

from stixpy.utils.table import drop_fits_checksums


def test_drop_fits_checksums_removes_known_keys():
    t = QTable({"x": [1, 2, 3]})
    t.meta["DATASUM"] = "1868371976"
    t.meta["CHECKSUM"] = "UGARV94QUGAQU93Q"
    t.meta["INSTRUME"] = "STIX"

    drop_fits_checksums(t)

    assert "DATASUM" not in t.meta
    assert "CHECKSUM" not in t.meta
    assert t.meta["INSTRUME"] == "STIX"  # unrelated keys preserved


def test_drop_fits_checksums_handles_missing_keys():
    t = QTable({"x": [1, 2]})  # no meta keys set
    drop_fits_checksums(t)  # must not raise


def test_drop_fits_checksums_accepts_multiple_tables():
    t1 = QTable({"x": [1]})
    t2 = QTable({"y": [2]})
    t1.meta["DATASUM"] = "a"
    t2.meta["CHECKSUM"] = "b"

    drop_fits_checksums(t1, t2)

    assert "DATASUM" not in t1.meta
    assert "CHECKSUM" not in t2.meta
