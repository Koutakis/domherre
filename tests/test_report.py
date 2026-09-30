import polars as pl

import domherre as kt
from domherre import Report


def test_report_counts(rows):
    _, report = kt.redact_with_report(pl.DataFrame(rows), "note")
    assert report.rows == 4
    assert report.rows_with_pii == 2
    assert report.names == 2
    assert report.personnummer == 2


def test_report_extra_counts():
    df = pl.DataFrame({"n": ["a@b.se och c@d.se"]})
    config = kt.Config(extra_patterns={"[EPOST]": r"\S+@\S+\.\w+"})
    _, report = kt.redact_with_report(df, "n", config)
    assert report.extra == {"[EPOST]": 2}


def test_report_addition():
    total = Report(rows=1, names=2, extra={"[X]": 1}) + Report(rows=3, names=1, extra={"[X]": 2})
    assert total.rows == 4
    assert total.names == 3
    assert total.extra == {"[X]": 3}


def test_scan_does_not_modify(rows):
    df = pl.DataFrame(rows)
    report = kt.scan(df, "note")
    assert df.columns == ["id", "note"]
    assert df["note"][0].startswith("Engelbert")
    assert report.rows_with_pii == 2


def test_scan_clean_frame():
    report = kt.scan(pl.DataFrame({"n": ["inget här", "heller inte"]}), "n")
    assert report.rows_with_pii == 0


def test_scan_multiple_columns(rows):
    df = pl.DataFrame(rows).with_columns(pl.col("note").alias("note2"))
    report = kt.scan(df, ["note", "note2"])
    assert report.rows == 8
    assert report.rows_with_pii == 4
