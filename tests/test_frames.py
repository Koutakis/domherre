import polars as pl
import pytest

import domherre as kt

pd = pytest.importorskip("pandas")


@pytest.fixture
def pdf(rows):
    return pd.DataFrame(rows)


def test_returns_same_frame_type(pdf, rows):
    assert isinstance(kt.redact(pdf, "note"), pd.DataFrame)
    assert isinstance(kt.redact(pl.DataFrame(rows), "note"), pl.DataFrame)


def _norm(values: list) -> list:
    return [None if v is None or pd.isna(v) else v for v in values]


def test_pandas_polars_parity(pdf, rows):
    from_pandas = kt.redact(pdf, "note")["note_redacted"].tolist()
    from_polars = kt.redact(pl.DataFrame(rows), "note")["note_redacted"].to_list()
    assert _norm(from_pandas) == _norm(from_polars)


def test_pandas_report_parity(pdf, rows):
    _, a = kt.redact_with_report(pdf, "note")
    _, b = kt.redact_with_report(pl.DataFrame(rows), "note")
    assert a == b


def test_pandas_column_renamed(pdf):
    out = kt.redact(pdf, "note")
    assert list(out.columns) == ["id", "note_redacted"]


def test_pandas_source_not_mutated(pdf):
    kt.redact(pdf, "note")
    assert list(pdf.columns) == ["id", "note"]
    assert pdf["note"][0].startswith("Engelbert")


def test_pandas_index_preserved(rows):
    df = pd.DataFrame(rows, index=[10, 20, 30, 40])
    assert list(kt.redact(df, "note").index) == [10, 20, 30, 40]


def test_pandas_nulls(pdf):
    # pandas keeps its own null convention (NaN/NA), domherre does not force None
    assert pd.isna(kt.redact(pdf, "note")["note_redacted"][3])


def test_pandas_string_dtype(rows):
    df = pd.DataFrame(rows).astype({"note": "string"})
    out = kt.redact(df, "note")
    assert out["note_redacted"].dtype == df["note"].dtype


def test_pandas_non_string_column_raises(pdf):
    with pytest.raises(TypeError, match="string typed"):
        kt.redact(pdf, "id")


def test_pandas_emits_perf_warning(pdf):
    with pytest.warns(kt.PandasPerformanceWarning, match="polars-first"):
        kt.redact(pdf, "note")


def test_warning_only_fires_once(pdf):
    with pytest.warns(kt.PandasPerformanceWarning):
        kt.redact(pdf, "note")
    with warnings_as_errors():
        kt.redact(pdf, "note")


def test_polars_emits_no_warning(rows):
    with warnings_as_errors():
        kt.redact(pl.DataFrame(rows), "note")


def test_quiet_env_suppresses(pdf, monkeypatch):
    monkeypatch.setenv("DOMHERRE_QUIET", "1")
    with warnings_as_errors():
        kt.redact(pdf, "note")


def warnings_as_errors():
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error", kt.PandasPerformanceWarning)
            yield

    return ctx()
