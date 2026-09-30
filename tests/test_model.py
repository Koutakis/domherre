import os

import polars as pl
import pytest

import domherre as kt

pytestmark = pytest.mark.skipif(
    not os.environ.get("DOMHERRE_INTEGRATION"),
    reason="set DOMHERRE_INTEGRATION=1 to run against the real model",
)


@pytest.fixture(autouse=True)
def real_model(monkeypatch):
    monkeypatch.undo()  # drop the stub from conftest
    yield


def test_model_loads():
    kt.load()


def test_detects_obvious_name():
    out = kt.redact_text("Anna Andersson ringde igår.")
    assert "[NAMN]" in out
    assert "Andersson" not in out


def test_leaves_organisations_alone():
    out = kt.redact_text("Region Stockholm fattade beslutet.")
    assert "[NAMN]" not in out


def test_leaves_locations_alone():
    out = kt.redact_text("Mötet hölls i Göteborg.")
    assert "[NAMN]" not in out


def test_name_and_personnummer_together():
    df = pl.DataFrame({"note": ["Erik Bremstedt, 850101-0006, kom in."]})
    out, report = kt.redact_with_report(df, "note")
    assert "[NAMN]" in out["note_redacted"][0]
    assert "[PERSONNUMMER]" in out["note_redacted"][0]
    assert report.names >= 1 and report.personnummer == 1


def test_local_only_missing_model_raises():
    with pytest.raises(kt.ModelNotAvailableError):
        kt.load(model="KB/does-not-exist-anywhere", local_only=True)
