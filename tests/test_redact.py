import polars as pl
import pytest

import domherre as kt
from domherre.core import _apply_spans


@pytest.fixture
def df(rows):
    return pl.DataFrame(rows)


def test_default_tags(df):
    out = kt.redact(df, "note")
    assert out["note_redacted"][0] == "[NAMN], personnummer [PERSONNUMMER], ringde."


def test_column_is_renamed(df):
    out = kt.redact(df, "note")
    assert out.columns == ["id", "note_redacted"]
    assert "note" not in out.columns


def test_source_frame_not_mutated(df):
    before = df["note"][0]
    kt.redact(df, "note")
    assert df.columns == ["id", "note"]
    assert df["note"][0] == before


def test_nulls_preserved(df):
    assert kt.redact(df, "note")["note_redacted"][3] is None


def test_non_pii_untouched(df):
    assert "0701234567" in kt.redact(df, "note")["note_redacted"][1]


def test_custom_tags_and_suffix(df):
    config = kt.Config(name_tag="<N>", personnummer_tag="<P>", suffix="_anon")
    out = kt.redact(df, "note", config)
    assert out.columns == ["id", "note_anon"]
    assert out["note_anon"][0] == "<N>, personnummer <P>, ringde."


def test_multiple_columns():
    df = pl.DataFrame({"a": ["Anna Andersson"], "b": ["Anna Andersson"], "c": ["Anna Andersson"]})
    out = kt.redact(df, ["a", "b"])
    assert out.columns == ["a_redacted", "b_redacted", "c"]
    assert out["c"][0] == "Anna Andersson"


def test_toggles(df):
    names_only = kt.redact(df, "note", kt.Config(redact_personnummer=False))
    assert "850101-0006" in names_only["note_redacted"][0]
    assert "[NAMN]" in names_only["note_redacted"][0]

    pnr_only = kt.redact(df, "note", kt.Config(redact_names=False))
    assert "Engelbert Karlsson" in pnr_only["note_redacted"][0]
    assert "[PERSONNUMMER]" in pnr_only["note_redacted"][0]


def test_enumerate_names_reuses_number():
    out = kt.redact_text("Anna Andersson mailade Anna Andersson", kt.Config(enumerate_names=True))
    assert out.count("[NAMN_1]") == 2
    assert "[NAMN_2]" not in out


def test_enumerate_names_distinguishes_people():
    out = kt.redact_text("Anna Andersson och Erik Bremstedt", kt.Config(enumerate_names=True))
    assert "[NAMN_1]" in out and "[NAMN_2]" in out


def test_extra_patterns():
    config = kt.Config(extra_patterns={"[EPOST]": r"\S+@\S+\.\w+"})
    assert kt.redact_text("maila kalle@example.se", config) == "maila [EPOST]"


def test_email_redacted():
    assert kt.redact_text("maila kalle@example.se") == "maila [EMAIL]"


def test_email_toggle_off():
    out = kt.redact_text("maila kalle@example.se", kt.Config(redact_email=False))
    assert "kalle@example.se" in out


def test_name_and_email_redacted():
    out = kt.redact_text("Görans mailadress är göran@görancosmetics.se")
    assert out == "[NAMN]s mailadress är [EMAIL]"


def test_longest_span_wins_on_overlap():
    # "Anna Andersson" and "Anna" both match; no truncated garbage left behind
    assert kt.redact_text("Anna Andersson kom") == "[NAMN] kom"


def test_fass_medicine_is_not_redacted_as_name():
    out = kt.redact_text("Anna Andersson tar Abilify")
    assert out == "[NAMN] tar Abilify"


def test_apply_spans_offsets():
    assert _apply_spans("abcdefghij", [(0, 5, "[A]"), (2, 7, "[B]"), (7, 10, "[C]")]) == "[A]fg[C]"


def test_redact_text_empty():
    assert kt.redact_text("") == ""


def test_empty_dataframe():
    df = pl.DataFrame({"note": []}, schema={"note": pl.Utf8})
    out = kt.redact(df, "note")
    assert out.height == 0
    assert out.columns == ["note_redacted"]


def test_single_column_as_string_or_list(df):
    assert kt.redact(df, "note").columns == kt.redact(df, ["note"]).columns


def test_missing_column_raises(df):
    with pytest.raises(ValueError, match="not in dataframe"):
        kt.redact(df, "nope")


def test_non_string_column_raises(df):
    with pytest.raises(TypeError, match="string typed"):
        kt.redact(df, "id")


def test_name_clash_raises():
    df = pl.DataFrame({"a": ["x"], "a_redacted": ["y"]})
    with pytest.raises(ValueError, match="already exist"):
        kt.redact(df, "a")


def test_non_frame_raises():
    with pytest.raises(TypeError, match="polars or pandas"):
        kt.redact({"note": ["hej"]}, "note")

def test_non_names_are_not_redacted():
    detector = lambda texts: [[(0, len(text))] for text in texts]

    for text in ("Han", "Hon", "Hen", "De", "Dem", "Patient", "Patienten"):
        config = kt.Config(backend=detector)
        assert kt.redact_text(text, config) == text