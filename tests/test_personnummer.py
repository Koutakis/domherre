import pytest

from domherre import find_personnummer, is_valid_personnummer

VALID = "8501010006"


@pytest.mark.parametrize(
    "raw",
    [
        "850101-0006",
        "850101+0006",
        "8501010006",
        "19850101-0006",
        "198501010006",
    ],
)
def test_valid_formats(raw):
    assert is_valid_personnummer(raw)
    assert find_personnummer(f"text {raw} text")


@pytest.mark.parametrize(
    "raw",
    [
        "850101-0000",  # bad checksum
        "0701234567",  # phone shaped
        "850101-000",  # too short
        "",
    ],
)
def test_invalid(raw):
    assert not is_valid_personnummer(raw)


def test_samordningsnummer_shape_matches():
    # day + 60, shape is accepted; checksum still decides
    hits = find_personnummer("850161-1234", validate=False)
    assert hits == [(0, 11)]


def test_validate_false_keeps_bad_checksum():
    assert find_personnummer("850101-0000", validate=False)
    assert not find_personnummer("850101-0000", validate=True)


def test_impossible_dates_rejected():
    assert not find_personnummer("851301-0006", validate=False)  # month 13
    assert not find_personnummer("850132-0006", validate=False)  # day 32


def test_multiple_in_one_string():
    text = f"{VALID} och 19{VALID} slut"
    assert len(find_personnummer(text)) == 2


def test_offsets_are_usable():
    text = "pnr 850101-0006 ok"
    (start, end), = find_personnummer(text)
    assert text[start:end] == "850101-0006"
