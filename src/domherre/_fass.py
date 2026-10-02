import csv
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TextIO

FASS_FILENAME = "fass_lakemedelsnamn_2026-09-23.csv"

NON_NAMES = frozenset({
    "han",
    "hon",
    "hen",
    "de",
    "dem",
    "patient",
    "patienten",
})

def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def _read_names(file: TextIO) -> frozenset[str]:
    reader = csv.DictReader(file)
    if reader.fieldnames is None or "namn" not in reader.fieldnames:
        raise ValueError("FASS CSV must contain a 'namn' column")
    return frozenset(
        _normalize(row["namn"])
        for row in reader
        if row.get("namn")
    )


@lru_cache(maxsize=1)
def names() -> frozenset[str]:
    packaged = files("domherre").joinpath(FASS_FILENAME)
    if packaged.is_file():
        with packaged.open("r", encoding="utf-8-sig", newline="") as file:
            return _read_names(file)

    source = Path(__file__).resolve().parents[2] / FASS_FILENAME
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        return _read_names(file)


def exclude_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    fass_names = names()
    return [
        (start, end)
        for start, end in spans
        if _normalize(text[start:end]) not in fass_names and _normalize(text[start:end]) not in NON_NAMES
    ]