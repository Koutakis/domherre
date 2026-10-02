import re

import pytest

import domherre
from domherre import backends as _backends

# names the fake model "recognises" - enough to exercise span logic deterministically
KNOWN_NAMES = [
    "Abilify",
    "Engelbert Karlsson",
    "Anna Andersson",
    "Erik Bremstedt",
    "Karl Karlsson",
    "Göran",
    "Anna",
]
_NAME_RE = re.compile("|".join(sorted(KNOWN_NAMES, key=len, reverse=True)))


class FakeBackend:
    def __init__(self, config=None):
        self.config = config
        self.loaded = False

    def load(self):
        self.loaded = True

    def person_spans(self, texts):
        return [[(m.start(), m.end()) for m in _NAME_RE.finditer(t)] for t in texts]


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    _backends.clear_cache()
    monkeypatch.setitem(_backends._REGISTRY, "kb-bert", FakeBackend)
    yield
    _backends.clear_cache()


@pytest.fixture(autouse=True)
def reset_pandas_warning():
    from domherre import _frames

    _frames._PANDAS_WARNED = False
    yield


@pytest.fixture
def rows() -> list[dict]:
    return [
        {"id": 1, "note": "Engelbert Karlsson, personnummer 850101-0006, ringde."},
        {"id": 2, "note": "Ingen känslig info, ordernummer 0701234567."},
        {"id": 3, "note": "Anna Andersson mailade om 198501010006."},
        {"id": 4, "note": None},
    ]


@pytest.fixture
def kt():
    return domherre
