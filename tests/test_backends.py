import polars as pl
import pytest

import domherre as kt
from domherre import backends as _backends
from domherre.backends import BackendConfig, GLiNERBackend, NoopBackend, PresidioBackend


def test_default_backend_is_kb_bert():
    assert kt.Config().backend == "kb-bert"


def test_registry_lists_builtins():
    for name in ("kb-bert", "gliner", "presidio", "none"):
        assert name in kt.available_backends()


def test_none_backend_skips_names():
    df = pl.DataFrame({"n": ["Anna Andersson 850101-0006"]})
    out = kt.redact(df, "n", kt.Config(backend="none"))
    assert out["n_redacted"][0] == "Anna Andersson [PERSONNUMMER]"


def test_unknown_backend_raises():
    with pytest.raises(kt.BackendNotRegisteredError, match="unknown backend"):
        kt.redact_text("hej", kt.Config(backend="nonesuch"))


def test_callable_backend():
    def detector(texts: list[str]) -> list[list[tuple[int, int]]]:
        return [[(0, 4)] for _ in texts]

    assert kt.redact_text("XXXX kvar", kt.Config(backend=detector)) == "[NAMN] kvar"


def test_backend_instance_passed_directly():
    class Always:
        def load(self) -> None:
            return

        def person_spans(self, texts):
            return [[(0, 3)] for _ in texts]

    assert kt.redact_text("abc def", kt.Config(backend=Always())) == "[NAMN] def"


def test_register_custom_backend():
    class Upper:
        def __init__(self, config):
            self.config = config

        def load(self) -> None:
            return

        def person_spans(self, texts):
            return [
                [(i, i + 3) for i in range(len(t)) if t[i : i + 3].isupper()][:1]
                for t in texts
            ]

    kt.register_backend("upper", Upper)
    try:
        assert "upper" in kt.available_backends()
        assert kt.redact_text("ABC def", kt.Config(backend="upper")) == "[NAMN] def"
    finally:
        _backends._REGISTRY.pop("upper", None)
        _backends.clear_cache()


def test_backend_config_carries_options():
    config = kt.Config(
        backend="gliner",
        model="x/y",
        labels=("person", "name"),
        backend_options={"map_location": "cpu"},
    )
    bc = config.backend_config()
    assert bc.model == "x/y"
    assert bc.labels == ("person", "name")
    assert bc.opts() == {"map_location": "cpu"}


def test_backend_config_is_hashable():
    assert hash(kt.Config().backend_config())


def test_backends_are_cached():
    config = BackendConfig()
    assert _backends.resolve("none", config) is _backends.resolve("none", config)


def test_different_config_gets_different_instance():
    a = _backends.resolve("none", BackendConfig(model="a"))
    b = _backends.resolve("none", BackendConfig(model="b"))
    assert a is not b


def test_noop_backend_shape():
    assert NoopBackend().person_spans(["a", "b"]) == [[], []]


def test_empty_input_short_circuits():
    assert NoopBackend().person_spans([]) == []


@pytest.mark.parametrize("cls,default", [(GLiNERBackend, kt.GLINER_MODEL)])
def test_backend_default_models(cls, default):
    assert cls(BackendConfig()).model == default


def test_gliner_import_error_is_clear():
    backend = GLiNERBackend(BackendConfig())
    with pytest.raises((ImportError, ModuleNotFoundError, kt.ModelNotAvailableError)):
        backend.load()


def test_presidio_import_error_is_clear():
    backend = PresidioBackend(BackendConfig())
    with pytest.raises((ImportError, ModuleNotFoundError, kt.ModelNotAvailableError)):
        backend.load()


def test_presidio_accepts_injected_analyzer():
    class Result:
        def __init__(self, start, end, score=0.9):
            self.start, self.end, self.score = start, end, score

    class Analyzer:
        def analyze(self, text, language, entities):
            return [Result(0, 4)]

    config = BackendConfig(options=(("analyzer", Analyzer()),))
    backend = PresidioBackend(config)
    assert backend.person_spans(["Anna kom"]) == [[(0, 4)]]


def test_min_score_filters_presidio():
    class Result:
        def __init__(self, start, end, score):
            self.start, self.end, self.score = start, end, score

    class Analyzer:
        def analyze(self, text, language, entities):
            return [Result(0, 4, 0.2), Result(5, 8, 0.95)]

    config = BackendConfig(options=(("analyzer", Analyzer()),), min_score=0.5)
    assert PresidioBackend(config).person_spans(["Anna kom"]) == [[(5, 8)]]
