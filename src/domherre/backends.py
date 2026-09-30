import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

Spans = list[tuple[int, int]]

KB_BERT_MODEL = "KB/bert-base-swedish-cased-ner"
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
DEFAULT_BACKEND = "kb-bert"


class ModelNotAvailableError(RuntimeError):
    pass


class BackendNotRegisteredError(KeyError):
    pass


@runtime_checkable
class Backend(Protocol):
    def load(self) -> None: ...
    def person_spans(self, texts: list[str]) -> list[Spans]: ...


@dataclass(frozen=True)
class BackendConfig:
    model: str | None = None
    local_only: bool = False
    num_threads: int | None = None
    batch_size: int = 32
    min_score: float = 0.0
    labels: tuple[str, ...] = ("person",)
    language: str = "sv"
    options: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def opts(self) -> dict[str, Any]:
        return dict(self.options)


def _pin_threads(num_threads: int | None) -> None:
    if num_threads is None:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    try:
        import torch

        torch.set_num_threads(num_threads)
    except ImportError:
        pass


def _missing(name: str, model: str, err: Exception) -> ModelNotAvailableError:
    return ModelNotAvailableError(
        f"backend {name!r} could not load model {model!r}: {err}. "
        "Bake the model into the image, mount the cache, or set local_only=False."
    )


class KBBertBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model = config.model or KB_BERT_MODEL
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        _pin_threads(self.config.num_threads)
        if self.config.local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import pipeline

        try:
            self._pipe = pipeline(
                "token-classification",
                model=self.model,
                aggregation_strategy="simple",
            )
        except Exception as e:
            raise _missing("kb-bert", self.model, e) from e

    def person_spans(self, texts: list[str]) -> list[Spans]:
        if not texts:
            return []
        self.load()
        results = self._pipe(texts, batch_size=self.config.batch_size)
        if results and not isinstance(results[0], list):
            results = [results]
        return [
            [
                (e["start"], e["end"])
                for e in row
                if e["entity_group"] == "PER" and e["score"] >= self.config.min_score
            ]
            for row in results
        ]


class GLiNERBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.model = config.model or GLINER_MODEL
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        _pin_threads(self.config.num_threads)
        from gliner import GLiNER

        try:
            self._model = GLiNER.from_pretrained(self.model, **self.config.opts())
        except Exception as e:
            raise _missing("gliner", self.model, e) from e

    def person_spans(self, texts: list[str]) -> list[Spans]:
        if not texts:
            return []
        self.load()
        labels = list(self.config.labels)
        threshold = self.config.min_score or 0.5
        batch_size=self.config.batch_size or 32
        results = self._model.batch_predict_entities(texts, labels, threshold=threshold, batch_size=batch_size)
        return [[(e["start"], e["end"]) for e in row] for row in results]


class PresidioBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._analyzer: Any = None

    def load(self) -> None:
        if self._analyzer is not None:
            return
        injected = self.config.opts().get("analyzer")
        if injected is not None:
            self._analyzer = injected
            return

        _pin_threads(self.config.num_threads)
        from presidio_analyzer import AnalyzerEngine

        try:
            self._analyzer = AnalyzerEngine()
        except Exception as e:
            raise _missing("presidio", self.config.model or "default", e) from e

    def person_spans(self, texts: list[str]) -> list[Spans]:
        if not texts:
            return []
        self.load()
        entities = list(self.config.opts().get("entities", ("PERSON",)))
        out: list[Spans] = []
        for text in texts:
            results = self._analyzer.analyze(
                text=text, language=self.config.language, entities=entities
            )
            out.append(
                [(r.start, r.end) for r in results if r.score >= self.config.min_score]
            )
        return out


class NoopBackend:
    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig()

    def load(self) -> None:
        return

    def person_spans(self, texts: list[str]) -> list[Spans]:
        return [[] for _ in texts]


class CallableBackend:
    def __init__(self, fn: Callable[[list[str]], list[Spans]]) -> None:
        self.fn = fn

    def load(self) -> None:
        return

    def person_spans(self, texts: list[str]) -> list[Spans]:
        return self.fn(texts) if texts else []


_REGISTRY: dict[str, Callable[[BackendConfig], Backend]] = {
    "kb-bert": KBBertBackend,
    "gliner": GLiNERBackend,
    "presidio": PresidioBackend,
    "none": NoopBackend,
}

_CACHE: dict[tuple, Backend] = {}


def register_backend(name: str, factory: Callable[[BackendConfig], Backend]) -> None:
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def resolve(spec: Any, config: BackendConfig) -> Backend:
    if isinstance(spec, Backend) and not isinstance(spec, type):
        return spec
    if callable(spec) and not isinstance(spec, str):
        return CallableBackend(spec)

    if spec not in _REGISTRY:
        raise BackendNotRegisteredError(
            f"unknown backend {spec!r}; available: {available_backends()}"
        )

    key = (spec, config)
    if key not in _CACHE:
        _CACHE[key] = _REGISTRY[spec](config)
    return _CACHE[key]


def load(
    backend: Any = DEFAULT_BACKEND,
    model: str | None = None,
    local_only: bool = False,
    num_threads: int | None = None,
    **options: Any,
) -> None:
    config = BackendConfig(
        model=model,
        local_only=local_only,
        num_threads=num_threads,
        options=tuple(sorted(options.items())),
    )
    resolve(backend, config).load()


def find_person_spans(
    texts: list[str],
    backend: Any = DEFAULT_BACKEND,
    config: BackendConfig | None = None,
) -> list[list[tuple[int, int]]]:
    if not texts:
        return []
    return resolve(backend, config or BackendConfig()).person_spans(texts)


__all__ = ["load", "find_person_spans", "ModelNotAvailableError", "clear_cache"]


def clear_cache() -> None:
    _CACHE.clear()