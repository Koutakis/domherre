import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

from . import _frames
from .backends import DEFAULT_BACKEND, KB_BERT_MODEL, BackendConfig
from .backends import find_person_spans
from ._personnummer import find_personnummer
from ._email import find_email

MODEL_NAME = KB_BERT_MODEL

Frame = TypeVar("Frame")

NAME_TAG = "[NAMN]"
PERSONNUMMER_TAG = "[PERSONNUMMER]"
EMAIL_TAG = "[EMAIL]"


@dataclass(frozen=True)
class Config:
    name_tag: str = NAME_TAG
    personnummer_tag: str = PERSONNUMMER_TAG
    email_tag: str = EMAIL_TAG
    suffix: str = "_redacted"
    backend: Any = DEFAULT_BACKEND
    model: str | None = None
    local_only: bool = False
    num_threads: int | None = None
    batch_size: int = 32
    min_score: float = 0.0
    labels: tuple[str, ...] = ("person",)
    language: str = "sv"
    backend_options: dict[str, Any] = field(default_factory=dict)
    redact_names: bool = True
    redact_personnummer: bool = True
    redact_email: bool = True
    validate_personnummer: bool = True
    enumerate_names: bool = False
    extra_patterns: dict[str, str] = field(default_factory=dict)

    def backend_config(self) -> BackendConfig:
        return BackendConfig(
            model=self.model,
            local_only=self.local_only,
            num_threads=self.num_threads,
            batch_size=self.batch_size,
            min_score=self.min_score,
            labels=tuple(self.labels),
            language=self.language,
            options=tuple(sorted(self.backend_options.items())),
        )


@dataclass
class Report:
    rows: int = 0
    rows_with_pii: int = 0
    names: int = 0
    personnummer: int = 0
    email: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def __add__(self, other: "Report") -> "Report":
        merged = dict(self.extra)
        for k, v in other.extra.items():
            merged[k] = merged.get(k, 0) + v
        return Report(
            rows=self.rows + other.rows,
            rows_with_pii=self.rows_with_pii + other.rows_with_pii,
            names=self.names + other.names,
            personnummer=self.personnummer + other.personnummer,
            extra=merged,
        )


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    if not spans:
        return text

    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))

    kept: list[tuple[int, int, str]] = []
    for span in spans:
        if kept and span[0] < kept[-1][1]:
            continue  # overlapping, longest-at-this-start already won
        kept.append(span)

    for start, end, tag in reversed(kept):  # right-to-left keeps offsets valid
        text = text[:start] + tag + text[end:]
    return text


def _numbered(spans: list[tuple[int, int, str]], text: str, tag: str) -> list[tuple[int, int, str]]:
    seen: dict[str, int] = {}
    out: list[tuple[int, int, str]] = []
    for start, end, t in spans:
        if t != tag:
            out.append((start, end, t))
            continue
        key = text[start:end].casefold()
        if key not in seen:
            seen[key] = len(seen) + 1
        out.append((start, end, f"{tag[:-1]}_{seen[key]}]"))
    return out


def _spans_for(text: str, names: list[tuple[int, int]], config: Config) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    if config.redact_personnummer:
        spans += [
            (s, e, config.personnummer_tag)
            for s, e in find_personnummer(text, validate=config.validate_personnummer)
        ]

    # Nike added
    if config.redact_email:
        spans += [
            (start, end, config.email_tag)
            for start, end in find_email(text)
        ]
    # end

    for tag, pattern in config.extra_patterns.items():
        spans += [(m.start(), m.end(), tag) for m in re.finditer(pattern, text)]

    spans += [(s, e, config.name_tag) for s, e in names]

    if config.enumerate_names:
        spans = _numbered(spans, text, config.name_tag)

    return spans


def _redact_texts(texts: list[str], config: Config) -> tuple[list[str], Report]:
    report = Report(rows=len(texts))
    filled: list[int] = [i for i, t in enumerate(texts) if t]
    if not filled:
        return list(texts), report

    subset: list[str] = [texts[i] for i in filled]

    if config.redact_names:
        name_spans = find_person_spans(
            subset,
            backend=config.backend,
            config=config.backend_config(),
        )
    else:
        name_spans = [[] for _ in subset]

    out = list(texts)
    for i, text, names in zip(filled, subset, name_spans):
        spans = _spans_for(text, names, config)
        if not spans:
            continue

        report.rows_with_pii += 1
        for _, _, tag in spans:
            if tag.startswith(config.name_tag[:-1]):
                report.names += 1
            elif tag == config.personnummer_tag:
                report.personnummer += 1
            else:
                report.extra[tag] = report.extra.get(tag, 0) + 1

        out[i] = _apply_spans(text, spans)

    return out, report


def redact_text(text: str, config: Config | None = None) -> str:
    if not text:
        return text
    out, _ = _redact_texts([text], config or Config())
    return out[0]


def _resolve(df: Any, columns: str | list[str]) -> list[str]:
    _frames.validate(df)
    cols = [columns] if isinstance(columns, str) else list(columns)

    present = _frames.columns(df)
    missing = [c for c in cols if c not in present]
    if missing:
        raise ValueError(f"columns not in dataframe: {missing}")

    wrong_type = _frames.check_string_columns(df, cols)
    if wrong_type:
        raise TypeError(f"columns must be string typed, got: {wrong_type}")

    return cols


def redact(
    df: Frame,
    columns: str | list[str],
    config: Config | None = None,
) -> Frame:
    out, _ = redact_with_report(df, columns, config)
    return out


def redact_with_report(
    df: Frame,
    columns: str | list[str],
    config: Config | None = None,
) -> tuple[Frame, Report]:
    cols = _resolve(df, columns)
    config = config or Config()

    present = _frames.columns(df)
    renames = {c: f"{c}{config.suffix}" for c in cols}
    clashes = [n for n in renames.values() if n in present]
    if clashes:
        raise ValueError(f"target column names already exist: {clashes}")

    out = df
    total = Report()
    for col in cols:
        redacted, report = _redact_texts(_frames.to_list(out, col), config)
        out = _frames.set_column(out, col, redacted)
        total = total + report

    return _frames.rename(out, renames), total


def scan(
    df: Frame,
    columns: str | list[str],
    config: Config | None = None,
) -> Report:
    cols = _resolve(df, columns)
    config = config or Config()

    total = Report()
    for col in cols:
        _, report = _redact_texts(_frames.to_list(df, col), config)
        total = total + report
    return total
