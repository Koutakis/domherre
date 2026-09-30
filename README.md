# domherre

[![PyPI](https://img.shields.io/pypi/v/domherre)](https://pypi.org/project/domherre/)
[![Python](https://img.shields.io/pypi/pyversions/domherre)](https://pypi.org/project/domherre/)

Redacts Swedish personal names, personnummer and e-mail addresses from string columns
in a polars DataFrame.

- **Personnummer** — regex (date-shaped) + Luhn checksum.
- **Names** — `KB/bert-base-swedish-cased-ner`, `PRS` entities only.
- **E-mail** — regex, shape-based.

## Install

```bash
pip install domherre
```

## What you get

```python
import polars as pl
import domherre as dh

df = pl.DataFrame({
    "id": [1, 2],
    "note": [
        "Engelbert Karlsson, personnummer 850101-0006, ringde idag.",
        "Ingen känslig info här.",
    ],
})

dh.redact(df, "note")
# note -> note_redacted
# "[NAMN], personnummer [PERSONNUMMER], ringde idag."
```

Redacted columns are **renamed** with a suffix (default `_redacted`) so a redacted
column can never be mistaken for the source. The original column is replaced, not
kept. Unlisted columns are untouched. Raises if a target name already exists.

Several columns at once:

```python
dh.redact(df, ["note", "kommentar"])
```

Single string:

```python
dh.redact_text("Anna Andersson, 850101-0006")
```

pandas input works and comes back as pandas. A one-time `PandasPerformanceWarning`
fires per process; silence it with `DOMHERRE_QUIET=1` if you have already made your
peace with pandas.

## Read before you trust us

The default model is trained on SUC 3.0 — formal written Swedish. On informal text
(chat logs, free-text notes, heavy abbreviation) it **will miss names, silently and
without error**. Personnummer and e-mail are regex and deterministic; names are not.

Sample the output before calling this compliance-grade, and use `scan` to assert on
what you are shipping rather than assuming.

## Counting what was found

`redact` returns just the frame. `redact_with_report` also returns counts, and `scan`
counts without modifying anything:

```python
out, report = dh.redact_with_report(df, "note")
report.rows, report.rows_with_pii
report.names, report.personnummer, report.email
report.extra            # {"[TELEFON]": 3} for extra_patterns

# guard a pipeline that must never carry PII
if dh.scan(df, ["note", "kommentar"]).rows_with_pii:
    raise ValueError("PII reached a clean layer")
```

Ship these counts to your metrics. Without them you have no way to notice the model
quietly stopped matching anything after a version bump.

## Config

```python
config = dh.Config(
    name_tag="[NAMN]",
    personnummer_tag="[PERSONNUMMER]",
    email_tag="[EMAIL]",
    suffix="_redacted",

    backend="kb-bert",       # or "gliner", "presidio", "none", or your own
    model=None,              # backend default, or a local path: "/models/ner"
    local_only=False,        # True = never hit the network, fail loudly if not cached
    num_threads=2,           # pin to your k8s cpu limit
    batch_size=32,
    min_score=0.0,           # raise to cut false-positive names, at the cost of misses
    labels=("person",),      # gliner only
    language="sv",
    backend_options={},      # passed through to the backend

    redact_names=True,
    redact_personnummer=True,
    redact_email=True,
    validate_personnummer=True,   # False = redact on shape alone, catches typo'd numbers
    enumerate_names=False,        # [NAMN_1], [NAMN_2] instead of flat [NAMN]
    extra_patterns={},            # {"[TELEFON]": r"\b0\d{1,3}[- ]?\d{5,8}\b"}
)

dh.redact(df, "note", config)
```

`extra_patterns` is applied alongside the built-ins, and wins on an identical span —
so `{"[EPOST]": r"\S+@\S+\.\w+"}` replaces the built-in `[EMAIL]` tag rather than
fighting it. On partial overlap the longest span wins.

### enumerate_names

Distinguishes people within a single string, so downstream analysis can still tell
that two mentions refer to the same person:

```
"[NAMN_1] mailade [NAMN_2] om [NAMN_1]s ärende"
```

Numbering is per string, not per DataFrame. `[NAMN_1]` in row 1 and row 2 are not
necessarily the same person — that would need identity resolution, which this does not do.

## Backend models

Name detection is pluggable. Personnummer and e-mail detection are always regex and
are not affected by the backend choice. `kb-bert` is the default; it is the strongest
Swedish-specific name model we have measured.

| Backend | Install | Notes |
|---|---|---|
| `kb-bert` (default) | included | `KB/bert-base-swedish-cased-ner`. Swedish-only, trained on formal text. |
| `gliner` | `domherre[gliner]` | Zero-shot, multilingual, CPU-optimised. Change `labels` to detect anything. |
| `presidio` | `domherre[presidio]` | Full PII framework. Bring your own configured `AnalyzerEngine`. |
| `none` | included | Skips name detection. Regex only. |

```python
dh.redact(df, "note", dh.Config(backend="gliner"))
```

`kb-bert` needs the `transformers` extra:

```bash
pip install domherre[transformers]
```

### gliner

```python
config = dh.Config(
    backend="gliner",
    model="urchade/gliner_multi_pii-v1",   # default
    labels=("person", "full name"),
    min_score=0.5,
    backend_options={"map_location": "cpu", "quantize": True},
)
```

### Custom backends

Anything with `load()` and `person_spans(texts) -> list[list[tuple[int, int]]]` works.
A plain callable works too:

```python
def detector(texts: list[str]) -> list[list[tuple[int, int]]]:
    return [[(m.start(), m.end()) for m in pattern.finditer(t)] for t in texts]

dh.redact(df, "note", dh.Config(backend=detector))
```

Register by name to make it selectable like a builtin:

```python
dh.register_backend("mine", MyBackend)
dh.available_backends()   # ['gliner', 'kb-bert', 'mine', 'none', 'presidio']
```

Backends are cached per `(name, config)`, so a Deployment loads each model once.

## Service deployment

Load the model at startup so the first request doesn't pay for it:

```python
dh.load(backend="kb-bert", num_threads=2, local_only=True)
```

Bake the weights into the image and set `local_only=True`. Otherwise every cold pod
pulls ~440MB from HuggingFace, and a network blip becomes a runtime failure in the
middle of a batch instead of a clear error at boot.

```dockerfile
RUN python -c "from transformers import pipeline; \
    pipeline('token-classification', model='KB/bert-base-swedish-cased-ner')"
```

```yaml
resources:
  requests: { cpu: "1", memory: "2Gi" }
  limits:   { cpu: "2", memory: "3Gi" }
```

Set `num_threads` to match the CPU limit. Torch otherwise reads the node's core count,
not the cgroup limit, and thrashes threads on a large node.

## Personnummer formats detected

`YYMMDD-XXXX`, `YYMMDD+XXXX`, `YYMMDDXXXX`, `YYYYMMDD-XXXX`, `YYYYMMDDXXXX`,
plus samordningsnummer (day + 60). All must pass the Luhn check unless
`validate_personnummer=False`.

Not detected: reservnummer/interimsnummer (region-specific, no fixed national format),
and personnummer with a typo'd control digit — set `validate_personnummer=False`
to catch those, at the cost of redacting unrelated 10 and 12-digit numbers.

## License

MIT
