# domherre

Redacts Swedish personal names, personnummer and e-mail adresses from string columns in a polars DataFrame.

- **Personnummer** — regex (date-shaped) + Luhn checksum.
- **Names** — `KB/bert-base-swedish-cased-ner`, `PRS` entities only. 

## Install

```bash
pip install domherre
```

## What you get

```python
import polars as pl
import domherre as kt

df = pl.DataFrame({
    "id": [1, 2],
    "note": [
        "Engelbert Karlsson, personnummer 850101-0006, ringde idag.",
        "Ingen känslig info här.",
    ],
})

kt.redact(df, "note")
# note -> note_redacted
# "[NAMN], personnummer [PERSONNUMMER], ringde idag."
```

- Redacted columns are **renamed** with a suffix (default `_redacted`) so a redacted column
- can never be mistaken for the source. The original column is replaced, not kept.
- Unlisted columns are untouched. Raises if a target name already exists.

Several columns at once:

```python
kt.redact(df, ["note", "kommentar"])
```

Single string:

```python
kt.redact_text("Anna Andersson, 850101-0006")
```


A one-time `PandasPerformanceWarning` fires per process. Silence it with
`DOMHERRE_QUIET=1` if you have already made your peace with pandas.

## Backend models

Name detection is pluggable. Personnummer detection is always regex + Luhn and is
not affected by the backend choice. This project defaults to kb-bert model as it has shown to be the formidable model for swedish name detection.

| Backend | Install | Notes |
|---|---|---|
| `kb-bert` (default) | included | `KB/bert-base-swedish-cased-ner`. Swedish-only, trained on formal text. |
| `gliner` | `domherre[gliner]` | Zero-shot, multilingual, CPU-optimised. Change `labels` to detect anything. |
| `presidio` | `domherre[presidio]` | Full PII framework. Bring your own configured `AnalyzerEngine`. |
| `none` | included | Skips name detection. Regex only. |

```python
kt.redact(df, "note", kt.Config(backend="gliner"))
```

### gliner

```python
config = kt.Config(
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

kt.redact(df, "note", kt.Config(backend=detector))
```

Register by name to make it selectable like a builtin:

```python
kt.register_backend("mine", MyBackend)
kt.available_backends()   # ['gliner', 'kb-bert', 'mine', 'none', 'presidio']
```

Backends are cached per (name, config), so a Deployment loads each model once.

## Config

```python
config = kt.Config(
    name_tag="[NAMN]",
    personnummer_tag="[PERSONNUMMER]",
    suffix="_redacted",
    backend="kb-bert",       # or "gliner", "presidio", "none", or your own
    model=None,              # backend default, or a local path: "/models/ner"
    local_only=False,        # True = never hit the network, fail loudly if not cached
    num_threads=2,           # pin to your k8s cpu limit
    batch_size=32,
    min_score=0.0,           # raise to cut false-positive names, at the cost of misses
    redact_names=True,
    redact_personnummer=True,
    validate_personnummer=True,   # False = redact on shape alone, catches typo'd numbers
    enumerate_names=False,        # [NAMN_1], [NAMN_2] instead of flat [NAMN]
    extra_patterns={},            # {"[EPOST]": r"\S+@\S+\.\w+"}
)

kt.redact(df, "note", config)
```

### enumerate_names

Distinguishes people within a single string, so downstream analysis can still tell
that two mentions refer to the same person:

```
"[NAMN_1] mailade [NAMN_2] om [NAMN_1]s ärende"
```

Numbering is per string, not per DataFrame. `[NAMN_1]` in row 1 and row 2 are not
necessarily the same person — that would need identity resolution, which this does not do.

## Service deployment

Domherre is suited as deployemnts that can be run as services, we thus support models pre-baked into images.
Load the model at startup so the first request doesn't pay for it:

```python
kt.load(backend="kb-bert", num_threads=2, local_only=True)
```

Bake the weights into the image and set `local_only=True`. Otherwise every cold pod
pulls ~440MB from HuggingFace, and a network blip becomes a runtime failure
in the middle of a batch instead of a clear error at boot.

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

