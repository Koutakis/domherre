# domherre

Redacts Swedish personal names and personnummer from string columns in a polars DataFrame.

- **Personnummer** — regex (date-shaped) + Luhn checksum. No false positives on random digit strings.
- **Names** — `KB/bert-base-swedish-cased-ner`, `PRS` entities only. Organisations, locations, events and times are left alone.

## Install

```bash
pip install domherre
```

## Usage

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

Redacted columns are **renamed** with a suffix (default `_redacted`) so a redacted column
can never be mistaken for the source. The original column is replaced, not kept —
keeping it would leave the PII sitting in the frame.

Same shape, same column order. Input is not mutated. Nulls stay null.
Unlisted columns are untouched. Raises if a target name already exists.

Several columns at once:

```python
kt.redact(df, ["note", "kommentar"])
```

Single string:

```python
kt.redact_text("Anna Andersson, 850101-0006")
```

## pandas

pandas frames work and return pandas frames, but polars is the supported path.
pandas is not a dependency — it is only touched if you pass a pandas frame in.

```python
kt.redact(pandas_df, "note")   # -> pandas.DataFrame
```

A one-time `PandasPerformanceWarning` fires per process. Silence it with
`DOMHERRE_QUIET=1` if you have already made your peace with pandas.

Null convention follows the input frame — polars gives back `None`, pandas gives back
`NaN`/`NA`. Use `pd.isna()` rather than `is None` when checking pandas output.

Arrow-backed columns (`string[pyarrow]`) avoid a conversion step and get a shorter
warning. If you are staying on pandas, at least use those.

## Backends

Name detection is pluggable. Personnummer detection is always regex + Luhn and is
not affected by the backend choice.

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

`labels` are free text — GLiNER is zero-shot, so `("person", "patient name", "doctor")`
works without retraining. `backend_options` is passed straight to `GLiNER.from_pretrained`.

Worth trying if `kb-bert` misses names in informal text. It is a newer model, multilingual,
and the PII variants are trained for exactly this job.

### presidio

Presidio is a whole PII framework, not a model — it wraps recognizers and does its own
anonymization. domherre uses only its analyzer, so you are using a slice of it. If you
want Presidio's operators, allow-lists and decision tracing, use Presidio directly
instead of through this.

Default construction gives you an English `AnalyzerEngine`. For Swedish you must build
and inject your own:

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

nlp = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "sv", "model_name": "sv_core_news_lg"}],
}).create_engine()

analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["sv"])

config = kt.Config(
    backend="presidio",
    language="sv",
    backend_options={"analyzer": analyzer, "entities": ("PERSON",)},
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

### extra_patterns

Regex patterns applied alongside the built-ins. Key is the replacement tag:

```python
kt.Config(extra_patterns={
    "[EPOST]": r"\S+@\S+\.\w+",
    "[TELEFON]": r"\b0\d{1,3}[- ]?\d{5,8}\b",
})
```

## Reporting

`redact` returns just the frame. When you need counts for audit or monitoring:

```python
df_out, report = kt.redact_with_report(df, "note")
report  # Report(rows=1000, rows_with_pii=143, names=201, personnummer=97, extra={})
```

Detect without modifying — useful for checking whether a dataset needs redaction
at all, or for alerting when PII shows up somewhere it shouldn't:

```python
report = kt.scan(df, ["note", "kommentar"])
if report.rows_with_pii:
    raise ValueError(f"unexpected PII: {report}")
```

## Service deployment

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

Sizing for ~100k strings/day:

```yaml
resources:
  requests: { cpu: "1", memory: "2Gi" }
  limits:   { cpu: "2", memory: "3Gi" }
```

Set `num_threads` to match the CPU limit. Torch otherwise reads the node's core count,
not the cgroup limit, and thrashes threads on a large node.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite stubs the NER pipeline, so it runs offline in CI with no model download.
Integration tests that hit the real model are skipped unless you ask for them:

```bash
DOMHERRE_INTEGRATION=1 pytest tests/test_model.py
```

Run those at least once against your deployment image — the stub proves the span
and frame logic is correct, not that the model still recognises Swedish names.

## Personnummer formats detected

`YYMMDD-XXXX`, `YYMMDD+XXXX`, `YYMMDDXXXX`, `YYYYMMDD-XXXX`, `YYYYMMDDXXXX`,
plus samordningsnummer (day + 60). All must pass the Luhn check unless
`validate_personnummer=False`.

Not detected: reservnummer/interimsnummer (region-specific, no fixed national format),
and personnummer with a typo'd control digit — set `validate_personnummer=False`
to catch those, at the cost of redacting unrelated 10 and 12-digit numbers.

## Caveats

The default `kb-bert` model is trained on SUC 3.0 — formal written Swedish. On informal text
(chat logs, free-text notes, heavy abbreviation) it will miss names, silently and
without error. Try the `gliner` backend if that is your data. Either way, sample the
output before treating this as compliance-grade redaction.

## Comparing models

The manual comparison script evaluates KB-BERT and the configured GLiNER models
against the annotated test data in `compareModels/`.

Install the project with the GLiNER dependencies:

```bash
python -m pip install -e ".[gliner]"
```

If GLiNER reports that protobuf is missing:

```bash
python -m pip install protobuf
```

Run the comparison from the repository root:

```bash
python compareModels/compare_models.py
```

The script:

Loads each model before starting its timer.
Processes the same test strings with every model - test_srtings.txt.
Writes redacted text to compareModels/results_<model>.txt.
Compares predictions with test_strings1.jsonl.
Writes precision, recall, F1, error examples, and timing to
stats_file.txt.
Models and settings are configured in the MODELS dictionary in
compare_models.py. Use the same min_score when comparing GLiNER
models. The configured batch_size controls how many strings are processed
together and affects both speed and memory consumption.

Model weights are downloaded from Hugging Face on the first run and reused from
the local cache afterward. An internet connection is therefore normally required
only for the first run.