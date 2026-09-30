from pathlib import Path
from time import perf_counter

import subprocess
import sys

import polars as pl

import domherre as kt
from domherre.backends import resolve

HERE = Path(__file__).parent
INPUT_FILE = HERE / "test_strings.txt"
REFERENCE_FILE = HERE / "test_strings.jsonl"
STATS_FILE = HERE / "stats_file.txt"

MODELS = {
    "kb_bert": kt.Config(
        backend="kb-bert",
        model="KB/bert-base-swedish-cased-ner",
        redact_personnummer=False,
        redact_email=False,
    ),
    "gliner_pii": kt.Config(
        backend="gliner",
        model="urchade/gliner_multi_pii-v1",
        labels=("person", "full name"),
        min_score=0.45,
        batch_size=128,
        backend_options={"map_location": "cpu"},
        redact_personnummer=False,
        redact_email=False,
    ),
    "gliner_multi_v2_1": kt.Config(
        backend="gliner",
        model="urchade/gliner_multi-v2.1",
        labels=("person",),
        min_score=0.45,
        batch_size=128,
        backend_options={"map_location": "cpu"},
        redact_personnummer=False,
        redact_email=False,
    ),
}

# Function to evaluate a single model on the given texts and return timing information
def evaluate(
    name: str,
    config: kt.Config,
    texts: list[str],
) -> dict[str, float]:
    output_file = HERE / f"results_{name}.txt"

    kt.clear_cache()

    print(f"\nLoading {name}...")
    backend = resolve(config.backend, config.backend_config())
    backend.load()

    print(f"Running {name}...")
    started = perf_counter()

    frame = pl.DataFrame({"text": texts})
    redacted = kt.redact(frame, "text", config)
    results = redacted["text_redacted"].to_list()

    elapsed = perf_counter() - started
    throughput = len(texts) / elapsed if elapsed else 0.0

    output_file.write_text(
        "\n".join(results) + "\n",
        encoding="utf-8",
    )

    print(f"{name}:")
    print(f"  strings: {len(texts)}")
    print(f"  seconds: {elapsed:.2f}")
    print(f"  strings/second: {throughput:.2f}")
    print(f"  output: {output_file}")

    return {
        "seconds": elapsed,
        "strings_per_second": throughput,
    }

# Function to check the results of the models against the test cases
def check_results(timings: dict[str, dict[str, float]]) -> None:
    result_files = [
        HERE / f"results_{name}.txt"
        for name in MODELS
    ]

    completed = subprocess.run(
        [
            sys.executable,
            str(HERE / "check_false_positives.py"),
            "--tests",
            str(REFERENCE_FILE),
            *(str(path) for path in result_files),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    timing_lines = ["", "=== Timing ==="]

    for name, timing in timings.items():
        timing_lines.extend(
            [
                f"{name}:",
                f"  seconds: {timing['seconds']:.2f}",
                (
                    "  strings/second: "
                    f"{timing['strings_per_second']:.2f}"
                ),
            ]
        )

    STATS_FILE.write_text(
        completed.stdout + "\n".join(timing_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Statistics written to: {STATS_FILE}")

# Main entry point for the script
def main() -> None:
    texts = INPUT_FILE.read_text(encoding="utf-8").splitlines()
    timings = {}
    
    print(f"Loaded {len(texts)} test strings.")
    print("Evaluating models...")

    for name, config in MODELS.items():
        timings[name] = evaluate(name, config, texts)

    print("\nChecking model results...")
    check_results(timings)

    print("\nTimings:")
    for name, timing in timings.items():
        print(f"  {name}: {timing['strings_per_second']:.2f} strings/second")

# Entry point check for the script
if __name__ == "__main__":
    main()
