"""
Merge Colosseum-V2 evaluation results CSVs and keep every non-duplicate row.

Expects the schema written by `run_mass_eval.py` / `eval_rgbd.py`:

checkpoint_path,pc_hostname,now,t_final,duration_sec,perturbation_set,env_id,
control_mode,include_depth,num_eval_episodes,max_episode_steps,message,
num_sucessful_episodes,success_percent

A row is a duplicate if it matches an already-kept row on the evaluation
identity keys (same rule as `parse_logs.row_exists`). The first occurrence
is kept; later copies are dropped. Rows that differ on any of those keys
are preserved.

### Example usage:
python scripts/colosseum_v2_paper/merge_logs.py \
    --results-paths outputs/run_a/results.csv outputs/run_b/results.csv \
    --output-path outputs/merged/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Identity keys used by parse_logs.row_exists (order-independent of CSV schema).
DUPLICATE_KEYS = [
    "checkpoint_path",
    "pc_hostname",
    "now",
    "perturbation_set",
    "env_id",
    "control_mode",
    "include_depth",
    "num_eval_episodes",
    "max_episode_steps",
    "message",
    "num_sucessful_episodes",
]


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(k, "")).strip() for k in DUPLICATE_KEYS)


def merge_logs(results_paths: list[Path]) -> tuple[list[str], list[dict[str, str]], int]:
    """Load CSVs in order, keep first occurrence of each unique identity key.

    Returns:
        (fieldnames, kept_rows, n_duplicates_dropped)
    """
    fieldnames: list[str] | None = None
    kept: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    n_dropped = 0

    for path in results_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: empty CSV or missing header")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            else:
                for name in reader.fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)

            for row in reader:
                key = row_key(row)
                if key in seen:
                    n_dropped += 1
                    continue
                seen.add(key)
                kept.append(row)

    if fieldnames is None:
        raise ValueError("No input CSVs provided")

    return fieldnames, kept, n_dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge evaluation results CSVs, keeping non-duplicate rows."
    )
    parser.add_argument(
        "--results-paths",
        nargs="+",
        type=Path,
        required=True,
        help="One or more evaluation results CSV paths (merged in order)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to write the merged CSV",
    )
    args = parser.parse_args()

    for path in args.results_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    fieldnames, kept, n_dropped = merge_logs(args.results_paths)
    n_input = len(kept) + n_dropped

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)

    print(f"Input files: {len(args.results_paths)}")
    print(f"Input rows: {n_input}")
    print(f"Duplicate rows dropped: {n_dropped}")
    print(f"Output rows: {len(kept)}")
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
