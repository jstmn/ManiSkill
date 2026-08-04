"""
Remove placeholder / incomplete evaluation rows from Colosseum-V2 results CSVs.

`eval_rgbd.py` writes a placeholder row (message="placeholder", num_sucessful_episodes=-1) before each run. If the run 
crashes, that row is left behind and blocks resume. This script deletes those rows in-place.

A row is treated as a placeholder if either:
  - message == "placeholder" (case-insensitive), or
  - num_sucessful_episodes < 0

Rows with message == "variation_factor_disabled" are kept even when
num_sucessful_episodes < 0 (those are completed skips, not placeholders).

### Example usage:
python scripts/colosseum_v2_paper/remove_placeholder_rows.py logs/act_clip/single_arm.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _safe_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def is_placeholder_row(row: dict[str, str]) -> bool:
    message = str(row.get("message", "")).strip().lower()
    # Completed skips (variation not applicable for this env) — keep them.
    if message == "variation_factor_disabled":
        return False
    if message == "placeholder":
        return True
    n_success = _safe_int(row.get("num_sucessful_episodes"))
    return n_success is not None and n_success < 0


def remove_placeholder_rows(path: Path, *, dry_run: bool) -> tuple[int, int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV or missing header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    kept = [row for row in rows if not is_placeholder_row(row)]
    n_removed = len(rows) - len(kept)

    if n_removed == 0:
        print(f"{path}: no placeholder rows ({len(rows)} total)")
        return 0, len(rows)

    print(f"{path}: removing {n_removed}/{len(rows)} placeholder row(s)")
    if dry_run:
        for row in rows:
            if is_placeholder_row(row):
                print(
                    f"  would remove: env_id={row.get('env_id')!r} "
                    f"perturbation_set={row.get('perturbation_set')!r} "
                    f"message={row.get('message')!r} "
                    f"num_sucessful_episodes={row.get('num_sucessful_episodes')!r}"
                )
        return n_removed, len(rows)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    return n_removed, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove placeholder evaluation rows from results CSVs."
    )
    parser.add_argument(
        "results_paths",
        nargs="+",
        type=Path,
        help="One or more evaluation results CSV paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rows that would be removed without writing",
    )
    args = parser.parse_args()

    total_removed = 0
    for path in args.results_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        n_removed, _ = remove_placeholder_rows(path, dry_run=args.dry_run)
        total_removed += n_removed

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {total_removed} placeholder row(s) across {len(args.results_paths)} file(s)")


if __name__ == "__main__":
    main()
