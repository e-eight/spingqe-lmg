"""Merge multiple benchmark CSV files into one, deduplicating by row key.

Usage:
  python scripts/merge-benchmark-csvs.py backend-incr.csv backend-peven.csv --output merged.csv
  python scripts/merge-benchmark-csvs.py backend-*.csv --output merged.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def row_key(row: dict) -> tuple:
    return (
        row.get("benchmark", ""),
        int(row.get("n_qubits", 0)),
        row.get("backend", ""),
        int(row.get("n_jobs", 0)),
        int(row.get("seq_len", 0)),
        int(row.get("rep", 0)),
    )


def merge_csvs(input_paths: list[str], output_path: str) -> int:
    merged: dict[tuple, dict] = {}
    fieldnames: list[str] = []

    for path in input_paths:
        if not os.path.exists(path):
            print(f"WARNING: {path} does not exist, skipping.", file=sys.stderr)
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                key = row_key(row)
                if key in merged:
                    print(
                        f"WARNING: duplicate key {key} -- keeping later row from {path}",
                        file=sys.stderr,
                    )
                merged[key] = row

    if not merged:
        print("ERROR: no rows to write.", file=sys.stderr)
        return 1

    sorted_rows = sorted(merged.values(), key=row_key)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Merged {len(input_paths)} files -> {len(sorted_rows)} rows written to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csvs",
        nargs="+",
        metavar="CSV",
        help="one or more benchmark CSV files to merge",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="output CSV path",
    )
    args = parser.parse_args(argv)

    sys.exit(merge_csvs(args.input_csvs, args.output))


if __name__ == "__main__":
    main()
