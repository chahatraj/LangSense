#!/usr/bin/env python3
"""
Rebuild the combined synthetic summary CSV from individual *_summary.csv files.

- Scans a directory (default: results/inference/synthetic_eval) for per-dataset summary CSVs.
- Concatenates them (skipping the existing all_synthetic_summary.csv) and writes a fresh combined file.
"""

import argparse
import csv
import os
from typing import List


def list_summary_files(directory: str) -> List[str]:
    files = []
    for name in os.listdir(directory):
        if not name.endswith("_summary.csv"):
            continue
        if name == "all_synthetic_summary.csv":
            continue
        files.append(os.path.join(directory, name))
    return sorted(files)


def read_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(rows: List[dict], path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote combined summary to {path} ({len(rows)} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild combined synthetic summary CSV")
    parser.add_argument(
        "--directory",
        default="/scratch/craj/langsense/results/inference/synthetic_eval",
        help="Directory containing per-dataset *_summary.csv files",
    )
    parser.add_argument(
        "--output",
        default="/scratch/craj/langsense/results/inference/synthetic_eval/all_synthetic_summary.csv",
        help="Path to write the combined summary CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_files = list_summary_files(args.directory)
    if not summary_files:
        raise SystemExit(f"No *_summary.csv files found in {args.directory}")

    all_rows = []
    fieldnames = None
    for path in summary_files:
        rows = read_rows(path)
        if not rows:
            continue
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {path}")

    if fieldnames is None:
        raise SystemExit("No rows loaded from summary files.")

    write_rows(all_rows, args.output, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
