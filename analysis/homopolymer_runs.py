#!/usr/bin/env python3
"""Homopolymer runs in the designed sequences (thesis Section 3.3).

Reads the DE-STRESS design_data.csv (design_name, full_sequence,
isoelectric_point) and writes one row per run of at least --min-run-len
identical residues to homopolymer_runs.csv. The output holds no sequences.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd


def find_runs(sequence: str, min_len: int):
    """(residue, run length, start position) for each run of min_len or more."""
    return [(m.group(1), len(m.group(0)), m.start())
            for m in re.finditer(r"(.)\1*", sequence)
            if len(m.group(0)) >= min_len]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("design_csv", type=Path, help="DE-STRESS design_data.csv")
    ap.add_argument("--min-run-len", type=int, default=3,
                    help="shortest run to report (default 3)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("homopolymers"),
                    help="output folder (default: homopolymers)")
    args = ap.parse_args()

    if not args.design_csv.exists():
        sys.exit(f"not found: {args.design_csv}")
    df = pd.read_csv(args.design_csv)
    if not {"design_name", "full_sequence"} <= set(df.columns):
        sys.exit("missing columns: design_name, full_sequence")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [{"design_name": name, "residue": aa, "run_length": length, "start_pos": pos}
            for name, seq in zip(df["design_name"], df["full_sequence"])
            if isinstance(seq, str)
            for aa, length, pos in find_runs(seq, args.min_run_len)]
    runs = pd.DataFrame(rows, columns=["design_name", "residue", "run_length", "start_pos"])
    runs.to_csv(args.out_dir / "homopolymer_runs.csv", index=False)

    n = len(df)
    with_run = runs.design_name.nunique()
    print(f"designs {n}, with a run of {args.min_run_len}+: {with_run} "
          f"({100 * with_run / n:.1f}%), runs {len(runs)}")
    if len(runs):
        print("runs by residue: " + ", ".join(
            f"{aa} {c} ({100 * c / len(runs):.1f}%)"
            for aa, c in runs.residue.value_counts().items()))
    if "isoelectric_point" in df.columns:
        pi = df["isoelectric_point"].dropna()
        print(f"pI: median {pi.median():.2f}, range {pi.min():.2f} to {pi.max():.2f}")


if __name__ == "__main__":
    main()
