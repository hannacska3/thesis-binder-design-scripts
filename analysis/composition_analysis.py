#!/usr/bin/env python3
"""Sequence composition after redesign and low-complexity runs (thesis Section 3.3).

Joins the DE-STRESS design_data.csv of the isolated binders and of the
complexes on the normalised design name, then reports Ala/Glu content and
homopolymer runs by redesign branch (interface/full) and round (round1/upscaled).
Optional: FASTA of the source scaffolds as a composition baseline.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

AA3_TO_1 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
            "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
            "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
            "TRP": "W", "TYR": "Y"}
BRANCH_RE = re.compile(r"_(interface|full)(?:_mpnn\d+_\d+)?$", re.IGNORECASE)
UPSCALED_RE = re.compile(r"_(?:interface|full)_mpnn\d+_\d+$", re.IGNORECASE)


def normalise_key(name: str) -> str:
    key = re.sub(r"\.(pdb|cif|ent)$", "", str(name).strip(), flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def longest_runs(seq: str) -> dict:
    """Longest homopolymer run per residue type."""
    runs = {}
    for m in re.finditer(r"(.)\1*", seq):
        runs[m.group(1)] = max(runs.get(m.group(1), 0), len(m.group(0)))
    return runs


def read_fasta(path: Path) -> list:
    seqs, chunks = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if chunks:
                seqs.append("".join(chunks))
            chunks = []
        elif line:
            chunks.append(line)
    if chunks:
        seqs.append("".join(chunks))
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("binder_csv", type=Path, help="DE-STRESS design_data.csv, isolated binders")
    ap.add_argument("complex_csv", type=Path, help="DE-STRESS design_data.csv, complexes")
    ap.add_argument("--scaffold-fasta", type=Path, help="source scaffold sequences (baseline)")
    ap.add_argument("--min-run", type=int, default=4,
                    help="shortest homopolymer run counted (default 4)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("composition"),
                    help="output folder (default: composition)")
    args = ap.parse_args()

    for f in (args.binder_csv, args.complex_csv, args.scaffold_fasta):
        if f is not None and not f.exists():
            sys.exit(f"not found: {f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    binder = pd.read_csv(args.binder_csv)
    cmplx = pd.read_csv(args.complex_csv)
    for df in (binder, cmplx):
        df["join_key"] = df["design_name"].map(normalise_key)
    m = binder.merge(cmplx, on="join_key", how="outer", suffixes=("_binder", "_complex"),
                     indicator=True)
    both = m[m["_merge"] == "both"].drop(columns="_merge").copy()
    if both.empty:
        sys.exit("no designs matched between the two files; check normalise_key()")
    print(f"matched {len(both)}, binder only {int((m['_merge'] == 'left_only').sum())}, "
          f"complex only {int((m['_merge'] == 'right_only').sum())}")

    name = both["design_name_binder"]
    both["branch"] = name.map(lambda n: (BRANCH_RE.search(normalise_key(n)) or [None, "unknown"])[1].lower())
    both["round"] = name.map(lambda n: "upscaled" if UPSCALED_RE.search(normalise_key(n)) else "round1")

    a3d_b, a3d_c = "aggrescan3d_max_value_binder", "aggrescan3d_max_value_complex"
    if a3d_b in both.columns and a3d_c in both.columns:
        both["a3d_max_drop"] = both[a3d_b] - both[a3d_c]
        print(f"Aggrescan3D max, binder minus complex: mean {both.a3d_max_drop.mean():.3f}, "
              f"sd {both.a3d_max_drop.std():.3f}")

    seq_col = "full_sequence_binder"
    comp_cols = {f"composition_{a3}_binder": a1 for a3, a1 in AA3_TO_1.items()
                 if f"composition_{a3}_binder" in both.columns}
    if not comp_cols or seq_col not in both.columns:
        sys.exit("missing composition_*_binder or full_sequence columns")
    aa = list(comp_cols.values())
    comp = both[list(comp_cols)].rename(columns=comp_cols)
    comp["branch"], comp["round"] = both["branch"].values, both["round"].values
    comp.groupby(["branch", "round"])[aa].agg(["mean", "std"]).to_csv(
        args.out_dir / "composition_by_branch_and_round.csv")

    rows = []
    for design, branch, rnd, seq in zip(name, both["branch"], both["round"], both[seq_col]):
        r = longest_runs(seq) if isinstance(seq, str) else {}
        rows.append({"design_name": design, "branch": branch, "round": rnd,
                     "length": len(seq) if isinstance(seq, str) else 0,
                     "longest_A_run": r.get("A", 0), "longest_E_run": r.get("E", 0),
                     "longest_any_run": max(r.values(), default=0)})
    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "low_complexity_runs.csv", index=False)

    drop = [c for c in both.columns if c.startswith("full_sequence")]
    both.drop(columns=drop).to_csv(args.out_dir / "merged_destress.csv", index=False)

    print(f"{'branch':<10}{'round':<10}{'n':>5}{'Ala':>7}{'Glu':>7}"
          f"{'run %':>8}{'max A':>7}{'max E':>7}")
    for (b, r), g in runs.groupby(["branch", "round"]):
        c = comp[(comp.branch == b) & (comp["round"] == r)]
        print(f"{b:<10}{r:<10}{len(g):>5}{c.A.mean():>7.3f}{c.E.mean():>7.3f}"
              f"{100 * (g.longest_any_run >= args.min_run).mean():>7.1f}%"
              f"{g.longest_A_run.max():>7d}{g.longest_E_run.max():>7d}")

    if args.scaffold_fasta is not None:
        seqs = read_fasta(args.scaffold_fasta)
        ala = pd.Series([s.count("A") / len(s) for s in seqs])
        glu = pd.Series([s.count("E") / len(s) for s in seqs])
        print(f"scaffolds {len(seqs)}: Ala {ala.mean():.3f}, Glu {glu.mean():.3f}")


if __name__ == "__main__":
    main()
