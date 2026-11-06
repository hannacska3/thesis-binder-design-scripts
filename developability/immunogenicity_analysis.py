#!/usr/bin/env python3
"""Epitope counts from ImmunoGeNN output (thesis Section 4.1).

Input is one row per 15-mer window (id, peptide_pos, core_pos, core_seq, pIRS,
pIRS_rank, core_0..core_6). Windows at or above RANK_HIGH are collapsed onto their
9-mer cores, and cores within MERGE_GAP residues of each other are merged into one
epitope.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

IMMUNO_CSV = Path("pIRS.csv")
OUT_DIR = Path("analysis")

RANK_HIGH = 95.0
MERGE_GAP = 5

CASE_ORDER = ["interface", "full", "upscaled"]
BRANCH_RE = re.compile(r"_(interface|full)(?:_mpnn\d+_\d+)?$", re.IGNORECASE)
UPSCALED_RE = re.compile(r"_(?:interface|full)_mpnn\d+_\d+$", re.IGNORECASE)


def normalise_key(name: str) -> str:
    key = str(name).strip()
    key = re.sub(r"\.(pdb|cif|ent|fasta|fa)$", "", key, flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def assign_case(name: str) -> str:
    name = normalise_key(name)
    if UPSCALED_RE.search(name):
        return "upscaled"
    m = BRANCH_RE.search(name)
    return m.group(1).lower() if m else "unknown"


def collapse_epitopes(sub: pd.DataFrame, gap: int = MERGE_GAP):
    """Merge the above-threshold cores of one design into distinct epitopes,
    keeping the best-scoring core of each group."""
    if sub.empty:
        return []
    sub = sub.sort_values("abs_core_pos")
    groups, current = [], [sub.iloc[0]]
    for _, row in sub.iloc[1:].iterrows():
        if row["abs_core_pos"] - current[-1]["abs_core_pos"] <= gap:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)

    out = []
    for g in groups:
        best = max(g, key=lambda r: r["pIRS_rank"])
        out.append({
            "abs_core_pos": int(best["abs_core_pos"]),
            "core_seq": best.get("core_seq", ""),
            "pIRS": float(best["pIRS"]),
            "pIRS_rank": float(best["pIRS_rank"]),
            "n_windows": len(g),
        })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IMMUNO_CSV.exists():
        sys.exit(f"not found: {IMMUNO_CSV}")
    df = pd.read_csv(IMMUNO_CSV)

    missing = {"id", "pIRS", "pIRS_rank", "peptide_pos", "core_pos"} - set(df.columns)
    if missing:
        sys.exit(f"missing columns: {sorted(missing)}")

    # core_pos is the register offset (0-6) inside the 15-mer window
    df["abs_core_pos"] = df["peptide_pos"] + df["core_pos"]
    threshold = RANK_HIGH / 100.0 if df["pIRS_rank"].max() <= 1.5 else RANK_HIGH

    rows, epi_rows = [], []
    for design, sub in df.groupby("id"):
        epitopes = collapse_epitopes(sub[sub["pIRS_rank"] >= threshold])
        epi_rows += [{"id": design, **e} for e in epitopes]
        rows.append({
            "id": design,
            "case": assign_case(design),
            "n_windows": len(sub),
            "n_epitopes": len(epitopes),
            "max_pIRS": float(sub["pIRS"].max()),
        })
    per = pd.DataFrame(rows)
    epi = pd.DataFrame(epi_rows, columns=["id", "abs_core_pos", "core_seq", "pIRS",
                                          "pIRS_rank", "n_windows"])

    per.to_csv(OUT_DIR / "per_design_immunogenicity.csv", index=False)
    epi.to_csv(OUT_DIR / "epitopes.csv", index=False)

    print(f"{'case':10s} {'n':>5s} {'mean':>6s} {'median':>7s} {'none':>5s} {'max':>4s}")
    groups = [(c, per[per["case"] == c]) for c in CASE_ORDER if (per["case"] == c).any()]
    for name, g in groups + [("all", per)]:
        k = g["n_epitopes"]
        print(f"{name:10s} {len(g):5d} {k.mean():6.2f} {k.median():7.1f} "
              f"{int((k == 0).sum()):5d} {k.max():4d}")

    counts = epi.groupby("core_seq")["id"].nunique()
    shared = counts[counts >= 5]
    n_shared = epi[epi["core_seq"].isin(shared.index)]["id"].nunique()
    print(f"windows {len(df)}, cores {len(epi)}, distinct {len(counts)}, "
          f"in 5+ designs {len(shared)} ({n_shared} designs)")

    cols = [f"core_{i}" for i in range(7) if f"core_{i}" in df.columns]
    if cols:
        regs = df[cols].astype(float)
        top = regs.max(axis=1) / regs.sum(axis=1)
        print(f"top-core probability: mean {top.mean():.1%}, "
              f"at least half in {(top >= 0.5).mean():.1%} of windows")


if __name__ == "__main__":
    main()