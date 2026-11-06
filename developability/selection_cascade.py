#!/usr/bin/env python3
"""Final selection cascade over the developability filters (thesis Section 4).

Applied in order: Aggrescan3D max <= A3D_MAX, |pI - 7.4| >= PI_MIN, no unpaired
cysteine (even Cys count), no MHC-II epitope core. Writes the surviving design
names (no sequences) and the cysteine counts.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

A3D_MAX = 2.0
FORMULATION_PH = 7.4
PI_MIN = 0.2


def normalise_key(name: str) -> str:
    key = re.sub(r"\.(pdb|cif|ent)$", "", str(name).strip(), flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("binder_csv", type=Path, help="DE-STRESS design_data.csv, isolated binders")
    ap.add_argument("immuno_csv", type=Path, help="per_design_immunogenicity.csv")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("selection"),
                    help="output folder (default: selection)")
    args = ap.parse_args()

    for f in (args.binder_csv, args.immuno_csv):
        if not f.exists():
            sys.exit(f"not found: {f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.binder_csv)
    need = {"design_name", "full_sequence", "aggrescan3d_max_value", "isoelectric_point"}
    if need - set(df.columns):
        sys.exit(f"missing columns: {sorted(need - set(df.columns))}")
    imm = pd.read_csv(args.immuno_csv)
    imm["key"] = imm["id"].map(normalise_key)
    df["key"] = df["design_name"].map(normalise_key)
    df = df.merge(imm[["key", "n_epitopes"]].drop_duplicates("key"), on="key", how="left")
    if df["n_epitopes"].isna().all():
        sys.exit("no design matched the immunogenicity table")
    df["n_epitopes"] = df["n_epitopes"].fillna(0)
    df["n_cys"] = df["full_sequence"].str.count("C")

    steps = [
        (f"Aggrescan3D max <= {A3D_MAX:g}", df["aggrescan3d_max_value"] <= A3D_MAX),
        (f"|pI - {FORMULATION_PH}| >= {PI_MIN:g}", (df["isoelectric_point"] - FORMULATION_PH).abs() >= PI_MIN),
        ("no unpaired cysteine", df["n_cys"] % 2 == 0),
        ("no MHC-II epitope core", df["n_epitopes"] == 0),
    ]
    keep = pd.Series(True, index=df.index)
    print(f"{'filter':<28}{'removed':>9}{'left':>7}\nstart{'':<27}{len(df):>7}")
    for label, ok in steps:
        before = int(keep.sum())
        keep &= ok
        print(f"{label:<28}{before - int(keep.sum()):>9}{int(keep.sum()):>7}")

    df.loc[keep, ["design_name", "n_cys", "aggrescan3d_max_value", "isoelectric_point"]].to_csv(
        args.out_dir / "selected.csv", index=False)
    print(f"cysteines among survivors: {df.loc[keep, 'n_cys'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()