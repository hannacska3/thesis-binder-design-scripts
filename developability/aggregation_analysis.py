#!/usr/bin/env python3
"""Aggregation propensity and isoelectric point of the free binders (thesis Section 4.2).

Reads the DE-STRESS design_data.csv of the isolated binders (design_name,
aggrescan3d_*_value, isoelectric_point, ...). A design is patch-prone if its
Aggrescan3D maximum exceeds PATCH_MAX. The complex file is optional and only
shows that the in-complex maximum describes the receptor. The runs file is
optional (columns design, residue; one row per homopolymer run) and tests
whether glutamate runs lower the isoelectric point.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

PATCH_MAX = 2.0
FORMULATION_PH = 7.4
PI_NEAR = 0.2
PI_WIDE = 1.0
PI_BASIC = 9.0

CASE_ORDER = ["interface", "full", "upscaled"]
BRANCH_RE = re.compile(r"_(interface|full)(?:_mpnn\d+_\d+)?$", re.IGNORECASE)
UPSCALED_RE = re.compile(r"_(?:interface|full)_mpnn\d+_\d+$", re.IGNORECASE)


def normalise_key(name: str) -> str:
    key = str(name).strip()
    key = re.sub(r"\.(pdb|cif|ent)$", "", key, flags=re.IGNORECASE)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("binder_csv", type=Path, help="DE-STRESS design_data.csv, isolated binders")
    ap.add_argument("--complex-csv", type=Path, help="DE-STRESS design_data.csv, complexes")
    ap.add_argument("--runs-csv", type=Path, help="homopolymer runs (design, residue)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("aggregation"),
                    help="output folder (default: aggregation)")
    args = ap.parse_args()

    for f in (args.binder_csv, args.complex_csv, args.runs_csv):
        if f is not None and not f.exists():
            sys.exit(f"not found: {f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    binder = pd.read_csv(args.binder_csv)
    missing = {"design_name", "aggrescan3d_max_value", "isoelectric_point"} - set(binder.columns)
    if missing:
        sys.exit(f"missing columns: {sorted(missing)}")

    df = binder.add_suffix("_binder")
    df["join_key"] = binder["design_name"].map(normalise_key)
    df["case"] = binder["design_name"].map(assign_case)

    b = "aggrescan3d_max_value_binder"
    avg = "aggrescan3d_avg_value_binder"
    pi = "isoelectric_point_binder"
    df["patch_prone"] = df[b] > PATCH_MAX
    df["pi_distance"] = (df[pi] - FORMULATION_PH).abs()

    keep = [c for c in [
        "design_name_binder", "case", b, "aggrescan3d_total_value_binder", avg,
        "patch_prone", "hydrophobic_fitness_binder", "packing_density_binder",
        pi, "pi_distance", "charge_binder", "num_residues_binder",
    ] if c in df.columns]
    df[keep].to_csv(args.out_dir / "aggregation_properties.csv", index=False)

    print(f"{'case':<11}{'n':>5}{'max (mean)':>12}{'patch-prone':>13}")
    groups = [(c, df[df.case == c]) for c in CASE_ORDER if (df.case == c).any()]
    for name, g in groups + [("all", df)]:
        print(f"{name:<11}{len(g):>5}{g[b].mean():>12.3f}{100 * g.patch_prone.mean():>12.1f}%")

    print(f"max: median {df[b].median():.3f}, range {df[b].min():.3f} to {df[b].max():.3f}; "
          f"patch-prone (max > {PATCH_MAX:g}): {int(df.patch_prone.sum())}")
    if avg in df.columns:
        print(f"per-residue average: mean {df[avg].mean():+.3f}, median {df[avg].median():+.3f}")

    if args.complex_csv is not None:
        comp = pd.read_csv(args.complex_csv)
        cmax = comp.set_index(comp["design_name"].map(normalise_key))["aggrescan3d_max_value"]
        cmax = cmax[~cmax.index.duplicated()]
        c = df["join_key"].map(cmax).dropna()
        print(f"in-complex max: {c.min():.3f} to {c.max():.3f} "
              f"(binder max exceeds it in {int((df.loc[c.index, b] > c).sum())} of {len(c)})")

    n_near = int((df.pi_distance < PI_NEAR).sum())
    n_wide = int((df.pi_distance < PI_WIDE).sum())
    n_basic = int((df[pi] > PI_BASIC).sum())
    print(f"pI: median {df[pi].median():.2f}, range {df[pi].min():.2f} to {df[pi].max():.2f}; "
          f"within {PI_NEAR:g} of {FORMULATION_PH}: {n_near}, within {PI_WIDE:g}: {n_wide}; "
          f"above {PI_BASIC:g}: {n_basic} ({100 * n_basic / len(df):.1f}%)")

    if args.runs_csv is not None:
        runs = pd.read_csv(args.runs_csv)
        parts = []
        for label, aa in (("Glu", "E"), ("Ala", "A")):
            has = df["join_key"].isin(set(runs.loc[runs["residue"] == aa, "design"]
                                          .map(normalise_key)))
            if has.any() and (~has).any():
                parts.append(f"{label} run: median pI {df.loc[has, pi].median():.2f} "
                             f"(n={int(has.sum())}) vs {df.loc[~has, pi].median():.2f} "
                             f"(n={int((~has).sum())})")
        print("; ".join(parts))


if __name__ == "__main__":
    main()