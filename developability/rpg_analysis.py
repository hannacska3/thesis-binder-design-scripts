#!/usr/bin/env python3
"""Exposure-filtered protease cleavage sites (thesis Section 4.3).

Reads an RPG digest (-f csv, with Original_header, Enzyme, Cleaving_pos and
Peptide_size) and the per_residue.csv written by sasa_analysis.py, and joins each
cleavage site to the accessibility of its P1 residue. The last fragment of each
digest has no cleavage after it, so the site at the sequence end is dropped.
"""

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

TERMINAL_WINDOW = 10


def normalise_key(name: str) -> str:
    key = str(name).strip().split()[0] if str(name).strip() else ""
    key = re.sub(r"\.(pdb|cif|ent|fasta|fa)$", "", key, flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return re.sub(r"_[A-Z]$", "", key)


def load_digest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = {"Original_header", "Enzyme", "Cleaving_pos", "Peptide_size"} - set(df.columns)
    if missing:
        sys.exit(f"digest is missing columns: {sorted(missing)}")
    df["design"] = df["Original_header"].map(normalise_key)
    df["Cleaving_pos"] = pd.to_numeric(df["Cleaving_pos"], errors="coerce")
    return df


def sequence_lengths(df: pd.DataFrame) -> pd.Series:
    """Protein length per design: the fragment lengths of one digest sum to it."""
    return (df.groupby(["design", "Enzyme"])["Peptide_size"].sum()
              .groupby("design").median())


def cleavage_sites(df: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    sites = df.dropna(subset=["Cleaving_pos"]).copy()
    sites["length"] = sites.design.map(lengths)
    sites = sites[sites.Cleaving_pos < sites.length].copy()
    sites["from_end"] = np.minimum(sites.Cleaving_pos,
                                   sites.length - sites.Cleaving_pos)
    sites["terminal"] = sites.from_end <= TERMINAL_WINDOW
    return sites


def add_exposure(sites: pd.DataFrame, per_residue_path: Path):
    pr = pd.read_csv(per_residue_path)
    pr["design"] = pr["design"].map(normalise_key)
    pr = pr[["design", "pos", "exposed", "rsa_free", "interface"]]
    sites = sites.merge(pr, left_on=["design", "Cleaving_pos"],
                        right_on=["design", "pos"], how="left")
    matched = int(sites.exposed.notna().sum())
    if matched == 0:
        sys.exit("no cleavage site matched the per-residue table: check that "
                 "design names and residue numbering agree")
    for col in ("exposed", "interface"):
        sites[col] = sites[col].fillna(False).astype(bool)
    return sites, matched


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("digest_csv", type=Path, help="RPG digest (-f csv)")
    ap.add_argument("per_residue_csv", type=Path,
                    help="per_residue.csv from sasa_analysis.py")
    ap.add_argument("-o", "--out-dir", type=Path, default=None,
                    help="output folder (default: next to the digest)")
    args = ap.parse_args()

    for f in (args.digest_csv, args.per_residue_csv):
        if not f.exists():
            sys.exit(f"not found: {f}")
    out_dir = args.out_dir or args.digest_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_digest(args.digest_csv)
    sites = cleavage_sites(df, sequence_lengths(df))
    sites, matched = add_exposure(sites, args.per_residue_csv)

    per = (sites.groupby(["design", "Enzyme"])
                .agg(n_sites=("Cleaving_pos", "size"),
                     n_terminal=("terminal", "sum"),
                     n_exposed=("exposed", "sum"))
                .reset_index())

    cols = ["design", "Enzyme", "Cleaving_pos", "Peptide_size", "from_end",
            "terminal", "exposed", "rsa_free", "interface"]
    sites[cols].to_csv(out_dir / "all_sites.csv", index=False)
    per.to_csv(out_dir / "sites_per_design.csv", index=False)

    print(f"{'enzyme':<28}{'sites':>7}{'exposed':>9}{'surviving':>11}")
    for enz, g in per.groupby("Enzyme"):
        frac = 100 * sites[sites.Enzyme == enz].exposed.mean()
        print(f"{str(enz):<28}{g.n_sites.median():>7.0f}"
              f"{g.n_exposed.median():>9.0f}{frac:>10.1f}%")

    exp = sites[sites.exposed]
    tot = per.groupby("design").n_sites.sum()
    exp_per_design = exp.groupby("design").size().reindex(tot.index, fill_value=0)
    print(f"sites {len(sites)} (median {tot.median():.0f} per design), "
          f"exposed {len(exp)} ({100 * len(exp) / len(sites):.1f}%, "
          f"median {exp_per_design.median():.0f} per design)")
    print(f"exposed sites at the interface {100 * exp.interface.mean():.1f}%, "
          f"within {TERMINAL_WINDOW} residues of a terminus "
          f"{100 * exp.terminal.mean():.1f}%")
    print(f"exposure joined for {matched}/{len(sites)} sites")


if __name__ == "__main__":
    main()