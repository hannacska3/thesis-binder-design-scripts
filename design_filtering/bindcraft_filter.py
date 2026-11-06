#!/usr/bin/env python3
"""BindCraft default interface filters on interface_analysis.csv (thesis Section 3.5).

Thresholds follow BindCraft's default_filters.json, with the shape
complementarity cut adjustable (--sc-min; 0.6 is the default, 0.4 the relaxed
setting). Designs passing every filter are copied from
<pdb_dir>/<design>/<design>*.pdb to --copy-to, with an optional label appended
to the file name.
"""

import argparse
import ast
import csv
import glob
import os
import shutil
import sys


def to_float(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def lys_met(raw):
    """(LYS count, MET count) from the InterfaceAAs field, or None."""
    try:
        d = ast.literal_eval(raw)
        return d.get("LYS", 0), d.get("MET", 0)
    except (ValueError, SyntaxError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv_path", help="interface_analysis.csv")
    ap.add_argument("pdb_dir", help="folder holding <design>/<design>*.pdb")
    ap.add_argument("--copy-to", help="copy passing PDBs into this folder")
    ap.add_argument("--label", default="", help="suffix added to copied file names, e.g. full")
    ap.add_argument("--sc-min", type=float, default=0.6, help="shape complementarity cut (default 0.6)")
    args = ap.parse_args()

    if not os.path.isfile(args.csv_path):
        sys.exit(f"not found: {args.csv_path}")
    with open(args.csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    filters = [
        ("Binder_Energy_Score < 0", "Binder_Energy_Score", lambda v: v < 0),
        ("Surface_Hydrophobicity < 0.35", "Surface_Hydrophobicity", lambda v: v < 0.35),
        (f"ShapeComplementarity > {args.sc_min:g}", "ShapeComplementarity", lambda v: v > args.sc_min),
        ("dG < 0", "dG", lambda v: v < 0),
        ("dSASA > 1", "dSASA", lambda v: v > 1),
        ("n_InterfaceResidues > 7", "n_InterfaceResidues", lambda v: v > 7),
        ("Binder_Loop% < 90", "Binder_Loop%", lambda v: v < 90),
    ]
    missing = [c for _, c, _ in filters + [("", "InterfaceAAs", None)]
               if rows and c not in rows[0]]
    if missing:
        sys.exit(f"missing columns: {missing}")

    def checks(row):
        out = []
        for label, col, ok in filters:
            v = to_float(row[col])
            out.append((label, v is not None and ok(v)))
        km = lys_met(row["InterfaceAAs"])
        out.append(("InterfaceAAs: Lys < 3, Met < 3", km is not None and km[0] < 3 and km[1] < 3))
        return out

    results = [checks(r) for r in rows]
    n = len(rows)
    for i, (label, _) in enumerate(results[0] if results else []):
        print(f"{label:<34}{sum(r[i][1] for r in results):>7}/{n}")
    passing = [r["design"] for r, res in zip(rows, results) if all(ok for _, ok in res)]
    print(f"{'all filters':<34}{len(passing):>7}/{n} ({100 * len(passing) / max(n, 1):.1f}%)")

    if args.copy_to and passing:
        os.makedirs(args.copy_to, exist_ok=True)
        copied = 0
        for design in passing:
            hits = sorted(glob.glob(os.path.join(args.pdb_dir, design, f"{design}*.pdb")))
            if not hits:
                continue
            base, ext = os.path.splitext(os.path.basename(hits[0]))
            name = f"{base}_{args.label}{ext}" if args.label else base + ext
            shutil.copy2(hits[0], os.path.join(args.copy_to, name))
            copied += 1
        print(f"copied {copied}/{len(passing)} PDBs to {args.copy_to}")


if __name__ == "__main__":
    main()
