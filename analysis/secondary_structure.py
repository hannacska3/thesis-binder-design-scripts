#!/usr/bin/env python3
"""Secondary structure and topology of the final designs (thesis Section 3.4).

Reads the DE-STRESS design_data.csv of the isolated binders (design_name,
ss_prop_*, dssp_assignment). Segments are runs in the DSSP string: helices are
at least MIN_HELIX_LEN of H/G/I, strands at least MIN_STRAND_LEN of E/B.
Topology is helical (helix fraction >= HELICAL_MIN), sheet-containing (strand
fraction >= SHEET_MIN), mixed (both) or other. A Foldseek easy-cluster TSV
(representative <TAB> member) is optional.
"""

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

MIN_HELIX_LEN = 5
MIN_STRAND_LEN = 3
HELICAL_MIN = 0.40
SHEET_MIN = 0.10

CASE_ORDER = ["interface", "full", "upscaled"]
BRANCH_RE = re.compile(r"_(interface|full)(?:_mpnn\d+_\d+)?$", re.IGNORECASE)
UPSCALED_RE = re.compile(r"_(?:interface|full)_mpnn\d+_\d+$", re.IGNORECASE)


def assign_case(name: str) -> str:
    name = re.sub(r"\.(pdb|cif|ent)$", "", str(name).strip(), flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if UPSCALED_RE.search(name):
        return "upscaled"
    m = BRANCH_RE.search(name)
    return m.group(1).lower() if m else "unknown"


def count_segments(dssp: str, codes: str, min_len: int) -> int:
    """Number of runs of `codes` at least min_len long in a DSSP string."""
    if not isinstance(dssp, str) or not dssp:
        return 0
    n, run = 0, 0
    for ch in dssp + "\0":
        if ch in codes:
            run += 1
        else:
            if run >= min_len:
                n += 1
            run = 0
    return n


def classify(row) -> str:
    helix, strand = row["frac_helix"], row["frac_strand"]
    if strand >= SHEET_MIN and helix >= HELICAL_MIN:
        return "mixed"
    if strand >= SHEET_MIN:
        return "sheet-containing"
    if helix >= HELICAL_MIN:
        return "helical"
    return "other"


def load_foldseek(path: Path) -> dict:
    """member -> representative, both by file stem."""
    mapping = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[Path(parts[1]).stem] = Path(parts[0]).stem
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("binder_csv", type=Path, help="DE-STRESS design_data.csv, isolated binders")
    ap.add_argument("--foldseek-tsv", type=Path, help="Foldseek easy-cluster TSV")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("secondary_structure"),
                    help="output folder (default: secondary_structure)")
    args = ap.parse_args()

    for f in (args.binder_csv, args.foldseek_tsv):
        if f is not None and not f.exists():
            sys.exit(f"not found: {f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.binder_csv)
    helix_cols = [c for c in ("ss_prop_alpha_helix", "ss_prop_3_10_helix") if c in df.columns]
    strand_cols = [c for c in ("ss_prop_beta_strand", "ss_prop_beta_bridge") if c in df.columns]
    if not helix_cols or not strand_cols or "dssp_assignment" not in df.columns:
        sys.exit("missing ss_prop_* or dssp_assignment columns")

    df["case"] = df["design_name"].map(assign_case)
    df["frac_helix"] = df[helix_cols].sum(axis=1)
    df["frac_strand"] = df[strand_cols].sum(axis=1)
    df["frac_loop"] = df.get("ss_prop_loop", 0)
    df["n_helices"] = df["dssp_assignment"].map(lambda s: count_segments(s, "HGI", MIN_HELIX_LEN))
    df["n_strands"] = df["dssp_assignment"].map(lambda s: count_segments(s, "EB", MIN_STRAND_LEN))
    df["topology"] = df.apply(classify, axis=1)

    keep = ["design_name", "case", "frac_helix", "frac_strand", "frac_loop",
            "n_helices", "n_strands", "topology"]
    if args.foldseek_tsv is not None:
        cluster_map = load_foldseek(args.foldseek_tsv)
        df["cluster"] = df["design_name"].map(lambda n: cluster_map.get(Path(str(n)).stem))
        keep.append("cluster")
    df[keep].to_csv(args.out_dir / "secondary_structure.csv", index=False)

    n = len(df)
    print(f"helix {df.frac_helix.mean():.3f} (median {df.frac_helix.median():.3f}), "
          f"strand {df.frac_strand.mean():.3f} (median {df.frac_strand.median():.3f}), "
          f"loop {df.frac_loop.mean():.3f}")
    print("topology: " + ", ".join(f"{k} {v} ({100 * v / n:.1f}%)"
                                   for k, v in df.topology.value_counts().items()))
    print(f"helical segments (>= {MIN_HELIX_LEN} residues) per design: " + ", ".join(
        f"{k}: {v} ({100 * v / n:.1f}%)" for k, v in df.n_helices.value_counts().sort_index().items()))
    three = int((df.n_helices == 3).sum())
    any_strand = int((df.n_strands > 0).sum())
    print(f"exactly 3 helices: {three} ({100 * three / n:.1f}%), "
          f"any strand segment (>= {MIN_STRAND_LEN}): {any_strand} ({100 * any_strand / n:.1f}%)")

    print(f"{'case':<11}{'n':>5}{'helix':>8}{'strand':>8}{'helical':>9}")
    groups = [(c, df[df.case == c]) for c in CASE_ORDER if (df.case == c).any()]
    for name, g in groups:
        print(f"{name:<11}{len(g):>5}{g.frac_helix.mean():>8.3f}{g.frac_strand.mean():>8.3f}"
              f"{100 * (g.topology == 'helical').mean():>8.1f}%")

    if args.foldseek_tsv is not None:
        sizes = df.cluster.value_counts()
        print(f"Foldseek clusters {len(sizes)} (3+ members: {int((sizes >= 3).sum())}, "
              f"largest {sizes.max() if len(sizes) else 0}), "
              f"joined {int(df.cluster.notna().sum())}/{n} designs")


if __name__ == "__main__":
    main()
