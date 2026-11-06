#!/usr/bin/env python3
"""Foldseek cluster statistics (thesis Section 3.4).

Reads a two-column easy-cluster TSV (representative <TAB> member) and writes the
clusters with at least --min-size members to robust_clusters.csv.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cluster_tsv", type=Path, help="Foldseek easy-cluster TSV")
    ap.add_argument("--min-size", type=int, default=3, help="robust cluster size (default 3)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("clusters"),
                    help="output folder (default: clusters)")
    args = ap.parse_args()

    if not args.cluster_tsv.exists():
        sys.exit(f"not found: {args.cluster_tsv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cluster_tsv, sep="\t", header=None, names=["representative", "member"])
    sizes = df.groupby("representative").size().sort_values(ascending=False)

    print(f"clusters {len(sizes)}, structures {len(df)}, mean size {sizes.mean():.2f}, "
          f"largest {sizes.max()}")
    for k in sorted({args.min_size, 5}):
        print(f"clusters with >= {k} members: {int((sizes >= k).sum())} "
              f"({100 * (sizes >= k).mean():.1f}%)")

    robust = sizes[sizes >= args.min_size]
    members = df.groupby("representative")["member"].apply(", ".join)
    pd.DataFrame({"representative": robust.index, "size": robust.values,
                  "members": members.reindex(robust.index).values}
                 ).to_csv(args.out_dir / "robust_clusters.csv", index=False)


if __name__ == "__main__":
    main()
