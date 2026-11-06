#!/usr/bin/env python3
"""All-against-all BLOSUM62 identity of the designed sequences (thesis Section 3.6).

Global alignment (gap open -11, extend -1); identity is identical columns over the
shorter sequence. Pairs are split by shared scaffold (the UniProt accession in the
design name), and an average-linkage threshold sweep gives the adjusted Rand
index of the clusters against the scaffold labels. Optional finalists.csv
(design_name) gives the finalist submatrix.
"""

import argparse
import itertools
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


def accession(name: str) -> str:
    m = re.match(r"AF-([A-Z0-9]+)-F1", str(name))
    return m.group(1) if m else "unknown"


def identity_matrix(seqs):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score, aligner.mode = -11, -1, "global"
    n = len(seqs)
    ident = np.full((n, n), 100.0)
    for i, j in itertools.combinations(range(n), 2):
        aln = aligner.align(seqs[i], seqs[j])[0]
        matches = sum(1 for x, y in zip(aln[0], aln[1]) if x == y and x != "-")
        ident[i, j] = ident[j, i] = 100.0 * matches / min(len(seqs[i]), len(seqs[j]))
    return ident


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("design_csv", type=Path, help="design_data.csv (design_name, full_sequence)")
    ap.add_argument("--finalists", type=Path, help="finalists.csv with a design_name column")
    ap.add_argument("--matrix", type=Path, help="reuse a pairwise_identity.csv from an earlier run")
    ap.add_argument("--dup-threshold", type=float, default=90.0,
                    help="identity counted as a near-duplicate (default 90)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("blosum"),
                    help="output folder (default: blosum)")
    args = ap.parse_args()

    for f in (args.design_csv, args.finalists, args.matrix):
        if f is not None and not f.exists():
            sys.exit(f"not found: {f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.design_csv)
    names = df["design_name"].tolist()
    n = len(names)
    if args.matrix:
        ident = pd.read_csv(args.matrix, index_col=0).reindex(index=names, columns=names).values
        if np.isnan(ident).any():
            sys.exit("the supplied matrix does not cover every design")
    else:
        ident = identity_matrix(df["full_sequence"].tolist())
        pd.DataFrame(ident, index=names, columns=names).to_csv(args.out_dir / "pairwise_identity.csv")

    scaf = np.array([accession(x) for x in names])
    iu = np.triu_indices(n, k=1)
    vals = ident[iu]
    same = scaf[iu[0]] == scaf[iu[1]]

    print(f"{n} designs, {len(vals):,} pairs, {len(set(scaf))} scaffolds; identity median "
          f"{np.median(vals):.1f}%, 95th {np.percentile(vals, 95):.1f}%, max {vals.max():.1f}%")
    for label, v in (("same scaffold", vals[same]), ("different scaffold", vals[~same])):
        if len(v):
            print(f"{label:<19} n={len(v):>6,}  median {np.median(v):.1f}%  max {v.max():.1f}%")

    Z = linkage(squareform(100.0 - ident, checks=False), method="average")
    sweep = []
    for t in range(25, 96, 5):
        labels = fcluster(Z, t=100.0 - t, criterion="distance")
        sweep.append({"threshold_pct": t, "n_clusters": len(set(labels)),
                      "ari_vs_scaffold": round(adjusted_rand_score(scaf, labels), 4),
                      "cross_scaffold_pairs_above": int(((vals >= t) & ~same).sum()),
                      "same_scaffold_pairs_below": int(((vals < t) & same).sum())})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(args.out_dir / "threshold_sweep.csv", index=False)
    plateau = sweep[sweep.ari_vs_scaffold >= 0.95]
    if len(plateau):
        print(f"ARI >= 0.95 from {plateau.threshold_pct.min()}% to {plateau.threshold_pct.max()}%, "
              f"peak {sweep.ari_vs_scaffold.max():.3f}")

    out = pd.DataFrame({"design_name": names, "scaffold": scaf})
    for t in (30, 40, 50, 60, 70):
        out[f"cluster_{t}pct"] = fcluster(Z, t=100.0 - t, criterion="distance")
    out.to_csv(args.out_dir / "sequence_clusters.csv", index=False)

    k = np.where(vals >= args.dup_threshold)[0]
    dups = pd.DataFrame({"design_a": [names[iu[0][i]] for i in k],
                         "design_b": [names[iu[1][i]] for i in k],
                         "identity_pct": vals[k].round(1), "same_scaffold": same[k]})
    dups.sort_values("identity_pct", ascending=False).to_csv(args.out_dir / "near_duplicates.csv", index=False)
    print(f"pairs >= {args.dup_threshold:g}% identity: {len(k)} "
          f"({int(same[k].sum())} within one scaffold)")

    if args.finalists:
        fin = pd.read_csv(args.finalists)["design_name"].tolist()
        if any(d not in names for d in fin):
            sys.exit("a finalist is not in the design table")
        idx = [names.index(d) for d in fin]
        sub = ident[np.ix_(idx, idx)]
        tags = [f"F{i:02d}" for i in range(1, len(fin) + 1)]
        pd.DataFrame(sub.round(1), index=tags, columns=tags).to_csv(args.out_dir / "finalist_identity.csv")
        fu = np.triu_indices(len(fin), 1)
        diff = np.array([accession(fin[i]) != accession(fin[j]) for i, j in zip(*fu)])
        print(f"finalists: median {np.median(sub[fu]):.1f}%, max {sub[fu].max():.1f}%"
              + (f"; different scaffolds max {sub[fu][diff].max():.1f}%" if diff.any() else ""))


if __name__ == "__main__":
    main()
