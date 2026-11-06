#!/usr/bin/env python3
"""Contact molecular surface of docked complexes (thesis Section 3.2, CMS >= 100).

For every PDB in --complex-dir, PyRosetta's ContactMolecularSurfaceFilter scores the
whole receptor-binder interface (cms_total) and the contacts between the binder and
the hotspot residues listed one per line in --hotspots (cms_hotspots). Resumable:
complexes already in --output are skipped.
"""

import argparse
import csv
import multiprocessing as mp
import os

import pyrosetta

pyrosetta.init("-beta_nov16 -mute all")
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.select.residue_selector import (
    AndResidueSelector, ChainSelector, ResidueIndexSelector)
from pyrosetta.rosetta.protocols.simple_filters import ContactMolecularSurfaceFilter


def cms(pose, sel1, sel2):
    f = ContactMolecularSurfaceFilter()
    f.selector1(sel1)
    f.selector2(sel2)
    f.distance_weight(0.5)
    return f.report_sm(pose)


def worker(job):
    pdb_file, receptor_chain, binder_chain, hotspots = job
    row = {"complex": os.path.basename(pdb_file), "cms_total": "", "cms_hotspots": "", "error": ""}
    try:
        pose = pose_from_pdb(pdb_file)
        receptor, binder = ChainSelector(receptor_chain), ChainSelector(binder_chain)
        receptor_hotspots = AndResidueSelector()
        receptor_hotspots.add_residue_selector(receptor)
        receptor_hotspots.add_residue_selector(ResidueIndexSelector(hotspots))
        row["cms_total"] = cms(pose, receptor, binder)
        row["cms_hotspots"] = cms(pose, receptor_hotspots, binder)
    except Exception as e:
        row["error"] = str(e)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--complex-dir", required=True)
    ap.add_argument("--hotspots", required=True, help="text file, one residue selector per line")
    ap.add_argument("--output", required=True, help="output CSV")
    ap.add_argument("--receptor-chain", default="A")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--ncpus", type=int, default=16)
    args = ap.parse_args()

    hotspots = ",".join(l.strip() for l in open(args.hotspots) if l.strip())
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as fh:
            done = {r["complex"] for r in csv.DictReader(fh)}
    jobs = [(os.path.join(args.complex_dir, f), args.receptor_chain, args.binder_chain, hotspots)
            for f in sorted(os.listdir(args.complex_dir)) if f.endswith(".pdb") and f not in done]
    print(f"{len(jobs)} complexes to score")

    new = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["complex", "cms_total", "cms_hotspots", "error"])
        if new:
            writer.writeheader()
        with mp.Pool(args.ncpus) as pool:
            for result in pool.imap_unordered(worker, jobs):
                writer.writerow(result)
                out.flush()


if __name__ == "__main__":
    main()
