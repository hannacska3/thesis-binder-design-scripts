#!/usr/bin/env python3
"""PyRosetta interface analysis of relaxed binder-target complexes (thesis Section 3.5).

Expects <input_dir>/<design>/<design>_0001.pdb with the binder on chain A and the
target on chain B. Each design runs in its own subprocess, so a Rosetta crash cannot
stop the batch. Results go to <input_dir>/interface_analysis.csv (resumable) and
failures to <input_dir>/interface_analysis_failed.txt.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

MARKER = "RESULT_JSON:"
THIS_SCRIPT = os.path.abspath(__file__)

FIELDS = ["design", "Clashes", "Binder_Energy_Score", "Surface_Hydrophobicity",
          "ShapeComplementarity", "PackStat", "dG", "dSASA", "dG/dSASA",
          "Interface_SASA_%", "Interface_Hydrophobicity", "n_InterfaceResidues",
          "n_InterfaceHbonds", "InterfaceHbondsPercentage", "n_InterfaceUnsatHbonds",
          "InterfaceUnsatHbondsPercentage", "Interface_Helix%", "Interface_BetaSheet%",
          "Interface_Loop%", "Binder_Helix%", "Binder_BetaSheet%", "Binder_Loop%",
          "InterfaceAAs"]

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225,
           "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
           "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
           "TYR": 263, "VAL": 174}
KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
      "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
      "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9,
      "TYR": -1.3, "VAL": 4.2}
HYDROPHOBIC = {aa for aa, v in KD.items() if v > 0}


def chains_of(pose):
    info = pose.pdb_info()
    return {info.chain(i) for i in range(1, pose.total_residue() + 1)}


def keep_chains(pose, keep):
    from pyrosetta import pose_from_pdb
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        path = tmp.name
    try:
        pose.dump_pdb(path)
        lines = [l for l in open(path)
                 if not l.startswith(("ATOM", "HETATM")) or l[21] in keep]
        open(path, "w").writelines(lines)
        return pose_from_pdb(path)
    finally:
        os.unlink(path)


def count_clashes(pose, chain_a, chain_b, tol, ca_cut=12.0):
    info = pose.pdb_info()
    res = range(1, pose.total_residue() + 1)
    a_res = [i for i in res if info.chain(i) == chain_a]
    b_res = [i for i in res if info.chain(i) == chain_b]

    def ca(i):
        r = pose.residue(i)
        return r.xyz("CA") if r.has("CA") else r.nbr_atom_xyz()

    b_ca = {i: ca(i) for i in b_res}
    clashes = 0
    for ia in a_res:
        near = [ib for ib, xyz in b_ca.items() if ca(ia).distance(xyz) <= ca_cut]
        ra = pose.residue(ia)
        for ib in near:
            rb = pose.residue(ib)
            for ai in range(1, ra.natoms() + 1):
                ea = ra.atom_type(ai).element()
                if ea == "H":
                    continue
                for bi in range(1, rb.natoms() + 1):
                    eb = rb.atom_type(bi).element()
                    if eb != "H" and ra.xyz(ai).distance(rb.xyz(bi)) < (
                            VDW.get(ea, 1.70) + VDW.get(eb, 1.70)) * tol:
                        clashes += 1
    return clashes


def ss_fractions(ss, idx):
    if not idx:
        return None, None, None
    h = sum(ss[i - 1] == "H" for i in idx)
    e = sum(ss[i - 1] == "E" for i in idx)
    n = len(idx)
    return 100 * h / n, 100 * e / n, 100 * (n - h - e) / n


def analyze(pose_ab, binder, target):
    import pyrosetta
    from pyrosetta.rosetta.core.scoring.dssp import Dssp
    from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set
    from pyrosetta.rosetta.core.scoring.sasa import SasaCalc
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector, InterGroupInterfaceByVectorSelector)
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.simple_filters import ShapeComplementarityFilter

    info = pose_ab.pdb_info()
    pose_b, pose_t = keep_chains(pose_ab, {binder}), keep_chains(pose_ab, {target})
    sasa_complex = SasaCalc().calculate(pose_ab)
    calc_b = SasaCalc()
    sasa_b = calc_b.calculate(pose_b)
    res_sasa = calc_b.get_residue_sasa()
    dsasa = sasa_b + SasaCalc().calculate(pose_t) - sasa_complex

    iam = InterfaceAnalyzerMover(1)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_delta_hbond_unsat(True)
    iam.set_compute_interface_energy(True)
    iam.apply(pose_ab)
    dg = iam.get_separated_interface_energy()
    unsat = iam.get_interface_delta_hbond_unsat()

    sc = ShapeComplementarityFilter()
    sc.jump_id(1)

    mask = InterGroupInterfaceByVectorSelector(ChainSelector(binder), ChainSelector(target)).apply(pose_ab)
    n = pose_ab.total_residue()
    iface = [i for i in range(1, n + 1) if mask[i] and info.chain(i) == binder]
    n_if = len(iface)

    hbs = HBondSet()
    fill_hbond_set(pose_ab, False, hbs)
    n_hb = sum(info.chain(hbs.hbond(k).don_res()) != info.chain(hbs.hbond(k).acc_res())
               for k in range(1, hbs.nhbonds() + 1))

    Dssp(pose_ab).insert_ss_into_pose(pose_ab)
    ss = pose_ab.secstruct()
    b_all = [i for i in range(1, n + 1) if info.chain(i) == binder]
    bh, be, bl = ss_fractions(ss, b_all)
    ih, ie, il = ss_fractions(ss, iface)

    surface = [i for i in range(1, pose_b.total_residue() + 1)
               if pose_b.residue(i).name3() in MAX_ASA
               and res_sasa[i] / MAX_ASA[pose_b.residue(i).name3()] >= 0.25]
    surf_hyd = (sum(pose_b.residue(i).name3() in HYDROPHOBIC for i in surface) / len(surface)
                if surface else None)
    iface_hyd = (sum(pose_ab.residue(i).name3() in HYDROPHOBIC for i in iface) / n_if
                 if n_if else None)

    return {
        "Binder_Energy_Score": pyrosetta.create_score_function("ref2015")(pose_b),
        "Surface_Hydrophobicity": surf_hyd, "ShapeComplementarity": sc.score(pose_ab),
        "PackStat": iam.get_interface_packstat(), "dG": dg, "dSASA": dsasa,
        "dG/dSASA": dg / dsasa if dsasa else None,
        "Interface_SASA_%": (dsasa / 2) / sasa_b * 100 if sasa_b else None,
        "Interface_Hydrophobicity": iface_hyd, "n_InterfaceResidues": n_if,
        "n_InterfaceHbonds": n_hb,
        "InterfaceHbondsPercentage": n_hb / n_if * 100 if n_if else None,
        "n_InterfaceUnsatHbonds": unsat,
        "InterfaceUnsatHbondsPercentage": unsat / n_if * 100 if n_if else None,
        "Interface_Helix%": ih, "Interface_BetaSheet%": ie, "Interface_Loop%": il,
        "Binder_Helix%": bh, "Binder_BetaSheet%": be, "Binder_Loop%": bl,
        "InterfaceAAs": json.dumps(dict(Counter(pose_ab.residue(i).name3() for i in iface))),
    }


def run_worker(args):
    import pyrosetta
    from pyrosetta import pose_from_pdb
    pyrosetta.init("-ex1 -ex2 -use_input_sc -no_optH false -flip_HNQ "
                   "-ignore_unrecognized_res true -ignore_zero_occupancy false", silent=True)
    pose = pose_from_pdb(args.worker)
    need = {args.binder_chain, args.target_chain}
    present = chains_of(pose)
    if need - present:
        print("\n" + MARKER + json.dumps({"design": args.design_name, "status": "missing_chains"}))
        return
    if present - need:
        pose = keep_chains(pose, need)
    data = {"design": args.design_name, "status": "ok",
            "Clashes": count_clashes(pose, args.binder_chain, args.target_chain, args.clash_tolerance)}
    data.update(analyze(pose, args.binder_chain, args.target_chain))
    print("\n" + MARKER + json.dumps(data), flush=True)


def run_one(pdb, name, args):
    cmd = [sys.executable, THIS_SCRIPT, "--worker", pdb, "--design-name", name,
           "--binder-chain", args.binder_chain, "--target-chain", args.target_chain,
           "--clash-tolerance", str(args.clash_tolerance)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    idx = proc.stdout.rfind(MARKER)
    if idx != -1:
        try:
            return name, json.loads(proc.stdout[idx + len(MARKER):].strip().splitlines()[0]), None
        except (json.JSONDecodeError, IndexError):
            pass
    err = proc.stderr.strip().splitlines()
    return name, None, f"exit={proc.returncode}; {err[-1] if err else 'no stderr'}"


def run_driver(args):
    root = os.path.abspath(args.input_dir)
    pairs = [(d, os.path.join(root, d, f"{d}_0001.pdb")) for d in sorted(os.listdir(root))
             if os.path.isfile(os.path.join(root, d, f"{d}_0001.pdb"))]
    out_csv = os.path.join(root, "interface_analysis.csv")
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as fh:
            done = {r["design"] for r in csv.DictReader(fh)}
    todo = [p for p in pairs if p[0] not in done]
    print(f"{len(pairs)} relaxed designs, {len(todo)} to analyze")

    failures, ok = [], 0
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, pdb, name, args) for name, pdb in todo]
            for fut in as_completed(futures):
                name, data, err = fut.result()
                if data and data.get("status") == "ok":
                    writer.writerow({k: data.get(k) for k in FIELDS})
                    fh.flush()
                    ok += 1
                else:
                    failures.append((name, err or (data or {}).get("status")))
    if failures:
        with open(os.path.join(root, "interface_analysis_failed.txt"), "a") as fh:
            fh.writelines(f"{n}\t{r}\n" for n, r in failures)
    print(f"{ok} analyzed, {len(failures)} failed or skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input_dir", nargs="?", help="folder of <design>/<design>_0001.pdb")
    ap.add_argument("--jobs", "-j", type=int, default=20)
    ap.add_argument("--binder-chain", default="A")
    ap.add_argument("--target-chain", default="B")
    ap.add_argument("--clash-tolerance", type=float, default=0.75)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--design-name", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        try:
            run_worker(args)
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    elif args.input_dir:
        run_driver(args)
    else:
        ap.error("input_dir is required")


if __name__ == "__main__":
    main()
