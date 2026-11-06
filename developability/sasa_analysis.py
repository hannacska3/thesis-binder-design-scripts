#!/usr/bin/env python3
"""Per-residue solvent accessibility of binders (thesis Sections 4.2 and 4.3).

Shrake-Rupley accessibility is computed on each isolated binder and on its
complex. A residue is interfacial if it loses at least DSASA_INTERFACE A^2 on
complexation. The exposed hydrophobic surface is split into interface and
non-interface, and protease motifs are counted before and after requiring the
cleavage residue to be exposed. Requires biopython.
"""

import argparse
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
except ImportError:
    sys.exit("biopython missing: pip install biopython")

BINDER_CHAIN = "A"
PDB_GLOB = "*.pdb"

RSA_EXPOSED = 0.25
DSASA_INTERFACE = 25.0
HYDROPHOBIC = set("AVLIMFWY")

# name -> (residues cut after, following residue that blocks the cut)
MOTIFS = {
    "trypsin":          ("KR", "P"),
    "chymotrypsin_low": ("FYWML", "P"),
    "elastase":         ("AV", "P"),
    "proline_endopep":  ("P", "P"),
}

# Maximum accessibility per residue type (Tien et al. 2013, theoretical)
MAX_ACC = dict(A=129, R=274, N=195, D=193, C=167, E=223, Q=225, G=104,
               H=224, I=197, L=201, K=236, M=224, F=240, P=159, S=155,
               T=172, W=285, Y=263, V=174)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def normalise_key(name: str) -> str:
    key = str(name).strip()
    key = re.sub(r"\.(pdb|cif|ent)$", "", key, flags=re.IGNORECASE)
    for suffix in ("_binder", "_complex", "_combined", "_only_binder"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return re.sub(r"_[A-Z]$", "", key)


def residue_sasa(structure, chain_id=None):
    """{residue number: (one-letter code, SASA)} for the first model."""
    out = {}
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for res in chain:
                aa = THREE_TO_ONE.get(res.get_resname())
                if aa is None:
                    continue
                out[res.id[1]] = (aa, sum(a.sasa for a in res))
        break
    return out


def motif_positions(seq_by_pos):
    """{motif name: residue numbers whose C-terminal bond the enzyme cuts}."""
    pos_sorted = sorted(seq_by_pos)
    seq = "".join(seq_by_pos[p] for p in pos_sorted)
    hits = {}
    for name, (cut_after, blocked) in MOTIFS.items():
        s = set()
        for i, aa in enumerate(seq):
            if aa not in cut_after or i + 1 >= len(seq):
                continue
            if seq[i + 1] in blocked:
                continue
            s.add(pos_sorted[i])
        hits[name] = s
    return hits


def analyse_one(binder_pdb: Path, complex_pdb, parser, sr):
    free = parser.get_structure("free", str(binder_pdb))
    sr.compute(free, level="A")
    free_sasa = residue_sasa(free)
    if not free_sasa:
        return [], None

    bound_sasa = {}
    if complex_pdb is not None:
        comp = parser.get_structure("cplx", str(complex_pdb))
        sr.compute(comp, level="A")
        bound_sasa = residue_sasa(comp, chain_id=BINDER_CHAIN)

    motifs = motif_positions({p: aa for p, (aa, _) in free_sasa.items()})

    rows = []
    for p in sorted(free_sasa):
        aa, s_free = free_sasa[p]
        rsa_free = s_free / MAX_ACC.get(aa, 200)
        s_bound = bound_sasa.get(p, (aa, np.nan))[1] if bound_sasa else np.nan
        d = (s_free - s_bound) if bound_sasa and not np.isnan(s_bound) else np.nan
        row = {
            "pos": p, "aa": aa,
            "sasa_free": round(s_free, 2),
            "rsa_free": round(rsa_free, 4),
            "sasa_bound": round(s_bound, 2) if not np.isnan(s_bound) else np.nan,
            "dsasa": round(d, 2) if not np.isnan(d) else np.nan,
            "exposed": rsa_free >= RSA_EXPOSED,
            "interface": bool(d >= DSASA_INTERFACE) if not np.isnan(d) else False,
            "hydrophobic": aa in HYDROPHOBIC,
        }
        for name in MOTIFS:
            row[f"motif_{name}"] = p in motifs.get(name, set())
        rows.append(row)

    rdf = pd.DataFrame(rows)
    exposed, iface, phob = rdf.exposed, rdf.interface, rdf.hydrophobic

    d = {
        "n_residues": len(rdf),
        "n_exposed": int(exposed.sum()),
        "n_interface": int(iface.sum()),
        "exposed_hydrophobic_area": round(
            float(rdf.loc[exposed & phob, "sasa_free"].sum()), 1),
        "exposed_hydrophobic_area_interface": round(
            float(rdf.loc[exposed & phob & iface, "sasa_free"].sum()), 1),
        "exposed_hydrophobic_area_elsewhere": round(
            float(rdf.loc[exposed & phob & ~iface, "sasa_free"].sum()), 1),
    }
    for name in MOTIFS:
        col = f"motif_{name}"
        d[f"{name}_motifs"] = int(rdf[col].sum())
        d[f"{name}_exposed"] = int((rdf[col] & exposed).sum())
    return rows, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("binder_dir", type=Path, help="folder of isolated binder PDBs")
    ap.add_argument("complex_dir", type=Path,
                    help=f"folder of complex PDBs (binder is chain {BINDER_CHAIN})")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("sasa"),
                    help="output folder (default: sasa)")
    args = ap.parse_args()

    for folder in (args.binder_dir, args.complex_dir):
        if not folder.is_dir():
            sys.exit(f"not a folder: {folder}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    binders = sorted(args.binder_dir.glob(PDB_GLOB))
    if not binders:
        sys.exit(f"no {PDB_GLOB} in {args.binder_dir}")
    complexes = {normalise_key(p.name): p for p in args.complex_dir.glob(PDB_GLOB)}

    parser = PDBParser(QUIET=True)
    sr = ShrakeRupley()  # probe radius 1.40 A, 100 points per atom

    all_res, per_design, n_cplx = [], [], 0
    for pdb in binders:
        key = normalise_key(pdb.name)
        cplx = complexes.get(key)
        rows, d = analyse_one(pdb, cplx, parser, sr)
        if d is None:
            continue
        n_cplx += cplx is not None
        for r in rows:
            r["design"] = key
        all_res.extend(rows)
        per_design.append({"design": key, **d})

    rdf = pd.DataFrame(all_res)
    ddf = pd.DataFrame(per_design)
    rdf.to_csv(args.out_dir / "per_residue.csv", index=False)
    ddf.to_csv(args.out_dir / "per_design.csv", index=False)

    tot = ddf.exposed_hydrophobic_area
    ifa = ddf.exposed_hydrophobic_area_interface
    els = ddf.exposed_hydrophobic_area_elsewhere
    print(f"designs {len(ddf)} ({n_cplx} with complex), residues {len(rdf)}, "
          f"exposed {100 * rdf.exposed.mean():.1f}%, "
          f"interface {100 * rdf.interface.mean():.1f}% "
          f"(median {ddf.n_interface.median():.0f} per design)")
    print(f"exposed hydrophobic area, median per design (A^2): "
          f"interface {ifa.median():.0f}, elsewhere {els.median():.0f}")
    print(f"share of pool total: interface {100 * ifa.sum() / tot.sum():.1f}%, "
          f"elsewhere {100 * els.sum() / tot.sum():.1f}%")
    print(f"{'enzyme':<18}{'motifs':>7}{'exposed':>8}{'surviving':>10}")
    for name in MOTIFS:
        m, e = ddf[f"{name}_motifs"], ddf[f"{name}_exposed"]
        frac = 100 * e.sum() / m.sum() if m.sum() else 0
        print(f"{name:<18}{m.median():>7.0f}{e.median():>8.0f}{frac:>9.1f}%")


if __name__ == "__main__":
    main()