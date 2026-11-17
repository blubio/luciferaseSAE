"""Parse the 12 holo-MD trajectories and write the summary table behind fig12.

Round-5 audit fix: fig12 was rendered in an interactive session and the per-system
numbers existed nowhere on disk, which is the provenance-drift defect this project
documents. This script regenerates them from prod.dcd + top.pdb.

Usage: python scripts/analyze_holo_md.py [holo_dir]
Writes: results/holo_md_summary.tsv
"""
import csv, glob, os, re, struct, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_MOTIF = "GLTE"          # unique in every topology; P08659 numbering anchor
GATE_OFFSET = 2              # start = idx(GLTE)+4 -> position 345; gate 347 = start+2
CONTACT_CUT = 4.0
CLASH_CUT = 2.8
AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V","HID":"H","HIE":"H","HIP":"H"}


def read_dcd(path):
    with open(path, "rb") as fh:
        blk = fh.read(4)
        hdr = fh.read(struct.unpack("<i", blk)[0])
        fh.read(4)
        nset = struct.unpack("<i", hdr[4:8])[0]
        has_cell = struct.unpack("<10i", hdr[44:84])[0]
        n = struct.unpack("<i", fh.read(4))[0]; fh.read(n); fh.read(4)          # title
        fh.read(4); natom = struct.unpack("<i", fh.read(4))[0]; fh.read(4)      # natom
        frames = []
        for _ in range(nset):
            if has_cell:
                sz = fh.read(4)
                if len(sz) < 4:
                    break
                fh.read(struct.unpack("<i", sz)[0]); fh.read(4)
            xyz = []
            for _ in range(3):
                sz = fh.read(4)
                if len(sz) < 4:
                    break
                xyz.append(np.frombuffer(fh.read(struct.unpack("<i", sz)[0]), dtype="<f4"))
                fh.read(4)
            if len(xyz) != 3:
                break
            frames.append(np.stack(xyz, axis=1))
    return np.array(frames), natom


def read_pdb(path):
    rec = []
    for l in open(path, encoding="utf-8", errors="ignore"):
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 78:
            rec.append((l[:6].strip(), l[12:16].strip(), l[17:20].strip(),
                        l[21], int(l[22:26]), l[76:78].strip() or l[12:16].strip()[0]))
    assert rec, f"no parsable atom records in {path}"
    return rec


def protein_sequence(rec):
    seen, seq, resids = set(), [], []
    for het, _, rn, ch, ri, _ in rec:
        if rn in AA3 and (ch, ri) not in seen:
            seen.add((ch, ri)); seq.append(AA3[rn]); resids.append((ch, ri))
    return "".join(seq), resids


def analyse(d):
    dcd, top = os.path.join(d, "prod.dcd"), os.path.join(d, "top.pdb")
    rec = read_pdb(top)
    traj, natom = read_dcd(dcd)
    assert traj.shape[1] == len(rec), f"{d}: {traj.shape[1]} coords vs {len(rec)} atoms"

    # NB: OpenMM writes the ligand as ATOM records here, not HETATM -- select by resname.
    lig = [i for i, r in enumerate(rec) if r[2] == "SLU" and r[5] != "H"]
    assert len(lig) == 40, f"{d}: {len(lig)} SLU heavy atoms, expected 40"

    seq, resids = protein_sequence(rec)
    k = seq.find(GATE_MOTIF)
    assert k >= 0 and seq.count(GATE_MOTIF) == 1, f"{d}: GATE motif not unique"
    gate_key = resids[k + 4 + GATE_OFFSET]
    gate_idx = [i for i, r in enumerate(rec) if (r[3], r[4]) == gate_key and r[5] != "H"]
    gate_resname = [r[2] for r in rec if (r[3], r[4]) == gate_key][0]
    prot = [i for i, r in enumerate(rec) if r[2] in AA3 and r[5] != "H"]

    # Ligand COM drift is measured AFTER removing protein rigid-body motion (Kabsch
    # superposition of each frame's protein heavy atoms onto frame 0); without this the
    # quantity is dominated by tumbling of the whole solute and is not interpretable.
    ref = traj[0][prot] - traj[0][prot].mean(0)
    dmin, com, contacts = [], [], []
    for fr in traj:
        P0 = fr[prot]; cP = P0.mean(0)
        U, _, Vt = np.linalg.svd((P0 - cP).T @ ref)
        dsign = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, dsign]) @ Vt
        al = lambda X: (X - cP) @ R
        g, L, P = al(fr[gate_idx]), al(fr[lig]), al(P0)
        dmin.append(np.sqrt(((g[:, None, :] - L[None]) ** 2).sum(-1)).min())
        com.append(L.mean(0))
        contacts.append(int((np.sqrt(((P[:, None, :] - L[None]) ** 2).sum(-1)) < CONTACT_CUT).any(1).sum()))
    dmin, com, contacts = np.array(dmin), np.array(com), np.array(contacts)
    return dict(gate_resname=gate_resname, gate_resid=gate_key[1], n_frames=len(traj),
                d_median=float(np.median(dmin)), d_min=float(dmin.min()),
                n_below_clash=int((dmin < CLASH_CUT).sum()),
                com_shift=float(np.linalg.norm(com[-1] - com[0])),
                com_max=float(np.linalg.norm(com - com[0], axis=1).max()),
                contacts_mean=float(contacts.mean()), contacts_min=int(contacts.min()))


def main(holo=None):
    holo = holo or os.path.join(ROOT, "holo_md_and_results")
    dirs = sorted(d for d in glob.glob(os.path.join(holo, "**", "*"), recursive=True)
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "prod.dcd")))
    rows = []
    for d in dirs:
        name = os.path.basename(d)
        r = analyse(d)
        r["system"] = re.sub(r"_rep\d+$", "", name)
        r["replicate"] = name[-1]
        rows.append(r)
        print(f"  {name:62s} {r['gate_resname']}{r['gate_resid']} "
              f"med {r['d_median']:.2f} min {r['d_min']:.2f} <{CLASH_CUT}A {r['n_below_clash']:3d} "
              f"COM {r['com_max']:.2f} cont {r['contacts_mean']:.0f}")
    cols = ["system", "replicate", "gate_resname", "gate_resid", "n_frames", "d_median",
            "d_min", "n_below_clash", "com_shift", "com_max", "contacts_mean", "contacts_min"]
    out = os.path.join(ROOT, "results", "holo_md_summary.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t"); wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})
    print(f"\n{len(rows)} trajectories -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
