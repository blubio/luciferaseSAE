"""Analysis of the three 10.0 ns APO OpenMM trajectories in md_trajectories_10ns/.

IMPORTANT (audit 2026-11-13): these runs contain NO ligand. `openmm_nativegpu_cuda.ipynb`
filtered HETATM records at read time, so SLU was never in the simulated system. They are a
ligand-free backbone / gate-loop stability control ONLY and cannot support any steric-gate,
binding, or "rescue" claim. For holo runs see notebooks/holo_md_slu_parameterized.ipynb.

Two bugs in the previous version of this file were fixed:
  1. The DCD header seek was 88 bytes; the first record is 4 + 84 + 4 = 92 bytes, so every
     subsequent offset was wrong and the frame loop ran off the end of the file. The script
     had therefore never executed successfully, and the published fig11 came from an
     out-of-repo Colab variant whose RMSF values do not reproduce (2.88-4.47 A published vs
     0.48-0.52 A verified here).
  2. The gate loop was taken at residues 347-349 in *structure* numbering on all three
     topologies. "347" is a P. pyralis (P08659) alignment coordinate: the gate actually sits
     at 346-348 in A0A8K0DA02 and 343-345 in A0A8K0C7P5. It is now located by motif anchoring.

Produces results/figures/fig11_apo_md_10ns_backbone_stability.png
"""
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "md_trajectories_10ns")
FIG = os.path.join(ROOT, "results", "figures")

SYSTEMS = [
    ("1_A0A8K0DA02_TrueLuc_WT_S347_apo", "True luciferase (S347)", "#0072B2", 1),
    ("2_A0A8K0C7P5_Paralog_WT_L347_apo", "Non-luminous paralog (L347)", "#E69F00", 4),
    ("3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_apo",
     "SAE-steered mutant (S347/A348/N349)", "#CC79A7", 4),
]
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
       "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C"}


def read_dcd(path, idx_sets):
    """CHARMM/OpenMM DCD reader. idx_sets maps name -> array of 0-based atom indices."""
    with open(path, "rb") as f:
        ri = lambda: struct.unpack("<i", f.read(4))[0]
        assert ri() == 84
        hdr = f.read(84)
        assert hdr[:4] == b"CORD"
        icntrl = struct.unpack("<9i", hdr[4:40])
        delta = struct.unpack("<f", hdr[40:44])[0]
        has_cell = struct.unpack("<10i", hdr[44:84])[0]
        nset, nsavc, nstep = icntrl[0], icntrl[2], icntrl[3]
        assert ri() == 84
        ts = ri(); f.read(ts); assert ri() == ts
        assert ri() == 4; natom = ri(); assert ri() == 4

        out = {k: np.empty((nset, len(v), 3), np.float32) for k, v in idx_sets.items()}
        for fr in range(nset):
            if has_cell:
                n = ri(); f.read(n); assert ri() == n
            xyz = []
            for _ in range(3):
                n = ri(); assert n == 4 * natom
                xyz.append(np.frombuffer(f.read(n), "<f4")); assert ri() == n
            for k, v in idx_sets.items():
                out[k][fr] = np.stack([xyz[0][v], xyz[1][v], xyz[2][v]], 1)
    meta = dict(nset=nset, nsavc=nsavc, nstep=nstep, dt_fs=delta * 0.04888821 * 1000,
                ns=nstep * delta * 0.04888821 / 1000, natom=natom)
    return out, meta


def parse_pdb(path):
    """Backbone / CA indices, CA residue numbers, and the chain-A sequence."""
    bb, ca, res, seq = [], [], [], []
    i = 0
    for l in open(path):
        if l.startswith("ATOM") and l[21:22] == "A":
            name, rn3 = l[12:16].strip(), l[17:20].strip()
            try:
                rn = int(l[22:26])
            except ValueError:
                i += 1
                continue
            if name in ("N", "CA", "C"):
                bb.append(i)
            if name == "CA":
                ca.append(i); res.append(rn); seq.append(AA3.get(rn3, "X"))
        i += 1
    return np.array(bb), np.array(ca), np.array(res), "".join(seq)


def gate_triad(seq, res):
    """P08659-aligned positions 347/348/349, located by the unique GLTE anchor.

    start = index('GLTE') + 4 is the residue aligned to P08659 345, so the gate is
    start+2 .. start+4.  Verified unique in all three topologies.
    """
    hits = [i for i in range(len(seq) - 3) if seq[i:i + 4] == "GLTE"]
    assert len(hits) == 1, f"GLTE anchor not unique: {hits}"
    s = hits[0] + 6
    return [s, s + 1, s + 2], res[[s, s + 1, s + 2]], seq[s:s + 3]


def kabsch_rmsd(C):
    ref = C[0] - C[0].mean(0)
    out = [0.0]
    for i in range(1, len(C)):
        t = C[i] - C[i].mean(0)
        U, _, Vt = np.linalg.svd(t.T @ ref)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        out.append(float(np.sqrt((((t @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T) - ref) ** 2).sum(1).mean())))
    return np.array(out)


def kabsch_rmsf(C):
    ref = C[0] - C[0].mean(0)
    A = np.empty_like(C); A[0] = ref
    for i in range(1, len(C)):
        t = C[i] - C[i].mean(0)
        U, _, Vt = np.linalg.svd(t.T @ ref)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        A[i] = t @ (Vt.T @ np.diag([1, 1, d]) @ U.T).T
    return np.sqrt(((A - A.mean(0)) ** 2).sum(2).mean(0))


def main():
    print("=" * 78)
    print("3 x 10.0 ns APO all-atom GPU MD (OpenMM, AMBER99SB-ILDN/TIP3P, 4 fs HMR)")
    print("LIGAND-FREE: SLU absent. Backbone-stability control only.")
    print("=" * 78)

    S = {}
    for tag, label, colour, off in SYSTEMS:
        d = os.path.join(BASE, tag)
        bb, ca, res, seq = parse_pdb(os.path.join(d, "solvated.pdb"))
        co, meta = read_dcd(os.path.join(d, "prod.dcd"), {"bb": bb, "ca": ca})
        gi, gres, gaa = gate_triad(seq, res)
        rmsd, rmsf = kabsch_rmsd(co["bb"]), kabsch_rmsf(co["ca"])
        S[tag] = dict(label=label, colour=colour, off=off, res=res, rmsd=rmsd, rmsf=rmsf,
                      gate_rmsf=rmsf[gi].mean(), gres=gres, gaa=gaa)
        print(f"\n{label}")
        print(f"  {meta['nset']} frames | dt {meta['dt_fs']:.2f} fs | {meta['ns']:.2f} ns | {meta['natom']} atoms")
        print(f"  backbone RMSD 5-10 ns : {rmsd[250:].mean():.2f} +/- {rmsd[250:].std():.2f} A")
        print(f"  gate {gaa} at residues {list(gres)} (P08659 347-349) : CA RMSF {rmsf[gi].mean():.2f} A")

    t = np.arange(1, 501) * 0.02
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.6, 3.6))
    for tag, s in S.items():
        axA.plot(t, s["rmsd"], color=s["colour"], lw=.5, alpha=.35)
        axA.plot(t[9:], np.convolve(s["rmsd"], np.ones(10) / 10, "valid"),
                 color=s["colour"], lw=1.6, label=s["label"])
    axA.set_xlabel("Simulation time (ns)"); axA.set_ylabel("Backbone RMSD (Å)")
    axA.set_title("Ligand-free runs reach a stable fold")
    axA.set_xlim(0, 10); axA.margins(y=.08); axA.legend(frameon=False, loc="lower right")

    eq = [S[t_]["rmsd"][250:] for t_, *_ in SYSTEMS]
    bp = axB.boxplot(eq, widths=.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.4))
    rng = np.random.default_rng(0)
    for i, (box, (tag, *_)) in enumerate(zip(bp["boxes"], SYSTEMS)):
        box.set_facecolor(S[tag]["colour"]); box.set_alpha(.75)
        box.set_edgecolor("black"); box.set_linewidth(.8)
        axB.scatter(rng.normal(i + 1, .07, len(eq[i])), eq[i], s=1.4, color="black", alpha=.18, zorder=3)
    axB.set_xticks([1, 2, 3])
    axB.set_xticklabels(["True\nluciferase", "Non-luminous\nparalog", "SAE-steered\nmutant"])
    axB.set_ylabel("Backbone RMSD, 5–10 ns (Å)")
    delta = abs(eq[0].mean() - eq[1].mean())
    axB.set_title(f"Enzyme–decoy gap ({delta:.2f} Å) < within-run spread")
    axB.plot([1, 2], [2.31, 2.31], color="black", lw=.8)
    axB.annotate(f"Δ = {delta:.2f} Å", xy=(1.5, 2.34), ha="center", fontsize=6.5)
    axB.margins(y=.14)

    for tag, s in S.items():
        x = s["res"] + s["off"]
        m = (x >= 332) & (x <= 366)
        axC.plot(x[m], s["rmsf"][m], marker="o", ms=2.6, lw=1.3, color=s["colour"])
    axC.axvspan(346.5, 349.5, color="#cccccc", alpha=.55, zorder=0)
    axC.set_xlabel("Position (aligned to $\\it{P.\\ pyralis}$ P08659)")
    axC.set_ylabel("Cα RMSF (Å)")
    axC.set_title("At the gate, all three are equally rigid")
    axC.margins(y=.16)
    lo = min(S[t_]["gate_rmsf"] for t_, *_ in SYSTEMS)
    hi = max(S[t_]["gate_rmsf"] for t_, *_ in SYSTEMS)
    axC.text(348, axC.get_ylim()[1], f"gate 347–349\n{lo:.2f}–{hi:.2f} Å",
             ha="center", va="top", fontsize=6.5)

    for ax, L in zip((axA, axB, axC), "abc"):
        ax.text(-.16, 1.05, L, transform=ax.transAxes, fontweight="bold", fontsize=11, va="bottom")

    fig.tight_layout()
    out = os.path.join(FIG, "fig11_apo_md_10ns_backbone_stability.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
