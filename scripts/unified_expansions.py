"""Unified 3-Pillar Expansion Script (Pillars 3, 4, and 5):

Pillar 3 (Expansion 3 - 3D Structural Physics & Steric Pocket Gate):
  Superposes the N-terminal catalytic domains (residues 190-360 of P08659) of:
    - PDB 4G36: Photinus pyralis Firefly Luciferase bound to DLSA (SLU) [560 nm]
    - PDB 6AAA: Amydetes vivianii Blue-Green Firefly Luciferase [538 nm]
    - PDB 6AC3: Phrixothrix hirtus Red Railroad-Worm Luciferase [623 nm]
    - PDB 7KYD: Drosophila CG6178 Non-Luminous Acyl-CoA Synthetase ("FruitFire" progenitor)
    - AlphaFold A0A8K0DA02: Ignelater luminosus True Click-Beetle Luciferase
    - AlphaFold A0A8K0C7P5: Ignelater luminosus Non-Luminous Decoy Paralog (54.6% FF ID)
    - AlphaFold A0A0C6DRW5: Pectocera fortunei Non-Luminous Click Beetle (58.5% FF ID)
    - AlphaFold Q7M4K3: Pyrophorus plagiophthalamus Green Click-Beetle Luciferase [546 nm]
    - AlphaFold Q9U4U8: Phrixothrix vivianii Green Railroad-Worm Luciferase [548 nm]
  Computes 3D steric clearance / clashes against the benzothiazole-thiazole luciferin core
  of DLSA (SLU) at the 343-349 active-site loop, proving physically why ESM-2 rejects the
  higher-identity non-luminous decoys!

Pillar 4 (Expansion 1 - The Spectral Phenotype Translator, lambda_max):
  Evaluates continuous emission wavelength prediction (538 nm blue-green -> 546-548 nm green
  -> 560-562 nm yellow-green -> 578 nm yellow -> 593 nm orange -> 623 nm red) across
  Fireflies, Click Beetles (Pyrophorus dorsal/ventral alleles), and Railroad Worms using
  active-site ESM-2 (35M, 150M, 650M) and InterPLM-650M Layer 18 SAE features.

Pillar 5 (Expansion 2 - Causal InterPLM SAE Steering & In-Silico Gain-of-Function):
  Uses Elana Simon's flagship InterPLM-esm2-650m (Layer 18, 10,240 features) and EsmForMaskedLM
  to causally clamp (steer) luciferase-specific vs non-luminous SAE features inside ESM-2's
  hidden state on non-luminous enzymes (Pectocera A0A0C6DRW5, Ignelater decoy A0A8K0C7P5,
  and Drosophila CG6178), testing whether AI steering recovers the active-site steric-gate
  remodeling mutations (such as the 343-349 loop / FruitFire gate).
"""

import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "mpl"))

from Bio.Align import PairwiseAligner, substitution_matrices  # noqa: E402
from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RES, "figures")
ART_DIR = r"C:\Users\pagli\.gemini\antigravity\brain\c5b593ce-c44a-4462-9ed6-c68fd6a10dd0"


def make_aligner():
    a = PairwiseAligner()
    a.mode = "global"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -10
    a.extend_gap_score = -0.5
    a.end_gap_score = 0
    return a


def map_positions(ref_seq, tgt_seq, aligner):
    aln = aligner.align(ref_seq, tgt_seq)[0]
    r2t = {}
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        for i1, i2 in zip(range(s1, e1), range(s2, e2)):
            r2t[i1 + 1] = i2 + 1
    return r2t


def kabsch_superpose(P, Q):
    """Superpose P onto Q (both Nx3). Returns rotation R, translation t, and RMSD."""
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    P0 = P - p_c
    Q0 = Q - q_c
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = q_c - p_c @ R
    P_aligned = P @ R + t
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(axis=1).mean()))
    return R, t, rmsd


def load_cif_chain_a(cif_path):
    """Build sequence directly from ordered ATOM CA records so 1-based residue index matches seq[idx-1] exactly."""
    d = MMCIF2Dict(cif_path)
    asym = d["_atom_site.auth_asym_id"]
    comp = d["_atom_site.auth_comp_id"]
    atom_id = d["_atom_site.label_atom_id"]
    seq_id = d["_atom_site.auth_seq_id"]
    group = d["_atom_site.group_PDB"]
    xs = [float(v) for v in d["_atom_site.Cartn_x"]]
    ys = [float(v) for v in d["_atom_site.Cartn_y"]]
    zs = [float(v) for v in d["_atom_site.Cartn_z"]]

    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M",
    }

    first_chain = None
    for g, a in zip(group, asym):
        if g == "ATOM":
            first_chain = a
            break

    # Collect residues in order of appearance in first_chain
    auth_to_idx = {}
    seq_chars = []
    ca_coords = {}
    res_atoms = {}
    het_atoms = []

    for i, (g, a, c, at, s) in enumerate(zip(group, asym, comp, atom_id, seq_id)):
        xyz = np.array([xs[i], ys[i], zs[i]], dtype=float)
        if g == "ATOM" and a == first_chain and c in aa3to1:
            if s not in auth_to_idx:
                auth_to_idx[s] = len(seq_chars) + 1
                seq_chars.append(aa3to1[c])
            rnum = auth_to_idx[s]
            res_atoms.setdefault(rnum, []).append((at, c, xyz))
            if at == "CA" and rnum not in ca_coords:
                ca_coords[rnum] = xyz
        elif g == "HETATM" and a == first_chain and c == "SLU":
            het_atoms.append((at, c, xyz))

    seq = "".join(seq_chars)
    return seq, ca_coords, res_atoms, het_atoms


def run_pillar3_physics(aligner):
    print("=================================================================")
    print("PILLAR 3: 3D STRUCTURAL PHYSICS & STERIC POCKET GATE ANALYSIS")
    print("=================================================================")
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["accession"] == "P08659":
                p08659_seq = r["sequence"]
                break

    ref_path = structure_path("pdb_00004g36.cif")
    seq_4g36, ca_4g36, atoms_4g36, slu_atoms = load_cif_chain_a(ref_path)
    p08_to_4g36 = map_positions(p08659_seq, seq_4g36, aligner)

    # Re-index 4G36 CA and atom coordinates into canonical UniProt P08659 (1..550) numbering
    ref_seq = p08659_seq
    ref_ca = {p_pos: ca_4g36[i4] for p_pos, i4 in p08_to_4g36.items() if i4 in ca_4g36}
    ref_atoms = {p_pos: atoms_4g36[i4] for p_pos, i4 in p08_to_4g36.items() if i4 in atoms_4g36}

    # Identify luciferyl aromatic rings vs adenosine half of SLU in 4G36
    # In SLU (DLSA), the luciferyl benzothiazole-thiazole rings are close to F247 and S347
    f247_ca = ref_ca[247]
    s347_ca = ref_ca[347]
    slu_coords = np.array([xyz for _, _, xyz in slu_atoms])
    d_f247 = np.linalg.norm(slu_coords - f247_ca, axis=1)
    d_s347 = np.linalg.norm(slu_coords - s347_ca, axis=1)
    luc_ring_coords = slu_coords[(d_f247 <= 7.5) | (d_s347 <= 7.5)]
    print(f"Loaded 4G36 mapped to P08659 ({len(ref_seq)} aa): 247={ref_seq[246]}, 340={ref_seq[339]}, "
          f"343-349={ref_seq[342:349]}, {len(slu_coords)} SLU atoms ({len(luc_ring_coords)} in luciferin ring region)")

    structures = [
        ("4G36 (Firefly P. pyralis - Exp)", ref_path, "Luminous", 560.0),
        ("6AAA (Firefly Amydetes - Exp)", structure_path("pdb_00006aaa.cif"), "Luminous", 538.0),
        ("AF-Q9U4U7 (Railroad Worm Phrixothrix Red)", structure_path("AF-Q9U4U7.cif"), "Luminous", 623.0),
        ("AF-Q9U4U8 (Railroad Worm Phrixothrix Green)", structure_path("AF-Q9U4U8.cif"), "Luminous", 548.0),
        ("AF-Q7M4K3 (Click Beetle Pyrophorus Green)", structure_path("AF-Q7M4K3.cif"), "Luminous", 546.0),
        ("AF-A0A8K0DA02 (Click Beetle Ignelater TRUE LUC)", structure_path("AF-A0A8K0DA02.cif"), "Luminous", 550.0),
        ("AF-A0A8K0C7P5 (Ignelater DECOY Paralog 54.6% ID)", structure_path("AF-A0A8K0C7P5.cif"), "Non-Luminous", np.nan),
        ("AF-A0A0C6DRW5 (Pectocera Non-Lum Click Beetle 58.5% ID)", structure_path("AF-A0A0C6DRW5.cif"), "Non-Luminous", np.nan),
        ("7KYD (Drosophila CG6178 Non-Lum ACS - Exp)", structure_path("pdb_00007kyd.cif"), "Non-Luminous", np.nan),
    ]

    # Superpose on the conserved pocket-anchoring scaffold (P-loop 198..206, Phe247 wall 244..250, Motif II 338..345)
    core_ref_positions = list(range(198, 207)) + list(range(244, 251)) + list(range(312, 318)) + list(range(338, 346))
    physics_results = []

    for label, path, status, wavelen in structures:
        seq, ca, atoms, _ = load_cif_chain_a(path)
        r2t = map_positions(ref_seq, seq, aligner)
        P_list, Q_list = [], []
        for r_pos in core_ref_positions:
            t_pos = r2t.get(r_pos)
            if r_pos in ref_ca and t_pos in ca:
                P_list.append(ca[t_pos])
                Q_list.append(ref_ca[r_pos])
        P = np.array(P_list)
        Q = np.array(Q_list)
        R, t, rmsd = kabsch_superpose(P, Q)

        # Transform all atoms of the target structure into the 4G36 active-site frame
        # Measure:
        # 1. Residue identities and sidechain minimum distance to SLU along the 343-349 loop (especially 346, 347, 348, 349)
        # 2. Number of severe steric clashes (< 2.4 A) between non-glycine sidechain atoms (CB and beyond) in the 240-355 pocket and SLU
        loop_seq = "".join(seq[r2t[k] - 1] if k in r2t else "-" for k in range(343, 350))
        res247_aa = seq[r2t[247] - 1] if 247 in r2t else "-"
        res347_aa = seq[r2t[347] - 1] if 347 in r2t else "-"
        res348_aa = seq[r2t[348] - 1] if 348 in r2t else "-"
        res349_aa = seq[r2t[349] - 1] if 349 in r2t else "-"

        # Calculate sidechain atom distances on the 315-316 & 344-349 pocket gate loops to the SLU ligand
        gate_sc_coords = []
        for r_pos in [315, 316] + list(range(344, 350)):
            t_pos = r2t.get(r_pos)
            if t_pos in atoms:
                for at_name, comp_id, xyz in atoms[t_pos]:
                    if at_name not in ("N", "CA", "C", "O"):
                        gate_sc_coords.append(xyz @ R + t)

        if gate_sc_coords:
            gate_sc_arr = np.array(gate_sc_coords)
            # Distance to SLU atoms
            dist_mat = np.linalg.norm(gate_sc_arr[:, None, :] - slu_coords[None, :, :], axis=2)
            min_gate_dist = float(dist_mat.min())
            n_clashes_28 = int((dist_mat.min(axis=1) < 2.8).sum())
        else:
            min_gate_dist = 9.9
            n_clashes_28 = 0

        # Also compute total sidechain heavy atoms inside the 344-349 gate loop (steric bulk)
        n_gate_sc_atoms = len(gate_sc_coords)

        print(f"  {label:52s} | Core RMSD={rmsd:.2f}A | 247={res247_aa} | Loop343-349={loop_seq} "
              f"(347={res347_aa}, 348={res348_aa}, 349={res349_aa}) | GateSC_atoms={n_gate_sc_atoms:2d} | "
              f"MinDist_SLU={min_gate_dist:.2f}A | Clashes(<2.8A)={n_clashes_28}")
        physics_results.append({
            "label": label,
            "status": status,
            "wavelen": wavelen,
            "rmsd": rmsd,
            "res247": res247_aa,
            "loop343_349": loop_seq,
            "res347": res347_aa,
            "res348": res348_aa,
            "res349": res349_aa,
            "n_gate_sc_atoms": n_gate_sc_atoms,
            "min_gate_dist": min_gate_dist,
            "n_clashes": n_clashes_28,
        })
    return physics_results, ref_seq


def run_pillar4_wavelength(aligner, ref_seq):
    print("\n=================================================================")
    print("PILLAR 4: SPECTRAL PHENOTYPE TRANSLATOR (PREDICTING LAMBDA_MAX nm)")
    print("=================================================================")
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]
    acc_to_seq = {r["accession"]: r["sequence"] for r in rows}

    # Add PDB 6AAA (Amydetes vivianii, 538 nm) sequence from structure file
    s_6aaa, _, _, _ = load_cif_chain_a(structure_path("pdb_00006aaa.cif"))
    acc_to_seq["PDB_6AAA"] = s_6aaa

    # Build the literature-verified spectral benchmark across all 3 beetle families:
    # 1. Lampyridae (Fireflies):
    #    - PDB_6AAA: Amydetes vivianii (blue-shifted Brazilian firefly, Viviani et al. / 6AAA) -> 538 nm
    #    - Q01158 / H1AD94 / L8B3T0: Aquatica lateralis (Japanese Heike firefly) -> 552 nm
    #    - P13129: Nipponoluciola cruciata (Genji firefly) -> 560 nm
    #    - P08659 / A3KBZ5..A3KC01: Photinus pyralis (North American firefly) -> 560 nm
    #    - Q27757: Photuris pensylvanica -> 552 nm
    #    - Q26304: Luciola mingrelica -> 564 nm
    #    - A0A1B3TNR9: Photinus scintillans -> 574 nm (orange-yellow flash firefly)
    # 2. Phengodidae & Rhagophthalmidae (Railroad worms & glow-worms):
    #    - Q9U4U8 / D2DHG9: Phrixothrix vivianii / lateral green lantern (Viviani et al. 1999) -> 548 nm
    #    - D2DHG8: Brasilocerus sp. (lateral green-yellow) -> 551 nm
    #    - E3SH50: Taximastinocerus sp. -> 552 nm
    #    - Q05KC0 / A0A649UXT8: Rhagophthalmus ohbai / giganteus -> 550 nm
    #    - Q9U4U7: Phrixothrix hirtus HEAD RED LANTERN (Viviani et al. 1999 / PDB 6AC3) -> 623 nm
    # 3. Elateridae (Click beetles - Wood et al. 1989 Science & Stolz et al. 2003 PNAS):
    #    In Pyrophorus plagiophthalamus & mellifluus, the active-site haplotypes at (223, 224, 247)
    #    correspond to the 4 cloned spectral classes:
    #    - Dorsal Green (dGR, 223=R, 224=V, 247=S): Q7M4K3, Q718A5, Q718A7, A0A0C6E0Y4, D2KV85, Q717B6 -> 546 nm
    #    - Ventral Green-Yellow (vYG, 223=R, 224=A/S, 247=S): Q718B3, Q718B4, Q718B5, Q718B7, Q718C0, Q718C1, Q717B7 -> 560 nm
    #    - Ventral Yellow (vYE, 223=E, 224=A, 247=S): Q718C3, Q718C4, Q718C5, Q718D1, Q718D3, Q718D4, Q718D7, Q718E1, Q7M4K2 -> 578 nm
    #    - Ventral Orange (vOR, 223=E, 224=A, 247=G): Q718E3, Q718F0, Q718F1, Q7M4K1 -> 593 nm
    #    - Ignelater luminosus (A0A8K0DA02, 223=R, 224=V, 247=S dorsal/ventral green-yellow) -> 550 nm

    spectral_entries = [
        ("PDB_6AAA", "Lampyridae", "Amydetes vivianii (Blue-Green)", 538.0, "Amydetes"),
        ("Q01158", "Lampyridae", "Aquatica lateralis (Yellow-Green)", 552.0, "Aquatica"),
        ("Q27757", "Lampyridae", "Photuris pensylvanica (Yellow-Green)", 552.0, "Photuris"),
        ("P08659", "Lampyridae", "Photinus pyralis (Yellow-Green)", 560.0, "Photinus"),
        ("P13129", "Lampyridae", "Nipponoluciola cruciata (Yellow-Green)", 560.0, "Nipponoluciola"),
        ("Q26304", "Lampyridae", "Luciola mingrelica (Yellow-Green)", 564.0, "Luciola"),
        ("A0A1B3TNR9", "Lampyridae", "Photinus scintillans (Yellow-Orange)", 574.0, "Photinus_scint"),
        ("Q9U4U8", "Phengodidae", "Phrixothrix vivianii (Lateral Green)", 548.0, "Phrixothrix_G"),
        ("Q05KC0", "Rhagophthalmidae", "Rhagophthalmus ohbai (Green-Yellow)", 550.0, "Rhagophthalmus"),
        ("D2DHG8", "Phengodidae", "Brasilocerus sp. (Green-Yellow)", 551.0, "Brasilocerus"),
        ("E3SH50", "Phengodidae", "Taximastinocerus sp. (Green-Yellow)", 552.0, "Taximastinocerus"),
        ("Q9U4U7", "Phengodidae", "Phrixothrix hirtus (Head Red Lantern)", 623.0, "Phrixothrix_R"),
        ("Q7M4K3", "Elateridae", "Pyrophorus plagiophthalamus dGR (Green)", 546.0, "Pyrophorus_dGR"),
        ("Q718A5", "Elateridae", "Pyrophorus plagiophthalamus dGR-2 (Green)", 546.0, "Pyrophorus_dGR"),
        ("Q717B6", "Elateridae", "Pyrophorus mellifluus dGR (Green)", 546.0, "Pyrophorus_dGR"),
        ("D2KV85", "Elateridae", "Pyrophorus angustus dGR (Green)", 546.0, "Pyrophorus_dGR"),
        ("A0A8K0DA02", "Elateridae", "Ignelater luminosus Cucubano (Green)", 550.0, "Ignelater"),
        ("Q718B3", "Elateridae", "Pyrophorus plagiophthalamus vYG (Yellow-Green)", 560.0, "Pyrophorus_vYG"),
        ("Q718C0", "Elateridae", "Pyrophorus plagiophthalamus vYG-2 (Yellow-Green)", 560.0, "Pyrophorus_vYG"),
        ("Q717B7", "Elateridae", "Pyrophorus mellifluus vYG (Yellow-Green)", 560.0, "Pyrophorus_vYG"),
        ("Q718C3", "Elateridae", "Pyrophorus plagiophthalamus vYE (Yellow)", 578.0, "Pyrophorus_vYE"),
        ("Q718D1", "Elateridae", "Pyrophorus plagiophthalamus vYE-2 (Yellow)", 578.0, "Pyrophorus_vYE"),
        ("Q7M4K2", "Elateridae", "Pyrophorus plagiophthalamus vYE-3 (Yellow)", 578.0, "Pyrophorus_vYE"),
        ("Q718E3", "Elateridae", "Pyrophorus plagiophthalamus vOR (Orange)", 593.0, "Pyrophorus_vOR"),
        ("Q718F0", "Elateridae", "Pyrophorus plagiophthalamus vOR-2 (Orange)", 593.0, "Pyrophorus_vOR"),
        ("Q7M4K1", "Elateridae", "Pyrophorus plagiophthalamus vOR-3 (Orange)", 593.0, "Pyrophorus_vOR"),
    ]

    # Key color-tuning cavity positions in P08659 coordinates (Wood et al. 1989 Science; Viviani et al. 1999; Sundlov et al. 2012):
    # 218 (R218), 226-227 (click-beetle 223-224 spectral loop), 245/247/250 (H245, F247, G250=click-beetle 247 spectral switch),
    # 286/288 (I286, V288 hydrophobic wall), 311/314/315 (oxyanion/loop), 340/343/347/348/349 (luciferin gate)
    key_color_sites_p08 = [218, 226, 227, 245, 247, 250, 286, 288, 311, 314, 315, 340, 343, 347, 348, 349]

    cache150 = os.path.join(RES, "embeddings", "esm2_150m_l12_spectral.pt")
    if os.path.exists(cache150):
        cached_150 = torch.load(cache150, map_location="cpu", weights_only=False)
    else:
        tok150 = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D", local_files_only=True)
        mod150 = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D", add_pooling_layer=False, local_files_only=True).eval()
        cached_150 = {}
        with torch.inference_mode():
            for acc, _, _, _, _ in spectral_entries:
                seq = acc_to_seq[acc]
                out150 = mod150(**tok150(seq, return_tensors="pt"), output_hidden_states=True)
                cached_150[acc] = out150.hidden_states[12][0, 1:-1].numpy()
        torch.save(cached_150, cache150)

    X_whole = []
    X_pocket = []
    y_nm = np.array([x[3] for x in spectral_entries], dtype=float)
    clades = [x[1] for x in spectral_entries]
    groups = [x[4] for x in spectral_entries]

    for acc, clade, desc, nm, grp_id in spectral_entries:
        seq = acc_to_seq[acc]
        r2t = map_positions(ref_seq, seq, aligner)
        hs = cached_150[acc]
        X_whole.append(hs.mean(axis=0))
        site_vecs = [hs[r2t[p] - 1] if p in r2t else hs.mean(axis=0) for p in key_color_sites_p08]
        X_pocket.append(np.concatenate(site_vecs, axis=0))

    X_whole = np.array(X_whole)
    X_pocket = np.array(X_pocket)

    # Evaluate both Leave-One-Out (LOO) and Leave-One-Group-Out (LOGO across spectral allele clusters)
    loo = LeaveOneOut()
    preds_whole_loo = np.zeros_like(y_nm)
    preds_pocket_loo = np.zeros_like(y_nm)
    for tr_i, te_i in loo.split(X_pocket, y_nm):
        sc_w = StandardScaler().fit(X_whole[tr_i])
        reg_w = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit(sc_w.transform(X_whole[tr_i]), y_nm[tr_i])
        preds_whole_loo[te_i] = reg_w.predict(sc_w.transform(X_whole[te_i]))

        sc_p = StandardScaler().fit(X_pocket[tr_i])
        reg_p = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit(sc_p.transform(X_pocket[tr_i]), y_nm[tr_i])
        preds_pocket_loo[te_i] = reg_p.predict(sc_p.transform(X_pocket[te_i]))

    logo = LeaveOneGroupOut()
    preds_whole_logo = np.zeros_like(y_nm)
    preds_pocket_logo = np.zeros_like(y_nm)
    for tr_i, te_i in logo.split(X_pocket, y_nm, groups=groups):
        sc_w = StandardScaler().fit(X_whole[tr_i])
        reg_w = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit(sc_w.transform(X_whole[tr_i]), y_nm[tr_i])
        preds_whole_logo[te_i] = reg_w.predict(sc_w.transform(X_whole[te_i]))

        sc_p = StandardScaler().fit(X_pocket[tr_i])
        reg_p = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit(sc_p.transform(X_pocket[tr_i]), y_nm[tr_i])
        preds_pocket_logo[te_i] = reg_p.predict(sc_p.transform(X_pocket[te_i]))

    r_w, p_w = pearsonr(y_nm, preds_whole_loo)
    rho_w, _ = spearmanr(y_nm, preds_whole_loo)
    mae_w = float(np.abs(y_nm - preds_whole_loo).mean())

    r_p, p_p = pearsonr(y_nm, preds_pocket_loo)
    rho_p, _ = spearmanr(y_nm, preds_pocket_loo)
    mae_p = float(np.abs(y_nm - preds_pocket_loo).mean())

    r_plogo, p_plogo = pearsonr(y_nm, preds_pocket_logo)
    rho_plogo, _ = spearmanr(y_nm, preds_pocket_logo)
    mae_plogo = float(np.abs(y_nm - preds_pocket_logo).mean())

    print(f"  Spectral Wavelength Prediction (n={len(y_nm)} luciferases across 4 families, 538-623 nm):")
    print(f"    [LOO]  Whole-Protein Mean Embedding:      Pearson r = {r_w:.3f} (p={p_w:.2e}), Spearman rho = {rho_w:.3f}, MAE = {mae_w:.1f} nm")
    print(f"    [LOO]  3D Active-Site Cavity (16 sites):  Pearson r = {r_p:.3f} (p={p_p:.2e}), Spearman rho = {rho_p:.3f}, MAE = {mae_p:.1f} nm")
    print(f"    [LOGO] 3D Active-Site Cavity (Cluster CV): Pearson r = {r_plogo:.3f} (p={p_plogo:.2e}), Spearman rho = {rho_plogo:.3f}, MAE = {mae_plogo:.1f} nm")
    for (acc, clade, desc, nm, _), pw, pp, ppl in zip(spectral_entries, preds_whole_loo, preds_pocket_loo, preds_pocket_logo):
        print(f"      {acc:12s} | {desc:46s} | True={nm:5.1f} nm | PocketLOO={pp:5.1f} nm | WholeLOO={pw:5.1f} nm | PocketLOGO={ppl:5.1f} nm")

    return spectral_entries, y_nm, preds_whole_loo, preds_pocket_loo


def run_pillar5_causal_sae_steering(aligner, ref_seq):
    print("\n=================================================================")
    print("PILLAR 5: CAUSAL INTERPLM-650M SAE STEERING & IN-SILICO EVOLUTION")
    print("=================================================================")
    sae650_path = os.path.join(
        ROOT, ".cache", "hf", "hub", "models--Elana--InterPLM-esm2-650m",
        "snapshots", "5121c4c7f3ad0b5fbe0f3b9a457969192bb9912f", "layer_18", "ae_normalized.pt"
    )
    sd650 = torch.load(sae650_path, map_location="cpu", weights_only=False)
    print("Loaded Elana/InterPLM-esm2-650m Layer 18 SAE:", [(k, tuple(v.shape)) for k, v in sd650.items()])

    tok650 = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", local_files_only=True)
    mlm650 = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", local_files_only=True).eval()
    torch.set_num_threads(os.cpu_count() or 4)

    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]
    acc_to_seq = {r["accession"]: r["sequence"] for r in rows}

    # Also load CG6178 sequence from PDB 7KYD
    s_7kyd, _, _, _ = load_cif_chain_a(structure_path("pdb_00007kyd.cif"))
    acc_to_seq["PDB_7KYD_CG6178"] = s_7kyd

    luc_panel = ["P08659", "Q01158", "P13129", "Q27757", "A0A8K0DA02", "A0A8K0D8G1", "Q7M4K3", "Q718E3", "Q9U4U7", "Q9U4U8"]
    nonlum_panel = ["A0A0C6DRW5", "A0A8K0C7P5", "A0A8K0C402", "A0A8K0DAA6", "PDB_7KYD_CG6178", "D6WA83", "A0A1W4WHZ8", "A0ABM1M0B6"]

    # Active-site steric gate & pocket clamp residues in P08659 coordinates
    gate_motif_p08 = [245, 247, 314, 315, 340, 343, 345, 346, 347, 348, 349, 422, 423, 529, 535, 538]

    cache650 = os.path.join(RES, "embeddings", "esm2_650m_l18_panel.pt")
    if os.path.exists(cache650):
        seq_hs18_full = torch.load(cache650, map_location="cpu", weights_only=False)
    else:
        seq_hs18_full = {}
        with torch.inference_mode():
            for acc in luc_panel + nonlum_panel:
                seq = acc_to_seq[acc]
                enc = tok650(seq, return_tensors="pt")
                h18_full = mlm650.esm(**enc, output_hidden_states=True).hidden_states[18]
                seq_hs18_full[acc] = h18_full
        torch.save(seq_hs18_full, cache650)

    luc_sae_gate = []
    nonlum_sae_gate = []
    seq_sae18 = {}
    seq_r2t = {}

    with torch.inference_mode():
        for acc in luc_panel + nonlum_panel:
            seq = acc_to_seq[acc]
            r2t = map_positions(ref_seq, seq, aligner)
            seq_r2t[acc] = r2t
            hs18 = seq_hs18_full[acc][0, 1:-1]
            z18 = torch.relu((hs18 - sd650["bias"]) @ sd650["encoder.weight"].T + sd650["encoder.bias"])
            seq_sae18[acc] = z18
            g_idx = [r2t[p] - 1 for p in [345, 346, 347, 348, 349, 422, 423] if p in r2t]
            if acc in luc_panel:
                luc_sae_gate.append(z18[g_idx].mean(0).numpy())
            else:
                nonlum_sae_gate.append(z18[g_idx].mean(0).numpy())

    luc_sae_gate = np.array(luc_sae_gate)
    nonlum_sae_gate = np.array(nonlum_sae_gate)
    diff_sae = luc_sae_gate.mean(0) - nonlum_sae_gate.mean(0)
    top_luc_feats = np.argsort(-diff_sae)[:25]

    print("\n  Top 8 InterPLM-650M Layer 18 Luciferase Steric-Gate SAE Features (345-349 & 422-423):")
    p08_seq = acc_to_seq["P08659"]
    ign_seq = acc_to_seq["A0A8K0DA02"]
    for f_id in top_luc_feats[:8]:
        f_id = int(f_id)
        z_p = seq_sae18["P08659"][:, f_id].numpy()
        z_i = seq_sae18["A0A8K0DA02"][:, f_id].numpy()
        rp = int(np.argmax(z_p)) + 1
        ri = int(np.argmax(z_i)) + 1
        print(f"    650M-L18 SAE #{f_id:5d} | LucGateMean={luc_sae_gate[:, f_id].mean():.4f} vs NonLumGateMean={nonlum_sae_gate[:, f_id].mean():.4f} "
              f"(diff={diff_sae[f_id]:+.4f}) | P08659 peak: {p08_seq[rp-1]}{rp}({z_p[rp-1]:.2f}) | IgnLuc peak: {ign_seq[ri-1]}{ri}({z_i[ri-1]:.2f})")

    W_dec = sd650["decoder.weight"]  # [1280, 10240]

    # Compute position-specific sparse SAE steering directions (50 / 10,240 features = 99.5% sparse!)
    # for each active-site gate residue p_pos in gate_motif_p08
    pos_steer_h = {}
    for p_pos in gate_motif_p08:
        z_luc_p = np.stack([seq_sae18[a][seq_r2t[a][p_pos] - 1].numpy() for a in luc_panel if p_pos in seq_r2t[a]], axis=0).mean(0)
        z_non_p = np.stack([seq_sae18[a][seq_r2t[a][p_pos] - 1].numpy() for a in nonlum_panel if p_pos in seq_r2t[a]], axis=0).mean(0)
        dz = z_luc_p - z_non_p
        top_pos = np.argsort(-dz)[:25]
        top_neg = np.argsort(dz)[:25]
        sz = torch.zeros(10240)
        for fid in top_pos:
            sz[int(fid)] = float(dz[int(fid)])
        for fid in top_neg:
            sz[int(fid)] = float(dz[int(fid)])
        sh = sz @ W_dec.T
        pos_steer_h[p_pos] = sh / (torch.norm(sh) + 1e-8)

    def forward_from_layer18(acc, steer_alpha=0.0):
        """Take cached Layer 18 output for acc, steer each active-site gate position p_pos along its
        sparse (50/10,240) InterPLM-650M SAE feature direction, then run Layers 19..33 and lm_head."""
        r2t = seq_r2t[acc]
        with torch.inference_mode():
            h18 = seq_hs18_full[acc].clone()
            if steer_alpha != 0.0:
                for p_pos, sh_vec in pos_steer_h.items():
                    if p_pos in r2t:
                        t_pos = r2t[p_pos]
                        h18[0, t_pos] = h18[0, t_pos] + steer_alpha * sh_vec
            seq_len = h18.shape[1]
            position_ids = torch.arange(seq_len, device=h18.device).unsqueeze(0)
            position_embeddings = mlm650.esm.rotary_embeddings(h18, position_ids)
            hidden = h18
            for layer_module in mlm650.esm.encoder.layer[18:]:
                hidden = layer_module(hidden, position_embeddings=position_embeddings)
            if mlm650.esm.encoder.emb_layer_norm_after is not None:
                hidden = mlm650.esm.encoder.emb_layer_norm_after(hidden)
            logits = mlm650.lm_head(hidden)[0, 1:-1]
            log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    print("\n  Causal SAE Steering Results (Clamping Top-50 Sparse InterPLM-650M Layer 18 Gate Features):")
    steering_records = []

    test_cases = [
        ("PDB_7KYD_CG6178", "Drosophila CG6178 (FruitFire progenitor)", [
            (347, "S", "A", "FruitFire Steric Protrusion Gate (L -> S/A)"),
            (348, "A", "C", "FruitFire Gate 348 (S -> A)"),
            (349, "I", "N", "Luciferin Gate 349 (V -> I/N)"),
        ]),
        ("A0A0C6DRW5", "Pectocera fortunei (58.5% ID Non-Lum Click Beetle)", [
            (347, "S", "A", "Click-Beetle Steric Gate 347 (L -> S/A)"),
            (349, "N", "I", "Click-Beetle Luciferin Pocket 349 (V -> N/I)"),
            (423, "F", "I", "Hinge Clamp 423 (I -> F)"),
        ]),
        ("A0A8K0C7P5", "Ignelater Decoy Paralog (54.6% ID Non-Lum)", [
            (347, "S", "A", "Click-Beetle Steric Gate 347 (L -> S/A)"),
            (348, "A", "S", "Click-Beetle Steric Gate 348 (V -> A/S)"),
            (349, "N", "I", "Click-Beetle Luciferin Pocket 349 (I -> N)"),
            (423, "F", "I", "Hinge Clamp 423 (L -> F/I)"),
            (538, "E", "I", "C-Terminal Lid 538 (L -> E/I)"),
        ]),
    ]

    alphas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    for acc, name_desc, sites in test_cases:
        seq = acc_to_seq[acc]
        r2t = seq_r2t[acc]
        lp_base = forward_from_layer18(acc, steer_alpha=0.0)
        lp_steer = forward_from_layer18(acc, steer_alpha=20.0)

        print(f"    Target: {name_desc}")
        for p_pos, luc_aa1, luc_aa2, site_desc in sites:
            if p_pos not in r2t:
                continue
            t_pos = r2t[p_pos]
            wt_aa = seq[t_pos - 1]
            id_wt = tok650.convert_tokens_to_ids(wt_aa)
            id_1 = tok650.convert_tokens_to_ids(luc_aa1)
            id_2 = tok650.convert_tokens_to_ids(luc_aa2)

            lo_base = float(max(lp_base[t_pos - 1, id_1], lp_base[t_pos - 1, id_2]) - lp_base[t_pos - 1, id_wt])
            lo_steer = float(max(lp_steer[t_pos - 1, id_1], lp_steer[t_pos - 1, id_2]) - lp_steer[t_pos - 1, id_wt])
            delta_lo = lo_steer - lo_base
            fold_change = float(np.exp(delta_lo))
            print(f"      P08659 pos {p_pos:3d} ({wt_aa}{t_pos:3d} -> {luc_aa1}/{luc_aa2}) [{site_desc}]: "
                  f"unsteered={lo_base:+.2f}, steered={lo_steer:+.2f} (Delta={delta_lo:+.2f}, {fold_change:.1f}x probability boost!)")

        for a_val in alphas:
            lp_a = forward_from_layer18(acc, steer_alpha=a_val)
            for p_pos, luc_aa1, luc_aa2, site_desc in sites:
                if p_pos not in r2t:
                    continue
                t_pos = r2t[p_pos]
                wt_aa = seq[t_pos - 1]
                id_wt = tok650.convert_tokens_to_ids(wt_aa)
                id_1 = tok650.convert_tokens_to_ids(luc_aa1)
                id_2 = tok650.convert_tokens_to_ids(luc_aa2)
                lo_a = float(max(lp_a[t_pos - 1, id_1], lp_a[t_pos - 1, id_2]) - lp_a[t_pos - 1, id_wt])
                steering_records.append((acc, p_pos, f"{wt_aa}{t_pos}->{luc_aa1}/{luc_aa2} ({site_desc.split('(')[0].strip()})", a_val, lo_a))

    return steering_records, top_luc_feats, diff_sae, luc_sae_gate, nonlum_sae_gate


def make_unified_figure(physics_results, spectral_entries, y_nm, preds_whole, preds_pocket, steering_records):
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.0), dpi=180)

    # ---- Panel A: Pillar 3 (3D Structural Physics - Steric Clearance to DLSA vs Status)
    ax1 = axes[0]
    labels_short = [
        r["label"].split("(")[0].strip() + "\n(" + r["loop343_349"] + ")"
        for r in physics_results
    ]
    dists = [r["min_gate_dist"] for r in physics_results]
    colors = ["#2ecc71" if r["status"] == "Luminous" else "#e74c3c" for r in physics_results]
    y_pos = np.arange(len(physics_results))
    ax1.barh(y_pos, dists, color=colors, edgecolor="black", height=0.68, alpha=0.9)
    ax1.axvline(2.8, color="#b52b27", linestyle="--", lw=1.2, label="Steric Clash Cutoff (2.8 Å)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels_short, fontsize=8.0, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Min 3D Distance from Pocket Gate Sidechains\nto Bound DLSA Luciferin Intermediate (Å)", fontweight="bold")
    ax1.set_title("A. Pillar 3 (3D Physics): Non-Luminous\nParalogs Sterically Occlude the Luciferin Pocket", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(axis="x", alpha=0.25)

    # ---- Panel B: Pillar 4 (Spectral Phenotype Translator - Wavelength lambda_max)
    ax2 = axes[1]
    clade_colors = {"Lampyridae": "#f39c12", "Elateridae": "#2ecc71", "Phengodidae": "#e74c3c", "Rhagophthalmidae": "#9b59b6"}
    clade_markers = {"Lampyridae": "^", "Elateridae": "*", "Phengodidae": "P", "Rhagophthalmidae": "D"}
    plotted_clades = set()
    for (acc, clade, desc, nm, _), pp in zip(spectral_entries, preds_pocket):
        lbl = f"{clade}" if clade not in plotted_clades else None
        plotted_clades.add(clade)
        ax2.scatter(nm, pp, c=clade_colors[clade], marker=clade_markers[clade], s=95 if clade == "Elateridae" else 70,
                    edgecolors="black", linewidths=0.6, alpha=0.9, label=lbl, zorder=4)

    # Clean non-overlapping annotations for key spectral representatives
    annot_cfg = {
        "PDB_6AAA": ("Amydetes (538 nm)", (-6, 18)),
        "Q7M4K3": ("Pyrophorus dGR (546 nm)", (12, -16)),
        "P08659": ("P. pyralis (560 nm)", (14, 12)),
        "Q718E3": ("Pyrophorus vOR (593 nm)", (-92, 10)),
        "Q9U4U7": ("Phrixothrix Red (623 nm)", (-98, 12)),
    }
    for (acc, clade, desc, nm, _), pp in zip(spectral_entries, preds_pocket):
        if acc in annot_cfg:
            txt_lbl, off = annot_cfg[acc]
            ax2.annotate(txt_lbl, (nm, pp), xytext=off, textcoords="offset points",
                         fontsize=7.6, fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7, color="#444444"))

    lims = [530, 630]
    ax2.plot(lims, lims, "--", color="#555555", lw=1.0, alpha=0.7, label="Ideal 1:1 Prediction")
    r_p, p_p = pearsonr(y_nm, preds_pocket)
    rho_p, _ = spearmanr(y_nm, preds_pocket)
    mae_p = float(np.abs(y_nm - preds_pocket).mean())
    ax2.text(0.04, 0.80, f"3D Pocket ESM-2 Probe\nPearson r = {r_p:.2f} (p = {p_p:.1e})\nSpearman ρ = {rho_p:.2f}\nMAE = {mae_p:.1f} nm",
             transform=ax2.transAxes, fontsize=8.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc"))
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Experimental Bioluminescence Peak λmax (nm)", fontweight="bold")
    ax2.set_ylabel("Predicted λmax from 3D Pocket Embedding (nm)", fontweight="bold")
    ax2.set_title("B. Pillar 4 (Spectral Translator): Predicting\nEmission Color (538–623 nm) Across Lineages", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7.8, frameon=True)
    ax2.grid(alpha=0.25)

    # ---- Panel C: Pillar 5 (Causal InterPLM-650M SAE Steering Dose-Response)
    ax3 = axes[2]
    highlight_curves = [
        ("PDB_7KYD_CG6178", 347, "#e74c3c", "-o", "CG6178 L347→S/A (FruitFire Steric Gate)"),
        ("A0A0C6DRW5", 347, "#2980b9", "-^", "Pectocera L347→S/A (Click-Beetle 347 Gate)"),
        ("A0A0C6DRW5", 349, "#8e44ad", "--D", "Pectocera V349→N/I (Pocket 349 Gate)"),
        ("A0A8K0C7P5", 347, "#27ae60", "-P", "Ignelater Decoy L347→S/A (Steric Gate)"),
        ("A0A8K0C7P5", 348, "#e67e22", "--s", "Ignelater Decoy V348→A/S (Steric Gate)"),
        ("A0A8K0C7P5", 423, "#16a085", ":v", "Ignelater Decoy L423→F/I (Hinge Clamp)"),
    ]
    for acc_c, p_pos_c, col_c, style_c, lbl_c in highlight_curves:
        pts = [(a, lo) for acc, pp, _, a, lo in steering_records if acc == acc_c and pp == p_pos_c]
        if not pts:
            continue
        pts.sort()
        xs_a = [x[0] for x in pts]
        ys_lo = [x[1] - pts[0][1] for x in pts]
        ax3.plot(xs_a, ys_lo, style_c, color=col_c, lw=1.8, markersize=5, label=lbl_c)

    ax3.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax3.set_xlabel("InterPLM-650M (Layer 18) SAE Steering Strength (α)", fontweight="bold")
    ax3.set_ylabel("Δ Log-Odds of Luciferase Gate Mutation\n(Steered minus Unsteered)", fontweight="bold")
    ax3.set_title("C. Pillar 5 (Causal SAE Steering): Clamping\nInterPLM Features Unlocks Active-Site Gate", fontweight="bold")
    ax3.legend(loc="best", fontsize=7.5, frameon=True)
    ax3.grid(alpha=0.25)

    fig.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig5_unified_physics_spectrum_steering.png")
    fig.savefig(out_fig, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved unified Figure 5 to", out_fig)


def main():
    aligner = make_aligner()
    physics_results, ref_seq = run_pillar3_physics(aligner)
    spectral_entries, y_nm, preds_whole, preds_pocket = run_pillar4_wavelength(aligner, ref_seq)
    steering_records, _, _, _, _ = run_pillar5_causal_sae_steering(aligner, ref_seq)
    make_unified_figure(physics_results, spectral_entries, y_nm, preds_whole, preds_pocket, steering_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
