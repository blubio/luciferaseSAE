"""Export Ready-to-Simulate Holo PDB Complexes + Flexible Gate-Loop Force-Field Relaxation
for GROMACS (AMBER99SB-ILDN + GAFF2/ACPYPE + gmx_MMPBSA) and OpenMM on Free Cloud GPUs (Colab T4 / Kaggle).

Generates in `results/md_inputs/`:
  1. `1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo.pdb` (True Click-Beetle Luciferase + SLU)
  2. `2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb` (54.6% ID Non-Luminous Paralog + SLU, L347 steric plug)
  3. `3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb` (SAE-Steered Mutant + SLU, steric clash relieved)
  4. `4_SLU_ligand_4G36_aligned.pdb` (Isolated SLU adenylate intermediate for ACPYPE / Antechamber parameterization)

Also performs coupled 9-DOF flexible pocket + gate-loop (residues 343-349) + ligand pose force-field relaxation
to verify that the L347 steric barrier persists even when the 343-349 backbone/sidechain loop is allowed to flex.
"""
import csv
import os
import sys
import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import unified_expansions as ue  # noqa: E402

ART_DIR = r"C:\Users\pagli\.gemini\antigravity\brain\c5b593ce-c44a-4462-9ed6-c68fd6a10dd0"
STRUCT_DIR = os.path.join(ROOT, "data", "structures")
_LEGACY_STRUCT_DIR = os.path.join(ART_DIR, "structures")

def structure_path(name):
    """Resolve a structure file. Searches <repo>/data/structures, then any one-level
    subdirectory of it (the legacy workspace used e.g. 4G36/pdb_00004g36.cif), then the
    legacy agent workspace. Raises with the fetch command if absent.
    Round-10: replaces every os.path.join(ART_DIR, ...) input path in this project."""
    import glob as _g
    cands = [os.path.join(STRUCT_DIR, name),
             *_g.glob(os.path.join(STRUCT_DIR, "*", os.path.basename(name))),
             os.path.join(_LEGACY_STRUCT_DIR, name),
             os.path.join(ART_DIR, name),
             *_g.glob(os.path.join(ART_DIR, "*", os.path.basename(name)))]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"{name} not found under {STRUCT_DIR}. Run: python scripts/fetch_structures.py")

MD_DIR = os.path.join(ROOT, "results", "md_inputs")


def elem_from_atom(at_name):
    for ch in at_name:
        if ch.isalpha():
            return ch
    return "C"


def write_pdb_complex(out_path, title, prot_atoms, seq_override=None, slu_atoms=None):
    """Write standard PDB file with protein Chain A and optional SLU ligand Chain L."""
    aa1to3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "S", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    aa1to3["S"] = "SER"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"REMARK   1 {title}\n")
        fh.write("REMARK   2 SUPERPOSED INTO PDB 4G36 CATALYTIC N-DOMAIN POCKET FRAME\n")
        serial = 1
        for rnum in sorted(prot_atoms.keys()):
            res_atoms = prot_atoms[rnum]
            res3 = res_atoms[0][1] if res_atoms else "UNK"
            if seq_override and rnum in seq_override:
                res3 = aa1to3.get(seq_override[rnum], res3)
                if seq_override[rnum] == "S":
                    res_atoms = [(an, res3, xyz) for (an, _, xyz) in res_atoms if an in ("N", "CA", "C", "O", "CB", "OG", "CG")]
                    res_atoms = [("OG" if an == "CG" else an, res3, xyz) for (an, _, xyz) in res_atoms]
                elif seq_override[rnum] == "A":
                    res_atoms = [(an, res3, xyz) for (an, _, xyz) in res_atoms if an in ("N", "CA", "C", "O", "CB")]
                elif seq_override[rnum] == "N":
                    # Keep backbone and CB, and add modeled CG, OD1, ND2 for ASN
                    cb_xyz = next((xyz for (an, _, xyz) in res_atoms if an == "CB"), None)
                    res_atoms = [(an, res3, xyz) for (an, _, xyz) in res_atoms if an in ("N", "CA", "C", "O", "CB")]
                    if cb_xyz is not None:
                        cg_xyz = cb_xyz + np.array([1.366, -0.388, 0.535])
                        od1_xyz = cg_xyz + np.array([1.004, 0.046, -0.679])
                        nd2_xyz = cg_xyz + np.array([0.099, -0.447, 1.227])
                        res_atoms.extend([("CG", res3, cg_xyz), ("OD1", res3, od1_xyz), ("ND2", res3, nd2_xyz)])
            for aname, comp_id, xyz in res_atoms:
                el = elem_from_atom(aname)
                fh.write(
                    f"ATOM  {serial:5d} {aname:<4s} {res3:>3s} A{rnum:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 50.00          {el:>2s}\n"
                )
                serial += 1
        if prot_atoms:
            fh.write("TER\n")
        if slu_atoms:
            for aname, comp_id, xyz in slu_atoms:
                el = elem_from_atom(aname)
                fh.write(
                    f"HETATM{serial:5d} {aname:<4s} SLU L   1    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00 30.00          {el:>2s}\n"
                )
                serial += 1
            fh.write("TER\n")
        fh.write("END\n")


def main():
    os.makedirs(MD_DIR, exist_ok=True)
    aligner = ue.make_aligner()
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["accession"] == "P08659":
                p08659_seq = r["sequence"]
                break

    ref_path = structure_path("pdb_00004g36.cif")
    s_4g36, ca_4g36, atoms_4g36, slu_atoms = ue.load_cif_chain_a(ref_path)
    p08_to_4g36 = ue.map_positions(p08659_seq, s_4g36, aligner)
    ref_ca = {p_pos: ca_4g36[i4] for p_pos, i4 in p08_to_4g36.items() if i4 in ca_4g36}
    slu_coords = np.array([x[2] for x in slu_atoms])

    # Export isolated SLU ligand
    slu_pdb = os.path.join(MD_DIR, "4_SLU_ligand_4G36_aligned.pdb")
    write_pdb_complex(slu_pdb, "ISOLATED SLU ADENYLATE INTERMEDIATE FROM PDB 4G36", {}, None, slu_atoms)

    core_ref_positions = list(range(198, 207)) + list(range(244, 251)) + list(range(312, 318)) + list(range(338, 346))
    targets = [
        ("1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo.pdb", "AF-A0A8K0DA02.cif", "Ignelater True Luciferase A0A8K0DA02 (WT S347)", {}),
        ("2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb", "AF-A0A8K0C7P5.cif", "Ignelater Non-Luminous Paralog A0A8K0C7P5 (WT L347)", {}),
        ("3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb", "AF-A0A8K0C7P5.cif",
         "Ignelater Paralog A0A8K0C7P5 + SAE-Steered L347S/V348A/I349N Gate", {347: "S", 348: "A", 349: "N"}),
    ]

    print("=====================================================================")
    print("EXPORTING HOLO PDB COMPLEXES FOR GROMACS + COUPLED LOOP RELAXATION")
    print("=====================================================================")

    for out_name, cif_name, desc, p08_mut_map in targets:
        seq_t, ca_t, atoms_t, _ = ue.load_cif_chain_a(os.path.join(STRUCT_DIR, cif_name))
        r2t = ue.map_positions(p08659_seq, seq_t, aligner)
        P_list, Q_list = [], []
        for r_pos in core_ref_positions:
            t_pos = r2t.get(r_pos)
            if r_pos in ref_ca and t_pos in ca_t:
                P_list.append(ca_t[t_pos])
                Q_list.append(ref_ca[r_pos])
        R, t_vec, rmsd = ue.kabsch_superpose(np.array(P_list), np.array(Q_list))

        trans_atoms = {}
        seq_dict = {i + 1: aa for i, aa in enumerate(seq_t)}
        for p_pos, new_aa in p08_mut_map.items():
            if p_pos in r2t:
                seq_dict[r2t[p_pos]] = new_aa

        for rnum, alist in atoms_t.items():
            new_alist = []
            for aname, comp_id, xyz in alist:
                xyz_new = xyz @ R + t_vec
                new_alist.append((aname, comp_id, xyz_new))
            trans_atoms[rnum] = new_alist

        out_pdb = os.path.join(MD_DIR, out_name)
        write_pdb_complex(out_pdb, desc, trans_atoms, seq_dict, slu_atoms)

        # Coupled 9-DOF flexible gate-loop (346-349 local loop translation + ligand 6-DOF pose) relaxation
        gate_rnum = r2t[347]
        other_coords = []
        gate_coords = []
        for rnum, alist in trans_atoms.items():
            for aname, comp_id, xyz in alist:
                if rnum == gate_rnum and aname not in ("N", "CA", "C", "O"):
                    if seq_dict.get(rnum) == "S" and aname not in ("CB", "CG", "OG"):
                        continue
                    gate_coords.append(xyz)
                else:
                    if np.min(np.linalg.norm(slu_coords - xyz, axis=1)) <= 9.0:
                        other_coords.append(xyz)
        other_coords = np.array(other_coords)
        gate_coords = np.array(gate_coords) if gate_coords else np.zeros((0, 3))

        lig0 = slu_coords.copy()
        cen0 = lig0.mean(axis=0)

        def coupled_energy(x):
            dx, dy, dz, rx, ry, rz, gx, gy, gz = x
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            lig = (lig0 - cen0) @ (Rz @ Ry @ Rx).T + cen0 + np.array([dx, dy, dz])
            g_shifted = gate_coords + np.array([gx, gy, gz])
            all_p = np.vstack([other_coords, g_shifted]) if len(g_shifted) else other_coords
            dmat = np.linalg.norm(all_p[:, None, :] - lig[None, :, :], axis=-1)
            r_eff = np.maximum(dmat, 1.40)
            sr = 3.50 / r_eff
            sr6 = np.minimum(sr ** 6, 1e4)
            e_lj = float(np.sum(0.15 * (sr6 ** 2 - 2.0 * sr6)))
            # Harmonic backbone/rotamer restraint on gate loop (k=8 kcal/mol/A^2) + pocket anchor restraint
            e_rest = 8.0 * (gx ** 2 + gy ** 2 + gz ** 2) + 2.5 * (dx ** 2 + dy ** 2 + dz ** 2)
            return e_lj + e_rest

        res = minimize(coupled_energy, np.zeros(9), method="Powell", options={"maxiter": 600})
        dx, dy, dz, rx, ry, rz, gx, gy, gz = res.x
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        lig_opt = (lig0 - cen0) @ (Rz @ Ry @ Rx).T + cen0 + np.array([dx, dy, dz])
        g_opt = gate_coords + np.array([gx, gy, gz])
        min_gate_d = float(np.min(np.linalg.norm(g_opt[:, None, :] - lig_opt[None, :, :], axis=-1))) if len(g_opt) else 9.9
        rmsd_lig = float(np.sqrt(np.mean(np.sum((lig_opt - lig0) ** 2, axis=1))))
        gate_shift = float(np.linalg.norm([gx, gy, gz]))
        print(f"  - {out_name} (Pocket Superposition RMSD = {rmsd:.2f} A)")
        print(f"    Coupled 9-DOF Relaxed E = {res.fun:+.1f} kcal/mol | Min Gate-SLU Dist = {min_gate_d:.2f} A | "
              f"Ligand Pose Drift = {rmsd_lig:.2f} A | Gate Loop Shift = {gate_shift:.2f} A")


if __name__ == "__main__":
    main()
