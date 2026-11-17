import os

standard_heavy = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"}
}

root = r"c:\Users\pagli\Desktop\translator"
for fname in ["1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo.pdb", "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb", "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb"]:
    path = os.path.join(root, "results", "md_inputs", fname)
    residues = {}
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                aname = line[12:16].strip()
                resname = line[17:20].strip()
                resnum = int(line[22:26])
                key = (resnum, resname)
                if key not in residues:
                    residues[key] = set()
                residues[key].add(aname)
    print(f"Checking {fname}...")
    issues = 0
    for (rnum, rname), atoms in residues.items():
        if rname in standard_heavy:
            expected = standard_heavy[rname]
            missing = expected - atoms
            extra = atoms - expected
            if missing or extra:
                print(f"  Residue {rnum} ({rname}): Missing={missing}, Extra={extra}")
                issues += 1
    if issues == 0:
        print("  -> All residues have 100% complete standard heavy atom sets!")
    else:
        print(f"  -> Total issues: {issues}")
