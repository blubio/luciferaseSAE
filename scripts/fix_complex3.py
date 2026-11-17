import os

root = r"c:\Users\pagli\Desktop\translator"
p3 = os.path.join(root, "results", "md_inputs", "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb")

with open(p3, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("ATOM") and "ASN A 345" in line:
        aname = line[12:16].strip()
        if aname == "CD1":
            # Replace CD1 with CG, OD1, ND2
            # CB coords in PDB: (-34.837,  25.576, -18.910)
            cg_x, cg_y, cg_z = -33.471, 25.188, -18.375
            od1_x, od1_y, od1_z = -32.467, 25.234, -19.054
            nd2_x, nd2_y, nd2_z = -33.372, 24.741, -17.148
            
            serial = int(line[6:11])
            new_lines.append(f"ATOM  {serial:5d}  CG  ASN A 345    {cg_x:8.3f}{cg_y:8.3f}{cg_z:8.3f}  1.00 50.00           C\n")
            new_lines.append(f"ATOM  {serial+1:5d}  OD1 ASN A 345    {od1_x:8.3f}{od1_y:8.3f}{od1_z:8.3f}  1.00 50.00           O\n")
            new_lines.append(f"ATOM  {serial+2:5d}  ND2 ASN A 345    {nd2_x:8.3f}{nd2_y:8.3f}{nd2_z:8.3f}  1.00 50.00           N\n")
            continue
    new_lines.append(line)

# Renumber serials
final_lines = []
serial = 1
for line in new_lines:
    if line.startswith(("ATOM", "HETATM")):
        line = line[:6] + f"{serial:5d}" + line[11:]
        serial += 1
    final_lines.append(line)

with open(p3, "w") as f:
    f.writelines(final_lines)

print("✅ Updated 3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb with full ASN sidechain atoms!")
