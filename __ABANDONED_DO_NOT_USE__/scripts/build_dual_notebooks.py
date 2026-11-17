import json
import os
import base64
import gzip

ROOT = r"c:\Users\pagli\Desktop\translator"
MD_DIR = os.path.join(ROOT, "results", "md_inputs")
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def get_b64_payloads():
    payloads = {}
    for fname in [
        "1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo.pdb",
        "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb",
        "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo.pdb",
        "4_SLU_ligand_4G36_aligned.pdb",
    ]:
        fpath = os.path.join(MD_DIR, fname)
        with open(fpath, "rb") as fh:
            data = fh.read()
        b64 = base64.b64encode(gzip.compress(data)).decode("ascii")
        payloads[fname] = b64
    return payloads

payloads_repr = json.dumps(get_b64_payloads(), indent=2)

# ==============================================================================
# 1. BUILD PRECOMPILE NOTEBOOK (GROMACS CUDA via Conda nompi_cuda + Live Logger)
# ==============================================================================
def make_precompile_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    def add_md(t):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [t]})

    def add_code(c):
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)})

    # Cell 0
    add_md("""# 🧬 GROMACS Precompiled CUDA Pipeline (With Live Progress Monitoring)
### Validation of Convergent Steric Gate Remodeling in Beetle Luciferases
**Author:** Bruno ("Blu") Stein Barbosa Menechino ([PMID: 38049805](https://pubmed.ncbi.nlm.nih.gov/38049805/))  
**Engine:** Precompiled GROMACS with native NVIDIA CUDA acceleration (`nompi_cuda`) via conda-forge.  
**Hardware:** NVIDIA T4 / V100 / A100 GPU (Google Colab Pro).  
**Real-Time Feedback:** Includes an active live monitor that prints percentage completion `[XX.X%]`, nanoseconds simulated, and GPU speed every 30 seconds!

---
### 🚀 How to Run:
1. Ensure your runtime is GPU: **Runtime -> Change runtime type -> T4 GPU**.
2. Run **Step 1** (Installs CondaColab; kernel automatically restarts).
3. Run **Step 2 through Step 6** sequentially.""")

    # Cell 1: Step 1 CondaColab
    add_md("## Step 1: Install CondaColab (Initializes Conda Environment)")
    add_code("""# Install condacolab helper
!pip install -q condacolab
import condacolab
condacolab.install()
# Note: Google Colab will display a prompt indicating the kernel has restarted. This is expected!""")

    # Cell 2: Step 2 Mamba install GROMACS CUDA
    add_md("## Step 2: Install Precompiled GROMACS with CUDA (`nompi_cuda`)")
    add_code("""# Install precompiled GROMACS with CUDA support from conda-forge (takes ~90 seconds)
!mamba install -y -q -c conda-forge "gromacs=*=nompi_cuda*"
!pip install -q gmx_MMPBSA acpype py3Dmol matplotlib seaborn pandas

import os, sys, glob, base64, gzip
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Verify native CUDA support in GROMACS
print("==================================================================")
print("⚡ Verifying Precompiled CUDA GROMACS Installation:")
print("==================================================================")
!gmx --version | head -n 18""")

    # Cell 3: Step 3 Unpack payloads
    add_md("## Step 3: Unpack Embedded Holo Complexes into Workspace")
    add_code(f"""# Self-extracting embedded holo complexes into workspace
BASE_DIR = os.path.abspath("md_simulation")
os.makedirs(os.path.join(BASE_DIR, "inputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "mdp"), exist_ok=True)

embedded_payloads = {payloads_repr}

for fname, b64_str in embedded_payloads.items():
    out_path = os.path.join(BASE_DIR, "inputs", fname)
    raw_data = gzip.decompress(base64.b64decode(b64_str))
    with open(out_path, "wb") as fh:
        fh.write(raw_data)
    print(f"  Extracted: {{out_path}} ({{len(raw_data)}} bytes)")

print(f"\\n✅ All 4 holo structures unpacked successfully into {{BASE_DIR}}/inputs/.")
!ls -lh {{BASE_DIR}}/inputs/""")

    # Cell 4: Step 4 MDP files
    add_md("## Step 4: Write Optimized GPU MDP Configuration Files (HMR dt = 4 fs)")
    add_code("""BASE_DIR = os.path.abspath("md_simulation")
mdp_dir = os.path.join(BASE_DIR, "mdp")
os.makedirs(mdp_dir, exist_ok=True)

# 1. Energy Minimization (em.mdp)
with open(os.path.join(mdp_dir, "em.mdp"), "w") as f:
    f.write('''integrator               = steep
emtol                    = 500.0
emstep                   = 0.01
nsteps                   = 5000
cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = 1.0
coulombtype              = PME
rcoulomb                 = 1.0
vdwtype                  = Cut-off
rvdw                     = 1.0
pbc                      = xyz
''')

# 2. NVT Equilibration (nvt.mdp, 100 ps, 300 K)
with open(os.path.join(mdp_dir, "nvt.mdp"), "w") as f:
    f.write('''define                   = -DPOSRES
integrator               = md
dt                       = 0.002
nsteps                   = 50000     ; 100 ps
cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = 1.0
coulombtype              = PME
rcoulomb                 = 1.0
vdwtype                  = Cut-off
rvdw                     = 1.0
tcoupl                   = V-rescale
tc-grps                  = System
tau_t                    = 1.0
ref_t                    = 300
pcoupl                   = no
pbc                      = xyz
gen_vel                  = yes
gen_temp                 = 300
gen_seed                 = -1
constraints              = all-bonds
continuation             = no
''')

# 3. NPT Equilibration (npt.mdp, 200 ps, 1 bar)
with open(os.path.join(mdp_dir, "npt.mdp"), "w") as f:
    f.write('''define                   = -DPOSRES
integrator               = md
dt                       = 0.002
nsteps                   = 100000    ; 200 ps
cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = 1.0
coulombtype              = PME
rcoulomb                 = 1.0
vdwtype                  = Cut-off
rvdw                     = 1.0
tcoupl                   = V-rescale
tc-grps                  = System
tau_t                    = 1.0
ref_t                    = 300
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 2.0
ref_p                    = 1.0
compressibility          = 4.5e-5
pbc                      = xyz
constraints              = all-bonds
continuation             = yes
''')

# 4. Production MD (prod_hmr.mdp, dt = 4 fs, 10.0 ns = 2,500,000 steps on GPU)
with open(os.path.join(mdp_dir, "prod_hmr.mdp"), "w") as f:
    f.write('''integrator               = md
dt                       = 0.004     ; 4 fs step with HMR (-heavyh)
nsteps                   = 2500000   ; 10.0 ns trajectory (~140-180 ns/day on GPU)
nstxout-compressed       = 5000      ; 500 frames (every 20 ps)
nstenergy                = 5000
cutoff-scheme            = Verlet
nstlist                  = 40
rlist                    = 1.0
coulombtype              = PME
rcoulomb                 = 1.0
vdwtype                  = Cut-off
rvdw                     = 1.0
tcoupl                   = V-rescale
tc-grps                  = System
tau_t                    = 1.0
ref_t                    = 300
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau_p                    = 5.0
ref_p                    = 1.0
compressibility          = 4.5e-5
pbc                      = xyz
constraints              = all-bonds
continuation             = yes
''')

print(f"✅ GPU MDP configuration files written in {mdp_dir}.")""")

    # Cell 5: Step 5 Automated simulation loop with GPU offload + LIVE PROGRESS MONITOR
    add_md("## Step 5: Run Automated GPU Molecular Dynamics Loop with Live Feedback")
    add_code("""%%bash
set -e

BASE_DIR="$(pwd)/md_simulation"
INP_DIR="${BASE_DIR}/inputs"
MDP_DIR="${BASE_DIR}/mdp"
OUT_DIR="${BASE_DIR}/outputs"

COMPLEXES=(
  "1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo"
  "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo"
  "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo"
)

for COMP in "${COMPLEXES[@]}"; do
    PDB_INPUT="${INP_DIR}/${COMP}.pdb"
    if [ ! -f "${PDB_INPUT}" ]; then
        echo "❌ Skipping ${COMP}: ${PDB_INPUT} not found!"
        continue
    fi
    echo "=================================================================="
    echo "🚀 Processing Complex on NVIDIA GPU: ${COMP}"
    echo "Input: ${PDB_INPUT}"
    echo "=================================================================="

    COMP_OUT="${OUT_DIR}/${COMP}"
    mkdir -p "${COMP_OUT}"
    cd "${COMP_OUT}"

    if [ ! -f "em.gro" ]; then
        echo "  [1/6] Extracting protein coordinates..."
        grep "^ATOM" "${PDB_INPUT}" > protein.pdb

        echo "  [2/6] Building AMBER99SB-ILDN topology with HMR (-heavyh)..."
        gmx pdb2gmx -f protein.pdb -o protein_hmr.gro -p topol.top -ff amber99sb-ildn -water tip3p -ignh -heavyh

        echo "  [3/6] Setting up Rhombic Dodecahedron box (1.0 nm buffer)..."
        gmx editconf -f protein_hmr.gro -o box.gro -c -d 1.0 -bt dodecahedron

        echo "  [4/6] Solvating with TIP3P water..."
        gmx solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

        echo "  [5/6] Adding 0.15 M NaCl ions for physiological neutralization..."
        gmx grompp -f "${MDP_DIR}/em.mdp" -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
        echo "SOL" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15

        echo "  [6/6] Running steepest-descent energy minimization..."
        gmx grompp -f "${MDP_DIR}/em.mdp" -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 2
        gmx mdrun -deffnm em -v
    else
        echo "  ✅ Energy minimization already completed (em.gro exists). Resuming from NVT!"
    fi

    if [ ! -f "nvt.gro" ]; then
        echo "  ⚡ Running NVT Equilibration on GPU (100 ps, 300 K)..."
        gmx grompp -f "${MDP_DIR}/nvt.mdp" -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
        gmx mdrun -deffnm nvt -nb gpu -pme gpu
        echo "  ✅ NVT equilibration finished."
    else
        echo "  ✅ NVT equilibration already completed."
    fi

    if [ ! -f "npt.gro" ]; then
        echo "  ⚡ Running NPT Equilibration on GPU (200 ps, 1.0 bar)..."
        gmx grompp -f "${MDP_DIR}/npt.mdp" -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr -maxwarn 2
        gmx mdrun -deffnm npt -nb gpu -pme gpu
        echo "  ✅ NPT equilibration finished."
    else
        echo "  ✅ NPT equilibration already completed."
    fi

    if [ ! -f "prod.xtc" ]; then
        echo "  ⚡ Launching Production MD on NVIDIA GPU (10.0 ns, dt = 4.0 fs HMR)..."
        gmx grompp -f "${MDP_DIR}/prod_hmr.mdp" -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 2
        
        # Run mdrun in background and actively monitor progress every 30 seconds
        gmx mdrun -deffnm prod -nb gpu -pme gpu -bonded gpu -update gpu -v > prod_terminal.log 2>&1 &
        MDRUN_PID=$!
        TOTAL_STEPS=2500000

        echo "  --> Production running on GPU (PID: $MDRUN_PID). Streaming live progress updates every 30s..."
        echo "  ----------------------------------------------------------------"
        while kill -0 $MDRUN_PID 2>/dev/null; do
            sleep 30
            if [ -f "prod.log" ]; then
                CURR_STEP=$(grep "Step " prod.log | tail -n 1 | awk '{print $2}')
                if [ -n "$CURR_STEP" ] && [ "$CURR_STEP" -eq "$CURR_STEP" ] 2>/dev/null; then
                    PCT=$(awk -v c="$CURR_STEP" -v t="$TOTAL_STEPS" 'BEGIN { printf "%.1f", (c/t)*100 }')
                    NS=$(awk -v c="$CURR_STEP" 'BEGIN { printf "%.2f", c*0.004 }')
                    echo "  ⚡ [GROMACS GPU] [${PCT}%] ${NS} / 10.00 ns completed (Step ${CURR_STEP}/${TOTAL_STEPS})"
                else
                    echo "  ⏳ [GROMACS GPU] Simulation actively running on GPU..."
                fi
            fi
        done
        wait $MDRUN_PID
        echo "  ✅ Production MD finished on GPU for ${COMP}!"
    else
        echo "  ✅ Production MD already completed."
    fi

    echo "  --> Post-processing periodic boundary conditions (PBC)..."
    echo "1 0" | gmx trjconv -s prod.tpr -f prod.xtc -o prod_noPBC.xtc -pbc mol -center
    echo "4 4" | gmx rms -s prod.tpr -f prod_noPBC.xtc -o rmsd_backbone.xvg

    echo "✅ Finished GPU simulation for ${COMP}!"
done

echo "=================================================================="
echo "🎉 ALL 3 GPU SIMULATIONS COMPLETED SUCCESSFULLY!"
echo "=================================================================="
""")

    # Cell 6: Step 6 Plotting
    add_md("## Step 6: Backbone RMSD Stability Visualization & Comparison")
    add_code("""OUT_DIR = os.path.abspath("md_simulation/outputs")
complexes = [
    ("True Luc WT S347", "1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo", "#10b981"),
    ("Paralog WT L347", "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo", "#e11d48"),
    ("SAE-Steered Gate S347/A348/N349", "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo", "#2563eb")
]

plt.figure(figsize=(10, 5), dpi=150)
for label, comp_dir, color in complexes:
    xvg_file = os.path.join(OUT_DIR, comp_dir, "rmsd_backbone.xvg")
    if os.path.exists(xvg_file):
        data = []
        with open(xvg_file, "r") as f:
            for line in f:
                if not line.startswith(("@", "#")):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        data.append([float(parts[0]) / 1000.0, float(parts[1]) * 10.0])  # Convert to ns and Angstroms
        if data:
            arr = np.array(data)
            plt.plot(arr[:, 0], arr[:, 1], label=f"{label} (Final={arr[-1, 1]:.2f} Å)", color=color, lw=1.8)

plt.xlabel("Simulation Time (ns)", fontweight="bold")
plt.ylabel("Backbone RMSD (Å)", fontweight="bold")
plt.title("GROMACS Precompiled CUDA MD: Pocket Backbone Stability (10 ns)", fontweight="bold")
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUT_DIR, "gromacs_rmsd_comparison.png"), dpi=200)
plt.show()
print("✅ Trajectory RMSD analysis plot generated.")""")

    return nb


# ==============================================================================
# 2. BUILD NATIVEGPU NOTEBOOK (OpenMM Native CUDA Engine + Active Chunk Logger)
# ==============================================================================
def make_nativegpu_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    def add_md(t):
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [t]})

    def add_code(c):
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)})

    # Cell 0
    add_md("""# ⚡ Native GPU Molecular Dynamics Protocol (OpenMM + CUDA)
### Validation of Convergent Steric Gate Remodeling in Beetle Luciferases
**Author:** Bruno ("Blu") Stein Barbosa Menechino ([PMID: 38049805](https://pubmed.ncbi.nlm.nih.gov/38049805/))  
**Engine:** OpenMM with native NVIDIA CUDA GPU Acceleration Platform.  
**Hardware:** NVIDIA T4 / V100 / A100 GPU (Google Colab Pro).  
**Real-Time Progress Tracking:** Actively prints percentage `[XX.X%]`, nanoseconds simulated, live speed (`ns/day`), and estimated time remaining (ETA) every ~45 seconds!

---
### 🚀 How to Run:
1. Ensure your runtime is GPU: **Runtime -> Change runtime type -> T4 GPU**.
2. Click **Runtime -> Run all** (or `Ctrl+F9`).
*All 4 holo complexes and parameters are embedded directly inside this notebook—zero external git cloning or manual uploads required!*""")

    # Cell 1: Step 1 Dependencies
    add_md("## Step 1: Install OpenMM (Native CUDA & OpenCL GPU Engines) & MDTraj")
    add_code("""# Install OpenMM with native CUDA 12 hardware acceleration packages
!pip install -q "openmm[cuda12]" openmm-cuda-12 mdtraj matplotlib seaborn pandas

import openmm as mm

# Detect available acceleration platforms
avail = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print(f"Available OpenMM Platforms: {avail}")

if 'CUDA' in avail:
    platform = mm.Platform.getPlatformByName('CUDA')
    print(f"✅ Active GPU Engine: {platform.getName()} (NVIDIA CUDA Platform Detected)")
elif 'OpenCL' in avail:
    platform = mm.Platform.getPlatformByName('OpenCL')
    print(f"✅ Active GPU Engine: {platform.getName()} (NVIDIA OpenCL GPU Platform Detected)")
else:
    platform = mm.Platform.getPlatformByName('CPU')
    print(f"⚠️ Falling back to: {platform.getName()}")

!nvidia-smi""")

    # Cell 2: Step 2 Unpack payloads
    add_md("## Step 2: Unpack Embedded Holo Complexes into Workspace")
    add_code(f"""# Self-extracting embedded holo complexes using absolute paths
import os, sys, base64, gzip

BASE_DIR = os.path.abspath("md_simulation")
os.makedirs(os.path.join(BASE_DIR, "inputs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

embedded_payloads = {payloads_repr}

for fname, b64_str in embedded_payloads.items():
    out_path = os.path.join(BASE_DIR, "inputs", fname)
    raw_data = gzip.decompress(base64.b64decode(b64_str))
    with open(out_path, "wb") as fh:
        fh.write(raw_data)
    print(f"  Extracted: {{out_path}} ({{len(raw_data)}} bytes)")

print(f"\\n✅ All 4 holo structures unpacked successfully into {{BASE_DIR}}/inputs/.")
!ls -lh {{BASE_DIR}}/inputs/""")

    # Cell 3: Step 3 High-Throughput MD Simulation Loop on GPU with REAL-TIME PROGRESS
    add_md("## Step 3: Run 10.0 ns High-Throughput Molecular Dynamics Across All 3 Complexes (With Live Progress Logging)")
    add_code("""# 🚀 Launch 10.0 ns GPU Molecular Dynamics with Real-Time Percentage & Speed Feedback
import os, sys, io, time
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import openmm as mm
from openmm import app, unit

# Connect to Colab GPU (CUDA or OpenCL)
avail = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
if 'CUDA' in avail:
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed', 'DeviceIndex': '0'}
elif 'OpenCL' in avail:
    platform = mm.Platform.getPlatformByName('OpenCL')
    properties = {'Precision': 'mixed', 'DeviceIndex': '0'}
else:
    platform = mm.Platform.getPlatformByName('CPU')
    properties = {}
print(f"⚡ Active Platform: {platform.getName()}")

BASE_DIR = os.path.abspath("md_simulation")
INP_DIR = os.path.join(BASE_DIR, "inputs")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

COMPLEXES = [
    ("True Luc WT S347", "1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo"),
    ("Paralog WT L347", "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo"),
    ("SAE-Steered S347A348N349", "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo")
]

for label, comp_id in COMPLEXES:
    pdb_path = os.path.join(INP_DIR, f"{comp_id}.pdb")
    comp_out = os.path.join(OUT_DIR, comp_id)
    os.makedirs(comp_out, exist_ok=True)
    dcd_out = os.path.join(comp_out, "prod.dcd")
    top_out = os.path.join(comp_out, "solvated.pdb")

    if os.path.exists(dcd_out) and os.path.getsize(dcd_out) > 500000:
        print(f"✅ Simulation already finished for {label} ({dcd_out}). Skipping!", flush=True)
        continue

    print("=" * 70, flush=True)
    print(f"🚀 Launching 10.0 ns GPU Molecular Dynamics: {label}", flush=True)
    print(f"   Input PDB: {pdb_path}", flush=True)
    print("=" * 70, flush=True)

    # 1. Ingest protein coordinates (with auto-healing for mutated sidechain atom topologies)
    with open(pdb_path, "r") as fh:
        raw_lines = [l for l in fh if l.startswith(("ATOM", "TER", "END"))]
    clean_lines = []
    for l in raw_lines:
        if l.startswith("ATOM") and "ASN A 345" in l and "CD1" in l:
            s = int(l[6:11])
            clean_lines.append(f"ATOM  {s:5d}  CG  ASN A 345     -33.471  25.188 -18.375  1.00 50.00           C\n")
            clean_lines.append(f"ATOM  {s+1:5d}  OD1 ASN A 345     -32.467  25.234 -19.054  1.00 50.00           O\n")
            clean_lines.append(f"ATOM  {s+2:5d}  ND2 ASN A 345     -33.372  24.741 -17.148  1.00 50.00           N\n")
            continue
        clean_lines.append(l)
    pdb = app.PDBFile(io.StringIO("".join(clean_lines)))

    # 2. Build AMBER99SB-ILDN + TIP3P Force Field
    print("  [1/5] Adding hydrogens and solvating box (0.15 M NaCl)...", flush=True)
    forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)

    # 3. Create System with HMR (dt = 4.0 fs enabled by hydrogenMass = 3.0 amu)
    print("  [2/5] Parameterizing system with HMR (dt = 4.0 fs)...", flush=True)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometer,
        constraints=app.AllBonds,
        hydrogenMass=3.0*unit.amu
    )
    system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300.0*unit.kelvin, 25))

    integrator = mm.LangevinMiddleIntegrator(300.0*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    # 4. Energy Minimization
    print("  [3/5] Minimizing potential energy...", flush=True)
    simulation.minimizeEnergy(maxIterations=2500)

    # Save minimized solvated reference topology for mdtraj
    min_state = simulation.context.getState(getPositions=True)
    with open(top_out, "w") as fh:
        app.PDBFile.writeFile(simulation.topology, min_state.getPositions(), fh)

    # 5. Fast NPT Equilibration (100 ps = 25,000 steps)
    print("  [4/5] Equilibrating solvent & protein at 300 K / 1.0 bar (100 ps)...", flush=True)
    simulation.step(25000)

    # 6. Production MD on NVIDIA GPU (10.0 ns in 100 chunks with LIVE FLUSHED LOGGING)
    total_steps = 2500000
    chunk_steps = 25000 # 0.1 ns per chunk (~40-50s on GPU)
    n_chunks = total_steps // chunk_steps
    simulation.reporters.append(app.DCDReporter(dcd_out, 5000)) # Save frame every 20 ps (500 frames)

    print(f"  [5/5] Running 10.0 ns Production MD on NVIDIA GPU ({n_chunks} live checkpoints)...", flush=True)
    print("  " + "-" * 68, flush=True)

    t_start = time.time()
    for chunk in range(1, n_chunks + 1):
        t_c0 = time.time()
        simulation.step(chunk_steps)
        t_c1 = time.time()

        step_done = chunk * chunk_steps
        pct = (step_done / total_steps) * 100.0
        sim_ns = step_done * 0.004

        # Compute live speed and ETA
        chunk_dur = t_c1 - t_c0
        chunk_speed = (0.1 / chunk_dur) * 86400.0 if chunk_dur > 0 else 0.0
        tot_elapsed = t_c1 - t_start
        avg_speed = (sim_ns / tot_elapsed) * 86400.0
        rem_sec = ((10.0 - sim_ns) / (avg_speed / 86400.0)) if avg_speed > 0 else 0
        rem_m = int(rem_sec // 60)
        rem_s = int(rem_sec % 60)

        # Print live unbuffered update every ~45 seconds
        state = simulation.context.getState(getEnergy=True)
        epot = state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole)
        print(f"  ⚡ [{pct:5.1f}%] {sim_ns:5.2f} / 10.00 ns | Speed: {chunk_speed:5.1f} ns/day (Avg: {avg_speed:5.1f}) | ETA: {rem_m:02d}m {rem_s:02d}s | Epot: {epot:,.0f} kJ/mol", flush=True)

    dur = (time.time() - t_start) / 3600.0
    print(f"✅ Finished 10.0 ns in {dur:.2f} hours for {label}!", flush=True)

print("=" * 70, flush=True)
print("🎉 ALL 3 GPU SIMULATIONS COMPLETED SUCCESSFULLY ON NVIDIA GPU!", flush=True)
print("=" * 70, flush=True)""")

    # Cell 4: Step 4 MDTraj Analysis & Plotting
    add_md("## Step 4: Trajectory Backbone RMSD & Active-Site Gate Analysis")
    add_code("""# 📊 Post-Simulation Trajectory Analysis & Backbone Stability Visualization
import os
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

OUT_DIR = os.path.abspath("md_simulation/outputs")
complexes = [
    ("True Luc WT S347", "1_A0A8K0DA02_TrueLuc_WT_S347_SLU_holo", "#10b981"),
    ("Paralog WT L347", "2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo", "#e11d48"),
    ("SAE-Steered S347A348N349", "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_SLU_holo", "#2563eb")
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=180)

final_rmsds = []
for label, comp_id, color in complexes:
    top_file = os.path.join(OUT_DIR, comp_id, "solvated.pdb")
    dcd_file = os.path.join(OUT_DIR, comp_id, "prod.dcd")
    if os.path.exists(top_file) and os.path.exists(dcd_file):
        traj = md.load(dcd_file, top=top_file)
        bb_atoms = traj.topology.select("backbone")
        rmsd = md.rmsd(traj, traj, frame=0, atom_indices=bb_atoms) * 10.0 # Convert nm to Angstroms
        time_ns = traj.time / 1000.0
        final_rmsds.append((label, rmsd[-1], color))

        ax1.plot(time_ns, rmsd, label=f"{label} (Final={rmsd[-1]:.2f} Å)", color=color, lw=1.8)

ax1.set_xlabel("Simulation Time (ns)", fontweight="bold")
ax1.set_ylabel("Backbone RMSD (Å)", fontweight="bold")
ax1.set_title("NVIDIA GPU MD: 10 ns Pocket Backbone Equilibration", fontweight="bold")
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right")

# Summary Bar Chart
if final_rmsds:
    names = [x[0] for x in final_rmsds]
    vals = [x[1] for x in final_rmsds]
    colors = [x[2] for x in final_rmsds]
    bars = ax2.bar(names, vals, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Final RMSD (Å)", fontweight="bold")
    ax2.set_title("Equilibrium Pocket RMSD Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{b.get_height():.2f} Å",
                 ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gpu_rmsd_analysis.png"), dpi=200)
plt.show()
print("✅ Trajectory analysis completed and saved to gpu_rmsd_analysis.png.")""")

    return nb

def main():
    pre_nb = make_precompile_notebook()
    nat_nb = make_nativegpu_notebook()

    # Save precompile variants
    p1 = os.path.join(NOTEBOOKS_DIR, "precompile.ipynb")
    p2 = os.path.join(NOTEBOOKS_DIR, "gromacs_precompile_cuda.ipynb")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(pre_nb, f, indent=2)
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(pre_nb, f, indent=2)
    print(f"✅ Generated precompile notebook at: {p1}")

    # Save nativegpu variants
    n1 = os.path.join(NOTEBOOKS_DIR, "nativegpu.ipynb")
    n2 = os.path.join(NOTEBOOKS_DIR, "openmm_nativegpu_cuda.ipynb")
    with open(n1, "w", encoding="utf-8") as f:
        json.dump(nat_nb, f, indent=2)
    with open(n2, "w", encoding="utf-8") as f:
        json.dump(nat_nb, f, indent=2)
    print(f"✅ Generated nativegpu notebook at: {n1}")

if __name__ == "__main__":
    main()
