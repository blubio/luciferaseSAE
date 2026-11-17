"""Generate standalone 140 Mya Ancestral MD simulation notebook for Google Colab."""
import os
import base64
import gzip
import json

ROOT = r"c:\Users\pagli\Desktop\translator"
pdb_path = os.path.join(ROOT, "results", "md_inputs", "5_Anc0_140Mya_Extinct_Ancestor_SLU_holo.pdb")
with open(pdb_path, "rb") as f:
    raw = f.read()

b64_str = base64.b64encode(gzip.compress(raw)).decode("utf-8")
print(f"PDB compressed: {len(raw)} -> {len(b64_str)} b64 chars")

nb_path = os.path.join(ROOT, "notebooks", "run_140mya_ancestral_md.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ 140 Mya Cretaceous Extinct Ancestor (10.0 ns All-Atom Explicit Solvent MD)\n",
            "### Paleogenomic Neofunctionalization & Physical Gate Stability\n",
            "**Author:** Bruno ('Blu') Stein Barbosa Menechino ([PMID: 38049805](https://pubmed.ncbi.nlm.nih.gov/38049805/))  \n",
            "**Engine:** OpenMM with Native NVIDIA CUDA GPU Platform  \n",
            "**Hardware:** Google Colab T4 GPU  \n",
            "**Protocol:** 10.0 ns Production MD in explicit TIP3P solvent (0.15 M NaCl) with 4.0 fs HMR timestep, live speed & ETA reporting, backbone RMSD analysis, and automated ZIP packaging.\n",
            "\n",
            "---\n",
            "### 🚀 Quick Start:\n",
            "1. Ensure runtime is GPU: **Runtime -> Change runtime type -> T4 GPU**.\n",
            "2. Click **Runtime -> Run all** (`Ctrl+F9`).\n",
            "*The 140 Mya ancestral coordinates are embedded directly in this notebook!*"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 1: Install OpenMM (CUDA 12) & MDTraj"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            '!pip install -q "openmm[cuda12]" openmm-cuda-12 mdtraj matplotlib seaborn pandas scipy\n',
            '\n',
            'import openmm as mm\n',
            'avail = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]\n',
            'print(f"Available Platforms: {avail}")\n',
            'if "CUDA" in avail:\n',
            '    print("✅ Active GPU Engine: CUDA (NVIDIA T4 detected)")\n',
            'else:\n',
            '    print(f"⚠️ Falling back to: {avail[0]}")\n',
            '!nvidia-smi\n'
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 2: Unpack Embedded 140 Mya Extinct Ancestor Coordinates"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'import os, sys, base64, gzip\n',
            '\n',
            'BASE_DIR = os.path.abspath("ancestral_md")\n',
            'INP_DIR = os.path.join(BASE_DIR, "inputs")\n',
            'OUT_DIR = os.path.join(BASE_DIR, "outputs")\n',
            'os.makedirs(INP_DIR, exist_ok=True)\n',
            'os.makedirs(OUT_DIR, exist_ok=True)\n',
            '\n',
            f'pdb_b64 = "{b64_str}"\n',
            'raw = gzip.decompress(base64.b64decode(pdb_b64))\n',
            'pdb_path = os.path.join(INP_DIR, "5_Anc0_140Mya_Extinct_Ancestor_SLU_holo.pdb")\n',
            'with open(pdb_path, "wb") as f:\n',
            '    f.write(raw)\n',
            'print(f"✅ Extracted: {pdb_path} ({len(raw)} bytes)")\n'
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 3: Run 10.0 ns Explicit-Solvent GPU MD (4.0 fs HMR + Live Logging)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'import os, sys, io, time\n',
            'import numpy as np\n',
            'import mdtraj as md\n',
            'import openmm as mm\n',
            'from openmm import app, unit\n',
            '\n',
            'BASE_DIR = os.path.abspath("ancestral_md")\n',
            'INP_DIR = os.path.join(BASE_DIR, "inputs")\n',
            'OUT_DIR = os.path.join(BASE_DIR, "outputs")\n',
            'anc_dir = os.path.join(OUT_DIR, "5_Anc0_140Mya_Extinct_Ancestor")\n',
            'os.makedirs(anc_dir, exist_ok=True)\n',
            '\n',
            'dcd_path = os.path.join(anc_dir, "prod.dcd")\n',
            'top_path = os.path.join(anc_dir, "solvated.pdb")\n',
            'log_path = os.path.join(anc_dir, "sim_progress.csv")\n',
            '\n',
            'if os.path.exists(dcd_path) and os.path.getsize(dcd_path) > 500000:\n',
            '    print(f"✅ Ancestral simulation already complete ({dcd_path}). Skipping!")\n',
            'else:\n',
            '    print("=" * 70, flush=True)\n',
            '    print("🚀 Launching 10.0 ns All-Atom GPU MD on 140 Mya Cretaceous Ancestor", flush=True)\n',
            '    print("=" * 70, flush=True)\n',
            '\n',
            '    pdb_in = os.path.join(INP_DIR, "5_Anc0_140Mya_Extinct_Ancestor_SLU_holo.pdb")\n',
            '\n',
            '    # 1. Ingest protein coordinates (clean protein ATOM lines to isolate APO tertiary fold)\n',
            '    with open(pdb_in, "r") as fh:\n',
            '        clean_lines = [l for l in fh if l.startswith(("ATOM", "TER", "END"))]\n',
            '    pdb = app.PDBFile(io.StringIO("".join(clean_lines)))\n',
            '\n',
            '    # 2. Build AMBER99SB-ILDN + TIP3P Force Field\n',
            '    print("  [1/5] Adding hydrogens and solvating box (0.15 M NaCl)...", flush=True)\n',
            '    forcefield = app.ForceField("amber99sbildn.xml", "tip3p.xml")\n',
            '    modeller = app.Modeller(pdb.topology, pdb.positions)\n',
            '    modeller.addHydrogens(forcefield)\n',
            '    modeller.addSolvent(forcefield, model="tip3p", padding=1.0*unit.nanometer, ionicStrength=0.15*unit.molar)\n',
            '\n',
            '    # 3. Create System with HMR (dt = 4.0 fs enabled by hydrogenMass = 3.0 amu)\n',
            '    print("  [2/5] Parameterizing system with HMR (dt = 4.0 fs)...", flush=True)\n',
            '    system = forcefield.createSystem(\n',
            '        modeller.topology,\n',
            '        nonbondedMethod=app.PME,\n',
            '        nonbondedCutoff=1.0*unit.nanometer,\n',
            '        constraints=app.AllBonds,\n',
            '        hydrogenMass=3.0*unit.amu\n',
            '    )\n',
            '    system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, 300.0*unit.kelvin, 25))\n',
            '\n',
            '    integrator = mm.LangevinMiddleIntegrator(300.0*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds)\n',
            '    platform = mm.Platform.getPlatformByName("CUDA")\n',
            '    properties = {"Precision": "mixed", "DeviceIndex": "0"}\n',
            '\n',
            '    simulation = app.Simulation(modeller.topology, system, integrator, platform, properties)\n',
            '    simulation.context.setPositions(modeller.positions)\n',
            '\n',
            '    # 4. Energy Minimization (5,000 steps)\n',
            '    print("  [3/5] Minimizing potential energy...", flush=True)\n',
            '    simulation.minimizeEnergy(maxIterations=5000)\n',
            '\n',
            '    # Save minimized solvated reference topology for mdtraj\n',
            '    min_state = simulation.context.getState(getPositions=True)\n',
            '    with open(top_path, "w") as fh:\n',
            '        app.PDBFile.writeFile(simulation.topology, min_state.getPositions(), fh)\n',
            '    print(f"  Solvated system: {simulation.topology.getNumAtoms()} atoms ({simulation.topology.getNumResidues()} residues)", flush=True)\n',
            '\n',
            '    # 5. Fast NPT Equilibration (100 ps = 25,000 steps)\n',
            '    print("  [4/5] Equilibrating solvent & protein at 300 K / 1.0 bar (100 ps)...", flush=True)\n',
            '    simulation.step(25000)\n',
            '\n',
            '    # 6. Production MD on NVIDIA GPU (10.0 ns in 100 chunks with LIVE FLUSHED LOGGING)\n',
            '    total_steps = 2500000\n',
            '    chunk_steps = 25000 # 0.1 ns per chunk (~40-50s on T4 GPU)\n',
            '    n_chunks = total_steps // chunk_steps\n',
            '    simulation.reporters.append(app.DCDReporter(dcd_path, 5000)) # Save frame every 20 ps (500 frames)\n',
            '\n',
            '    print(f"  [5/5] Running 10.0 ns Production MD on NVIDIA GPU ({n_chunks} live checkpoints)...", flush=True)\n',
            '    print("  " + "-" * 68, flush=True)\n',
            '\n',
            '    with open(log_path, "w") as log_fh:\n',
            '        log_fh.write("step,time_ns,speed_ns_per_day,epot_kj\\n")\n',
            '\n',
            '    t_start = time.time()\n',
            '    for chunk in range(1, n_chunks + 1):\n',
            '        t_c0 = time.time()\n',
            '        simulation.step(chunk_steps)\n',
            '        t_c1 = time.time()\n',
            '\n',
            '        step_done = chunk * chunk_steps\n',
            '        pct = (step_done / total_steps) * 100.0\n',
            '        sim_ns = step_done * 0.004\n',
            '\n',
            '        chunk_dur = t_c1 - t_c0\n',
            '        chunk_speed = (0.1 / chunk_dur) * 86400.0 if chunk_dur > 0 else 0.0\n',
            '        tot_elapsed = t_c1 - t_start\n',
            '        avg_speed = (sim_ns / tot_elapsed) * 86400.0\n',
            '        rem_sec = ((10.0 - sim_ns) / (avg_speed / 86400.0)) if avg_speed > 0 else 0\n',
            '        rem_m = int(rem_sec // 60)\n',
            '        rem_s = int(rem_sec % 60)\n',
            '\n',
            '        state = simulation.context.getState(getEnergy=True)\n',
            '        epot = state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole)\n',
            '        print(f"  ⚡ [{pct:5.1f}%] {sim_ns:5.2f} / 10.00 ns | Speed: {chunk_speed:5.1f} ns/day (Avg: {avg_speed:5.1f}) | ETA: {rem_m:02d}m {rem_s:02d}s | Epot: {epot:,.0f} kJ/mol", flush=True)\n',
            '\n',
            '        with open(log_path, "a") as log_fh:\n',
            '            log_fh.write(f"{step_done},{sim_ns:.4f},{avg_speed:.2f},{epot:.1f}\\n")\n',
            '\n',
            '    dur = (time.time() - t_start) / 3600.0\n',
            '    print(f"\\n✅ Finished 10.0 ns in {dur:.2f} hours for 140 Mya Extinct Ancestor!", flush=True)\n'
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 4: Trajectory Backbone Stability & Catalytic Gate Loop Analysis"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'import os\n',
            'import matplotlib.pyplot as plt\n',
            'import mdtraj as md\n',
            'import numpy as np\n',
            '\n',
            'BASE_DIR = os.path.abspath("ancestral_md")\n',
            'anc_dir = os.path.join(BASE_DIR, "outputs", "5_Anc0_140Mya_Extinct_Ancestor")\n',
            'top_file = os.path.join(anc_dir, "solvated.pdb")\n',
            'dcd_file = os.path.join(anc_dir, "prod.dcd")\n',
            '\n',
            'if os.path.exists(top_file) and os.path.exists(dcd_file):\n',
            '    traj = md.load(dcd_file, top=top_file)\n',
            '    bb_atoms = traj.topology.select("backbone")\n',
            '    rmsd = md.rmsd(traj, traj, frame=0, atom_indices=bb_atoms) * 10.0 # nm to Angstroms\n',
            '    time_ns = traj.time / 1000.0\n',
            '\n',
            '    loop_atoms = traj.topology.select("resid 343 to 350 and backbone")\n',
            '    if len(loop_atoms) > 0:\n',
            '        loop_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=loop_atoms) * 10.0\n',
            '    else:\n',
            '        loop_rmsd = np.zeros_like(rmsd)\n',
            '\n',
            '    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=180)\n',
            '\n',
            '    # Panel 1: Full Backbone RMSD\n',
            '    ax1.plot(time_ns, rmsd, color="#d97706", lw=2.0, label=f"140 Mya Ancestor Backbone (Final={rmsd[-1]:.2f} Å)")\n',
            '    ax1.set_xlabel("Simulation Time (ns)", fontweight="bold")\n',
            '    ax1.set_ylabel("Backbone RMSD (Å)", fontweight="bold")\n',
            '    ax1.set_title("140 Mya Cretaceous Extinct Ancestor: 10 ns Stability", fontweight="bold")\n',
            '    ax1.grid(alpha=0.3)\n',
            '    ax1.legend(loc="lower right")\n',
            '\n',
            '    # Panel 2: Catalytic Gate Loop (Residues 343-350)\n',
            '    ax2.plot(time_ns, loop_rmsd, color="#dc2626", lw=2.0, label=f"Gate Loop 343-350 (Final={loop_rmsd[-1]:.2f} Å)")\n',
            '    ax2.set_xlabel("Simulation Time (ns)", fontweight="bold")\n',
            '    ax2.set_ylabel("Loop RMSD (Å)", fontweight="bold")\n',
            '    ax2.set_title("Catalytic Gate Loop Flexibility (L347 Plug)", fontweight="bold")\n',
            '    ax2.grid(alpha=0.3)\n',
            '    ax2.legend(loc="lower right")\n',
            '\n',
            '    plt.tight_layout()\n',
            '    analysis_img = os.path.join(anc_dir, "ancestral_10ns_analysis.png")\n',
            '    plt.savefig(analysis_img, dpi=200)\n',
            '    plt.show()\n',
            '    print(f"✅ Trajectory analysis saved to: {analysis_img}")\n'
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 5: Package Trajectory and Artifacts into Zip for Direct Download"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            'import os, zipfile\n',
            'from google.colab import files\n',
            '\n',
            'BASE_DIR = os.path.abspath("ancestral_md")\n',
            'anc_dir = os.path.join(BASE_DIR, "outputs", "5_Anc0_140Mya_Extinct_Ancestor")\n',
            'zip_path = os.path.join(BASE_DIR, "outputs", "140Mya_Ancestral_MD_10ns.zip")\n',
            '\n',
            'with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:\n',
            '    for root, dirs, filenames in os.walk(anc_dir):\n',
            '        for f in filenames:\n',
            '            fp = os.path.join(root, f)\n',
            '            arcname = os.path.relpath(fp, anc_dir)\n',
            '            zipf.write(fp, arcname)\n',
            '\n',
            'print(f"✅ Created ZIP Archive: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")\n',
            'print("🚀 Triggering browser download...")\n',
            'files.download(zip_path)\n'
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": [],
            "toc_visible": True
        },
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Saved clean notebook to {nb_path}")
