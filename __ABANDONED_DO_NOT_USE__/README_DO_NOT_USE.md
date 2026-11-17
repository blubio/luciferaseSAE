# ⚠️ ABANDONED / SUPERSEDED FILES — DO NOT USE

Everything inside this directory has been quarantined and **must NOT be used** in the manuscript, downstream analyses, or audits as active project results.

## Reasons for Quarantine

0. **RETRACTED 2026-11-13 after adversarial audit (`AUDIT_translator.md`).** Three result families were withdrawn. None of them may be cited, plotted, or referenced in the manuscript.

   **0a. Fake ancestral reconstruction (`scripts/run_ancestral_reconstruction_and_ignition.py`, `scripts/build_140mya_ancestral_notebook.py`, `notebooks/run_140mya_ancestral_md.ipynb`, `results/figures/fig13_ancestral_ignition_140mya.png`, `results/md_inputs/5_Anc0_140Mya_Extinct_Ancestor_SLU_holo.pdb`)**
   - The reconstructed "Anc_0 140 Mya Cretaceous ancestor" is **100.00% identical over 542 aa to the extant *Ignelater* paralog `A0A8K0C7P5`**. It is a living protein, not an ancestor.
   - The `5_Anc0_*.pdb` coordinates are **byte-identical** to `2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb` (atom-record MD5 `b681f6cc8d8d`, 4299 atoms). No residue was ever threaded or mutated.
   - Root cause: `KEY_SPECIES` lists 3 non-luminous taxa but `P0C027` is absent from `dataset.tsv`, leaving n=2; `Counter.most_common(1)` resolves the 2-way tie to the first sequence, always `A0A8K0C7P5`.
   - `reconstruct_ancestral_nodes()` takes column consensus over **unaligned** sequences of differing length (542-550 aa) truncated to `min_len`. Columns past the first indel compare non-homologous residues. `Anc_int` is only 17.9-38.9% identical to any real member.
   - Consequence: in `fig13`, the "140 Mya Anc_0" node and the "Ignelater Decoy Paralog" node are **the same sequence plotted at two different ages**.
   - No IQ-TREE or FastML was ever run anywhere in this project. A real ASR requires MAFFT/MUSCLE -> IQ-TREE + ModelFinder + bootstraps -> marginal ancestral states with per-site posteriors.

   **0b. Toy "molecular dynamics" presented as all-atom MD (`scripts/run_active_site_md.py`, `results/figures/fig10_molecular_dynamics_mmpbsa.png`)**
   - Not molecular dynamics: **no bonded terms at all** (no bonds/angles/dihedrals); connectivity is replaced by a harmonic tether to the starting coordinates.
   - "Amber ff99SB / GAFF" is a 6-element vdW lookup table plus a 9-entry hand-written partial-charge dict with a +/-0.1-by-element fallback.
   - "MM/GBSA" / "MMPBSA" is a misnomer: there is no Generalized Born term, no surface-area term, no Poisson-Boltzmann solver.
   - **Dimensional error:** the integrator adds `F/m` (kcal/mol/A/amu) directly to a velocity in A/ps without the 418.4 conversion factor, under-weighting the physical force against the thermal noise by ~20.5x.
   - Also: mass = 12 amu for every atom including hydrogens and ligand; forces clamped to +/-50; `torch.manual_seed(42)` called *after* the initial velocities are drawn.
   - All numbers it produced are retracted, specifically `+3.5 kcal/mol`, `+0.4 kcal/mol`, `-8.5 kcal/mol`, and `108.1 -> 99.6 kcal/mol`.

   **0c. Circular opsin wavelength result.** The `MAE = 0.75 nm, r = 0.985` figure across 134 RH1 haplotypes is derived from **formula-estimated** lambda_max values, which are themselves a deterministic function of the tuning sites being used as predictors. It is also below the ~1-2 nm reproducibility of microspectrophotometry. It is superseded by the de-circularized experimental set (n = 41, MAE = 5.20 nm) and must not be reported alongside it.

   **Note on the 10 ns trajectories (NOT quarantined, but relabelled):** `md_trajectories_10ns/` holds three genuine OpenMM runs (2.5M steps x 4 fs = 10.0 ns each, verified from DCD headers). They are **apo** - `openmm_nativegpu_cuda.ipynb` filters `HETATM` at read time, so SLU was never present. The directories were renamed `*_apo` on 2026-11-13. They are a backbone-stability control, not a ligand-gate experiment.

1. **Abandoned 50-Superfamily Atlas Scope-Creep (`data/superfamilies_50/`, `scripts/*50_superfam*`, `strict_uniprot_validator_and_fetcher.py`, `audit_superfamily_headers.py`, `results/empirical_50_superfamilies_results.json`, `fig12_*.png`, `fig14_*.png`)**
   - Diverged from the core convergent-evolution hypothesis (Lineage A -> Lineage B cross-clade generalization across Beetles, Deep-Sea/Cavefish Opsins, and Polar Fish AFGPs).
   - Contained accession mismatches in early drafts (`P0C027`, `P04637`, etc.) and uncontrolled cross-kingdom/cross-phylum pairings (e.g., Firefly vs. Rat; Human vs. Barley; Horse vs. Human orthologs yielding `1.000`).

2. **Abandoned Stage-2 Placeholder FEP Sprint (`notebooks/sprint_stage2_ancestral_md_and_fep.ipynb`, `scripts/generate_sprint_stage2_notebook.py`, `scripts/prepare_fast_switching_fep_pipeline.py`, `results/fep_pipeline/`)**
   - Step 3 crashed on unparameterized `HETATM` `SLU` residue.
   - Step 4 used synthetic `np.random.normal(...)` values rather than actual OpenMM alchemical FEP trajectories.

3. **Duplicate / Superseded Draft Notebooks (`notebooks/nativegpu.ipynb`, `notebooks/precompile.ipynb`, `notebooks/gromacs_precompile_cuda.ipynb`, `scripts/build_dual_notebooks.py`)**
   - Identical duplicate copies or superseded GROMACS Colab drafts replaced by `notebooks/openmm_nativegpu_cuda.ipynb`.
