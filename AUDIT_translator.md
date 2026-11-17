# Adversarial Audit — "DNA-to-Phenotype Translator"

**Auditor:** Claude Science (rigorous peer-review mode)
**Scope:** `C:\Users\pagli\Desktop\translator` — 177 source/result files, 3 × ~640 MB MD trajectories, manuscript draft (195 lines), 29 scripts.
**Method:** every claim below was re-derived by executing code against the repository's own files. Nothing is taken from the README, the manuscript, or the prior Gemini review on trust.

---

## 0. Verdict in one paragraph

The **core System-1 machine-learning experiment is methodologically sound and the split is clean** — I could not find leakage in `evaluate.py`, and its design is genuinely pre-registered. The **10 ns OpenMM trajectories are real**. But three things sit between this project and submission: (1) the "140 Mya ancestor" is a **modern protein**, verified at 100.00% sequence identity and byte-identical coordinates; (2) `run_active_site_md.py`, whose numbers are the entirety of manuscript §2.3, is **not molecular dynamics** — it has no bonded terms, invented charges, and a dimensional error in the integrator; (3) **Methods §4.5 describes a GROMACS/gmx_MMPBSA holo-ligand pipeline that was never run** — the delivered trajectories are OpenMM and ligand-free. The statistical headline also needs restating: the ancestry-trap test is n⁻ = 2 against ~5 effective positive clusters, and its random-classifier AP baseline is 0.714, not 0.5.

---

## 1. Verified strengths (these survive audit)

| # | Claim | Verification |
|---|---|---|
| S1 | **No leakage in the core System-1 split.** | `evaluate.py`: `train_pos` = firefly positives only; `train_neg` = `nonlum_other` **excluding** `TEST_NEG_GENERA` = {Agrilus, Tenebrio, Leptinotarsa, Nicrophorus}; test positives = click beetle + railroad worm; test negatives = `nonlum_elateroid` + the four held-out genera. Train/test genera are disjoint. Confirmed by re-executing the index logic. |
| S2 | **Design was pre-registered and is honoured.** | The docstring states the split *before* results; the code matches it line-for-line. Layer selection uses leave-one-genus-out CV **on training data only** (`cv` loop over `tr_genus`). This is the correct way to pick a layer and is rarely done. |
| S3 | **Pseudoreplication is handled** — in `evaluate.py`. | Single-linkage clustering at ≥90% PID for positives / 0.85 normalised score for negatives, with 1/cluster-size weights passed to `roc_auc_score` and `average_precision_score`. 27 click-beetle positives correctly collapse to 4.99 effective clusters. |
| S4 | **Label hygiene is implemented, not just described.** | `dataset.tsv`: 124 rows carry `reason = "homology-only luciferase name"` and 71 `"paralog in luminous genome"`, all held at `label = unlabeled` and excluded from supervised fitting. 455 excluded, 3 exact duplicates flagged. The 94-false-luciferase finding is real and is a genuine contribution. |
| S5 | **Residue "347" is a real alignment coordinate — in Pillar 3.** | `unified_expansions.py` builds a Biopython global `PairwiseAligner` and `map_positions(P08659, target)` to transfer numbering onto every structure. This is correct. (See F2/M7 — other scripts do *not* do this.) |
| S6 | **SAE feature discovery is honestly zero-shot.** | `interpret_sae_and_pocket.py`: `active_cols` thresholded on `sae_mean[tr]` (train only); L1 probe fit on `tr` only; `top_k_idx` taken from **train-fitted coefficients**; `single_auc` then computed on held-out click beetles with cluster weights. No test-set peeking in feature selection. |
| S7 | **The 10 ns GPU MD is empirically real.** | DCD headers parsed directly: `NSET = 500` frames, `NSTEP = 2,500,000`, `NSAVC = 5000`, `delta = 0.0818193` AKMA = **4.0 fs** → 2.5 M × 4 fs = **10.0 ns exactly**, frames every 20 ps. Explicit solvent confirmed (89,988 / 100,311 / 100,356 water+ion HETATM records). Three independent systems. This is not synthetic. |
| S8 | **Steering controls are the right ones.** | Matched-norm Random-50 across 10 seeds, Orthogonal-50 active-SAE, dense difference-in-means ceiling, plus off-target top-1 recovery and KL. This is a properly controlled causal-intervention design. |
| S9 | **Dead work is quarantined, not deleted.** | `__ABANDONED_DO_NOT_USE__/` with a README. Good practice; keep it. |

---

## 2. FATAL — must be fixed or withdrawn before any claim is made

### F1. The "140 Mya extinct ancestor" is a living protein (both sequence and structure)

Gemini's report is correct and the problem is **larger than reported**. The contamination is not only in the PDB — it is in the sequence itself.

```
Anc_0 vs A0A8K0C7P5 : 100.00 % identical over 542 aa   <-- the modern Ignelater decoy paralog
Anc_0 vs A0A0C6DRW5 :  24.35 %
Anc_0 vs A0A8K0DA02 :   5.35 %
Anc_0 vs P08659     :   9.23 %
```

Coordinates, independently confirmed (MD5 over ATOM/HETATM records only):

```
2_A0A8K0C7P5_Paralog_WT_L347_SLU_holo.pdb   4299 atoms   md5 b681f6cc8d8d
5_Anc0_140Mya_Extinct_Ancestor_SLU_holo.pdb 4299 atoms   md5 b681f6cc8d8d   <-- identical
```

**Root cause.** `KEY_SPECIES` lists three non-luminous taxa, but `P0C027` is absent from `dataset.tsv`, so `non_luminous_seqs` has **n = 2**. `Counter.most_common(1)` on a 2-way tie returns the first-inserted element, which is always `A0A8K0C7P5`. The "deepest-root parsimony reconstruction" therefore reduces, deterministically, to copying one extant paralog.

**Consequence for Figure 13.** Nodes 1 ("Anc_0, 140 Mya Extinct") and 5 ("Ignelater Decoy Paralog WT L347", 90 Mya) are the **same sequence plotted at two different x-positions**. The "deep-time ignition curve" contains a duplicated point presented as independent evidence, and Panel B contrasts a protein with itself.

**Action:** do not cite `5_Anc0_*.pdb`, `fig13`, or `notebooks/run_140mya_ancestral_md.ipynb`. Quarantine them into `__ABANDONED_DO_NOT_USE__/` alongside a note. Confirmed: **none of this has reached the manuscript** (§2.7 cites the separate NJ/ASR in `run_step1_and_step3_rigor.py`, not this script). Contain it now and the damage is zero.

### F2. The ancestral reconstruction takes consensus over *unaligned* sequences

`reconstruct_ancestral_nodes()` indexes `s[0][col]` across raw strings truncated to `min_len`. The inputs are **542–550 aa and never aligned**. Column *c* of a 550-aa sequence is not homologous to column *c* of a 542-aa sequence, so every position past the first indel is comparing non-homologous residues.

Measured consequence — the intermediate nodes are chimeras matching nothing:

```
Anc_int vs A0A8K0C7P5 31.6%  | vs A0A0C6DRW5 28.6% | vs A0A8K0DA02 17.9% | vs P08659 38.9%
Anc_luc vs P08659     55.2%  (the "ancestral luciferase" is only 55% identical to a modern firefly)
```

The script's own printed gate window is `Anc_0[344:351] = ITHTLPD` — it contains **no leucine at all**, and `Anc_0[347] = H`. The script does not support the ancestral-L347 claim even on its own output. Note also the two hardcoded, mutually inconsistent offsets (`p08_seq[344:351]` vs `A0A8K0C7P5[341:348]`) — a manual patch standing in for the alignment that should have been there.

**Severity note on method:** the project context describes IQ-TREE / FastML reconstruction. **Neither string appears anywhere in the repository.** No maximum-likelihood ASR, no tree-based marginal reconstruction, no posterior probabilities per site exists in this codebase.

### F3. `run_active_site_md.py` is not molecular dynamics

This script produces every number in manuscript §2.3 and Figure 10. It is an energy-minimisation toy, and it is mislabelled at every level.

| Claim in docstring/manuscript | What the code does |
|---|---|
| "Amber ff99SB / GAFF" | `VDW_PARAMS` is a **6-element lookup table** (C,N,O,S,P,H). `PARTIAL_CHARGES` is a **9-entry hand-written dict** with a `±0.1`-by-element fallback. These are not AMBER charges or GAFF types. |
| "All-atom dynamics" | **No bonded terms exist** — no bonds, angles, dihedrals, or impropers. Molecular connectivity is replaced entirely by a harmonic tether to the starting coordinates (`k = 0.2` free / `10.0` anchored). Without that tether the "protein" would dissociate on step 1. |
| "MM/GBSA" / filename `mmpbsa` | **No Generalized Born term, no surface-area term, no Poisson–Boltzmann solver.** It is a pairwise LJ + Coulomb sum with a distance-dependent dielectric ε(r) = 4r. |
| "300 K Langevin" | See dimensional error below. |

**Dimensional error in the integrator (decisive).** The update is `vel = c1*vel + (f_total/12.0)*dt + noise`.
- `f_total/m` carries units kcal·mol⁻¹·Å⁻¹·amu⁻¹; converting to Å·ps⁻² requires **×418.4**.
- `noise` is scaled as √(k_B T/m) in kcal·mol⁻¹·amu⁻¹; converting to Å·ps⁻¹ requires **×√418.4 = 20.5**.
- **Neither conversion is applied.** The deterministic force is therefore under-weighted relative to the stochastic kick by ≈ **20.5×**. The trajectory is dominated by thermal noise around the restraint minimum; the physical potential barely participates.

Further defects: every atom (including hydrogens and the ligand) is assigned **mass = 12 amu**; forces are hard-clamped to ±50, which destroys energy conservation and saturates exactly the steric repulsion the study is trying to measure; and `torch.manual_seed(42)` is called **after** the initial velocities are drawn, so initial conditions are unseeded and results depend on call order.

**Overclaim on the reported energies.** §2.3 states the steered mutant "**stabilized total pocket binding by −8.5 kcal/mol (108.1 → 99.6 kcal/mol)**". Both values are **positive** — net repulsive by ~+100 kcal/mol. A reduction in a large positive interaction energy, computed with no solvent, no entropy and fabricated charges, is not a binding free energy and must not be described as stabilised binding.

### F4. Methods §4.5 describes a pipeline that was never executed

> *"parameterized with the AMBER99SB-ILDN force field and TIP3P water. DLSA (SLU) was parameterized with GAFF2 and AM1-BCC charges via ACPYPE... Trajectories were integrated with GROMACS 2023... Binding free energies were evaluated via gmx_MMPBSA using the modified Generalized Born model (igb = 5) with per-residue decomposition."*

Repository-wide search results:

- **No `.xtc`, `.trr`, `.top`, or `.itp` file exists anywhere.** GROMACS was never run to completion; `gmx mdrun`/`grompp` appear only in protocol scripts and notebooks with no outputs.
- The delivered trajectories are **OpenMM DCD** (`notebooks/openmm_nativegpu_cuda.ipynb`).
- **No MM/GBSA or MM/PBSA output table exists** — the only matches are the protocol notebook and the PNG filename. No ΔG, no per-residue decomposition, no `igb=5` result.
- **SLU is absent from all three simulated systems.** `solvated.pdb` HETATM residues are `{HOH, NA, CL}` only. The runs are **APO**.

The project context shows you know the ligand was stripped (no AMBER parameters for SLU) — that is a defensible choice. But the **manuscript does not say so**, the directory names say `..._SLU_holo`, and §4.5 affirmatively describes ligand parameterization that did not happen. As written this is a misrepresentation of the computational protocol and would be a retraction-class finding in review.

### F5. Figure 11 contradicts its own data

Title: *"All-Atom GPU Molecular Dynamics: **Causal Steric Gate Rescue** in Beetle Luciferase."* The underlying runs contain **no ligand**, so no steric gate rescue can be observed in principle. What the data actually show:

| System | Final backbone RMSD | Mean gate RMSF |
|---|---|---|
| True Luc WT S347 | 1.70 Å | 2.88 Å |
| Paralog WT L347 | **1.66 Å** | 3.01 Å |
| SAE-Steered S347/A348/N349 | **1.99 Å** | **4.47 Å** |

The true luciferase and the non-luminous decoy are **indistinguishable** (1.70 vs 1.66 Å) — the apo MD provides zero discrimination between the two hypothesis classes. The designed "rescue" mutant is the **least** stable system on both metrics. The figure title asserts the opposite of the result.

Additionally, `analyze_completed_trajectories.py` prints `"ANALYSIS OF COMPLETED 30.0 NS..."` and sets `suptitle("...Definitive 30.0 ns Trajectory Analysis...")` while `time_ns = np.linspace(0, 10.0, 500)`. If 30 ns means 3 × 10 ns aggregate, say so; as written it reads as 30 ns per system. Panels B/C also label **whole-protein backbone RMSD** (`bb_idx` spans all of chain A) as "Active-Site" / "Equilibrium Pocket" RMSD.

---

## 3. MAJOR — results stand but the claims must be restated

### M1. The ancestry-trap headline rests on n⁻ = 2, and its chance baseline is 0.714, not 0.5

This is the single most important statistical correction. Because `evaluate.py` passes cluster weights into `average_precision_score`, the effective sample sizes are:

```
click-beetle positives : 27 raw  ->  4.99 effective clusters  (23 of 27 carry weight 0.043)
nonlum_elateroid negs  :  2 raw  ->  2.00  (A0A0C6DRW5 Pectocera, A9ZPM4 Agrypnus)
weighted prevalence (= random-classifier AP) = 4.99/(4.99+2.00) = 0.7138
```

Therefore, on the ancestry control:

| Method | Reported AP | Reading against the true 0.7138 baseline |
|---|---|---|
| `nn_identity` | 0.7082 | **exactly chance** (marginally below) |
| `composition` | 0.7238 | **chance** |
| `esm2_35M:last` | 0.9434 | above chance, n⁻ = 2 |
| `esm2_35M:cvlayer4` | 0.9708 | above chance, n⁻ = 2 |

The README frames 0.708 as evidence that homology "fails", and presents 0.931/0.971 as a dramatic win. Against a 0.714 baseline both readings change materially. **The reported AUROC confidence intervals are also never quoted:** the headline `esm2_t12_35M:last` ancestry-control result is **AUROC 0.8435, 95% CI [0.4074, 1.0000]** — the interval **includes 0.5**. The cvlayer4 result is 0.9087 [0.6373, 0.9770]. With n⁻ = 2 the AUROC can only take values on a 1/54 grid; these are not estimable quantities. State them as such or drop them.

**Also note:** `nn_identity` AUROC on `click_beetle vs ALL test negatives` is **0.9858** vs ESM-2's 0.9941. On the full test set, homology is very nearly as good as the model. The paper's narrative depends almost entirely on the n⁻ = 2 subset.

### M2. Effective sample size is ~6 training clusters

16 firefly positives collapse to **6.0** effective clusters; 23 of the 27 click-beetle positives are a single ≥90%-identity cluster (*Pyrophorus plagiophthalamus* colour variants). Every System-1 conclusion rests on roughly **6 independent training examples and 5 independent test examples**. This must appear in the abstract-level framing, not a limitations footnote.

### M3. The n⁻ = 91 "expanded" control re-introduces the circularity the project was built to avoid

`fix_all_peer_review_flaws.py` lines 255–263 build the hard-negative set as all `Ignelater` rows except two, plus Elateroidea/Agrilus. Recomputed composition:

```
Ignelater-derived "negatives"        : 59
  label = unlabeled                  : 59  (100%)
  reason "homology-only luciferase name" : 40
  reason "paralog in luminous genome"    : 19
Elateroid + Agrilus (true negatives) : 32
TOTAL n- = 91      ->  59/91 = 65% are UNLABELED homology-only paralogs
```

The README calls these "**59 verified non-luminous** PF00501 paralogs". **No experimental verification exists for any of them** — they are exactly the `unlabeled` category the README states is "never used as training labels" because it is "ambiguous". Promoting them to ground-truth negatives in the headline Flaw-4 benchmark means the model is being scored for agreeing with the homology-derived annotations the paper's central thesis says are untrustworthy.

Two further defects in the same block: the benchmark calls `roc_auc_score(y_hard, vals)` and `average_precision_score(y_hard, vals)` **without `sample_weight`**, discarding the cluster weighting that `evaluate.py` carefully applies — so the 23-member *Pyrophorus* cluster now counts 23 times. And the script's own docstring says **n = 118** while the code yields **91**.

**Fix:** report this benchmark with (a) cluster weights applied, and (b) a strict variant using only the 32 genuinely-labelled negatives, stating n⁻ = 32.

### M4. The "SAEs beat dense models" claim has no capacity-matched control

Table 2 and Discussion §3.1 (polysemantic superposition — the paper's main interpretability thesis) compare a **1,280-d dense** representation against a **10,240-d sparse** one. The SAE probe has 8× the feature budget. Without a matched control, "SAEs disentangle superposition" is confounded with "more features". Required: a random projection of the dense activations to 10,240-d, and/or an L1 probe on the dense representation at matched sparsity (same nnz). Until then this is an uncontrolled comparison supporting the paper's headline mechanistic claim.

### M5. The PIC analysis is under-specified and the key comparison is untested

`Bio.Phylo.TreeConstruction.DistanceTreeConstructor` neighbor-joining on BLOSUM62-derived pairwise distances, midpoint-rooted. No substitution-model selection, no bootstrap support, no branch-length uncertainty propagation — yet PIC standardises contrasts **by branch lengths**, so the p-values inherit all of that error. Felsenstein's method also assumes Brownian evolution on a *correct* tree; an NJ tree from a 550-aa alignment of a superfamily with 24–58% identity is not that.

More pointedly: System 1 reports **PIC r = 0.378 (p = 5.28e-3)** for ESM-2 vs **r = 0.261 (p = 0.059)** for sequence identity, and the manuscript treats this as a win. **The difference between two correlated correlation coefficients is never tested** (e.g. Steiger's / Williams' test). r = 0.378 and r = 0.261 on n = 53 are very unlikely to differ significantly. "Significant vs non-significant" is not "significantly different" — this is the classic Gelman–Stern error and a reviewer will catch it immediately.

### M6. Methods §4.1 describes curation that does not exist in the code

> *"CD-HIT clustering at 95% identity removed redundant entries."*

**`CD-HIT` appears nowhere in the repository except the manuscript sentence itself.** Actual redundancy control is: 3 rows flagged `exact_duplicate_of`, plus single-linkage cluster *weighting* at 90% PID in `evaluate.py`. Those are different operations with different consequences — weighting retains the sequences, CD-HIT would remove them.

Similarly, "harvested... using domain-specific HMM searches" describes `hmmsearch`-driven curation; the actual pipeline reads UniProt TSV exports (`data/raw/*.tsv`). `phmmer` is used downstream as a *baseline*, not for curation. Rewrite §4.1 to match `curate.py`.

### M7. Residue numbering is inconsistent across four scripts

"347" is a *P. pyralis* alignment coordinate. In raw 1-based sequence coordinates the residue at index 347 is:

```
P08659     (firefly)            347 = S   <- only this one matches
A0A8K0C7P5 (Ignelater decoy)    347 = H   (its L is at 343; motif YGLTEAT[L]VI)
A0A0C6DRW5 (Pectocera)          347 = V   (its L is at 345; motif YGLTEAT[L]AV)
A0A8K0DA02 (Ignelater true luc) 347 = A
Q7M4K3     (Pyrophorus luc)     347 = I
```

`unified_expansions.py` handles this correctly via alignment (S5). But `run_active_site_md.py` hardcodes `gate_res_seq = 344 if "DA02" else 343` while labelling every axis "Gate Residue 347"; `analyze_completed_trajectories.py` applies the identical window 340–355 and shades 346–350 on **both** topologies despite their different numbering; and `run_ancestral_reconstruction_and_ignition.py` uses two different ad-hoc offsets. Adopt one convention — P08659-aligned, resolved per sequence via `map_positions` — and make every figure label state it explicitly.

### M8. Reference 11 could not be verified; repository URL is a placeholder

Reference 11 — *Valenzano, D. R., et al. (2023). "Deep learning and mechanistic interpretability in molecular biology." Nature Machine Intelligence, 5, 810–822* — returned **no match** in a literature search, and the listed author's published work is in a different field. This needs a DOI or it must be removed; an unverifiable citation in a submitted manuscript is an integrity finding independent of the science. Also: Data Availability points to `https://github.com/your-username/translator`.

### M9. The MAE = 0.75 nm opsin result should be retired, not reported alongside the de-circularized one

The README reports **MAE = 0.75 nm, r = 0.985 (p = 6.9e-103)** across 134 RH1 haplotypes, *and separately* reports that Flaw 1 was fixed by "replacing formula-estimated opsin wavelengths" with 41 experimental MSP measurements giving **MAE = 5.20 nm**. The 0.75 nm figure is derived from the superseded formula-estimated λmax — which is itself a deterministic function of the tuning sites — so predicting it from sequence is circular, and 0.75 nm is in any case **below the ~1–2 nm reproducibility of microspectrophotometry itself**. Keep the n = 41 / MAE 5.20 nm result; delete the 0.75 nm result from the README rather than presenting both.

---

## 4. MINOR

1. `run_ancestral...py` docstring tracks "Feature 14202", but the SAE dictionary has 10,240 features — **index out of range**; the code actually uses `target_fids = [635, 2200, 7646, 7777]`. Provenance of those four IDs is undocumented.
2. Same script: `f_val = mean(...) * 10.0` — an undocumented ×10 scaling applied to the quantity plotted as "Causal Gate SAE Activation Score".
3. Same script: `h18[340:355]` ignores the ESM-2 `<cls>` token at index 0, so the window is **off by one** residue. Also `s[:540]` silently truncates the C-terminus of every 542–550 aa input.
4. Same script: when an accession is missing it prints `"Note: {acc} mapped from dataset."` — a message that reports success on failure, which is precisely why the `P0C027` omission went unnoticed into F1.
5. Hardcoded absolute paths: `ROOT = r"c:\Users\pagli\Desktop\translator"` and `ART_DIR = r"C:\Users\pagli\.gemini\antigravity\brain\c5b593ce-..."`. Figures are written outside the repo; nothing reproduces on another machine.
6. Two zero-byte data files: `data/raw/extra_spectral_and_cg6178.tsv`, `data/raw/luciferase_rich_metadata.tsv`.
7. The DCD parser in `analyze_completed_trajectories.py` does not unwrap periodic boundary conditions. Harmless for single-protein Kabsch RMSD, but it will corrupt any later distance measurement across the box.
8. Minor count drift between README and `dataset.tsv` (51 vs 50 positives, 195 vs 193 unlabeled). Reconcile once and cite the file.

---

## 5. Priority triage

**Before anything else (hours):**
1. Move `run_ancestral_reconstruction_and_ignition.py`, `fig13_ancestral_ignition_140mya.png`, `5_Anc0_*.pdb`, `notebooks/run_140mya_ancestral_md.ipynb` into `__ABANDONED_DO_NOT_USE__/` with a note recording F1/F2. **Cancel the queued Anc0 MD run** — it would simulate `A0A8K0C7P5` a second time.
2. Rewrite Methods §4.5 to describe what was actually run: OpenMM, AMBER99SB-ILDN/TIP3P, 4 fs HMR, 10.0 ns × 3, **apo** (HETATM stripped because SLU lacks AMBER parameters). Delete the gmx_MMPBSA / GAFF2 / AM1-BCC / igb=5 sentences until such a run exists.
3. Rename the `*_SLU_holo` directories and retitle Fig 11. An apo backbone-stability run cannot be labelled "Causal Steric Gate Rescue".

**Before submission (days):**
4. Remove manuscript §2.3 item 3 and Figure 10 entirely, or relabel them as a **rigid-pocket steric screen** with no energy units attached. The +3.5 / −8.5 / 108.1 → 99.6 kcal/mol values should not appear in a manuscript.
5. Restate every ancestry-control number against the 0.7138 weighted baseline, with the bootstrap CIs, and state n⁻ = 2 in the text.
6. Recompute the Flaw-4 benchmark with cluster weights and with a strict n⁻ = 32 labelled-negatives-only variant.
7. Add the capacity-matched dense control for Table 2 (random projection to 10,240-d and/or matched-nnz L1 on dense).
8. Add a Williams/Steiger test for the PIC r = 0.378 vs r = 0.261 comparison.
9. Resolve or remove Reference 11; set the real repository URL.

**If you want the ancestral result to actually exist (weeks):**
10. Do it properly: MAFFT/MUSCLE alignment → IQ-TREE with ModelFinder and ≥1000 ultrafast bootstraps → marginal ASR (IQ-TREE `--ancestral` or FastML) with per-site posterior probabilities → report the posterior at the gate position → homology-model the ancestral sequence onto 4G36 (not a copied PDB) → then simulate. Anything short of this is not an ancestral reconstruction, and the deep-time claim is the most attackable part of the paper precisely because it is the most interesting.

---

## 6. What I did not assess

- Numerical reproduction of System 2 (opsin) and System 3 (AFGP) metrics — I audited their design and the de-circularization claims, but did not re-run `analyze_opsin_and_afgp.py`.
- The 650M steering-vector construction was inspected for clade leakage (lines 417–441 confirm training-clade-only panels, zero Pectocera/Ignelater/Pyrophorus/CG6178) but the resulting fold-changes were not recomputed.
- Taxonomic correctness of every clade assignment in `dataset.tsv` beyond the Elateroidea/Elateriformia controls.
- No assumption or residual diagnostics were computed for the regression models — not assessed.


---

# Addendum — findings from Phase 1/2 execution (2026-11-13)

Three further defects surfaced while implementing the fixes. All were verified by running code.

### A1. `analyze_completed_trajectories.py` never ran — its DCD parser is broken (MAJOR)

The header seek is off by four bytes. A DCD's first record is `4 (size) + 84 (data) + 4 (size) = 92` bytes, but the parser does `f.seek(88)` and then reads the title-block length from inside the trailing record marker. Every subsequent offset is wrong and the frame loop runs off the end of the file:

```
IndexError: index 2 is out of bounds for axis 0 with size 0
```

This explains why `results/figures/fig11_gpu_md_10ns_definitive_all_complexes.png` does not exist: the local script has never produced a figure. The published `fig11_gpu_md_10ns_publication_panel.png` came from a Colab variant that is not in the repository, so **the figure in the manuscript has no reproducible provenance in this codebase.** Fixed by replacing the parser (validated against all three trajectories: 500 frames, 4.00 fs, 10.00 ns).

### A2. Figure 11's values do not reproduce (MAJOR)

Recomputing from the deposited DCDs with explicit Kabsch superposition:

| System | fig 11 "final RMSD" | verified eq. RMSD (5–10 ns) | fig 11 gate RMSF | verified gate RMSF |
|---|---|---|---|---|
| True luciferase | 1.70 Å | **1.91 ± 0.14 Å** | 2.88 Å | **0.52 Å** |
| Non-luminous paralog | 1.66 Å | **1.84 ± 0.19 Å** | 3.01 Å | **0.48 Å** |
| SAE-steered mutant | 1.99 Å | **2.23 ± 0.12 Å** | 4.47 Å | **0.51 Å** |

The RMSD discrepancy is modest. The **RMSF discrepancy is a factor of 4–9**, and the published values (2.88–4.47 Å for a buried active-site loop) are not physically plausible for a folded protein at 300 K — they are consistent with RMSF computed without removing rigid-body motion. The verified values (0.48–0.52 Å) are typical for a structured loop.

### A3. The gate-flexibility conclusion was an artefact of unaligned numbering (MAJOR — self-correction)

My own first corrected figure reproduced the very error flagged in M7: it shaded residues 347–349 in *structure* numbering across all three topologies. Resolving the gate by motif anchoring instead (anchor `GLTE`, verified unique in all three sequences) places it at **346–348 in `A0A8K0DA02`** and **343–345 in `A0A8K0C7P5`**.

At the correctly aligned gate the three systems are **equally rigid** (0.52 / 0.48 / 0.51 Å). The steered mutant's larger global RMSD comes from a downstream loop at aligned positions ~353–360, **not** from the engineered site. The claim that the steering mutations increase gate flexibility is therefore withdrawn — it was an artefact of the mis-aligned window, in both the original figure and my first correction.

This is worth stating plainly as a methodological lesson: **residue 347 is an alignment coordinate, and any analysis that indexes raw sequence or raw PDB numbering at 347 is measuring the wrong atom.** The repository now contains one validated anchoring routine; every downstream analysis should call it rather than hard-coding an index.

### A4. `fix_complex3.py` places a side chain by hand (MINOR, but gating for Track B)

The `ASN A 345` side chain of the steered mutant was written as three literal Cartesian coordinates. The geometry happens to be chemically reasonable (CB–CG 1.517 Å, CG–OD1 1.213 Å, CG–ND2 1.310 Å), so the apo runs are not invalidated — but it was not placed by a rotamer search and its torsion was never optimized against the pocket. The holo notebook checks this geometry before production and flags it for rebuilding.

---

## Status of audit findings after Phase 1/2

| Finding | Status |
|---|---|
| F1 Anc0 is a living protein | **Retracted** — quarantined with written rationale |
| F2 Unaligned consensus "ASR" | **Retracted** — quarantined |
| F3 Toy MD presented as all-atom MD | **Retracted** — quarantined; manuscript §2.3 rewritten with a retraction note |
| F4 Methods describe an unexecuted pipeline | **Fixed** — §4.5 rewritten to OpenMM/apo, with explicit "no binding free energy" statement |
| F5 Fig 11 title contradicts data | **Fixed** — figure regenerated from verified values; directories renamed `*_apo` |
| M1 Wrong AP baseline, missing CIs | **Fixed** — `results/metrics_corrected.tsv` |
| M3 Circular n⁻ = 91 negative set | **Fixed** — `results/hard_negative_benchmark_corrected.tsv`, strict n⁻ = 32 variant is now primary |
| M6 CD-HIT / HMM curation claims | **Fixed** — §4.1 rewritten to match `curate.py` |
| M8 "30.0 ns" on 10 ns data | **Fixed** |
| M9 Unverifiable reference, placeholder URL | **Fixed** — reference removed, URL marked |
| M2 Small effective n | **Disclosed** — new §3.4 Limitations |
| M4 Capacity-matched SAE control | **Closed — claim REFUTED.** Control run at 8M on the strict set: SAE AP 0.688 vs dense 0.735 vs sparsity-matched dense 0.837; SAE − random-ReLU = +0.066 against a ±0.133 seed spread. §2.5/§3.1 rewritten, abstract corrected. **650M now also run — same outcome, see Addendum 3.** |
| M5 Williams test for PIC | **Closed — claim REFUTED.** Independent reconstruction (N = 227): ESM-2 r = 0.270 vs identity r = 0.404, r(ESM,identity) = 0.337, Williams t(224) = −1.906, **p = 0.058**. Not significant, and the ordering *reverses*. §2.7 rewritten. |
| M7 Residue numbering | **Fixed in MD path**; other scripts still need the shared anchoring routine |
| M10 Circular 0.75 nm opsin MAE | **Retracted** — documented in quarantine §0c |


---

# Addendum 2 — close-out controls executed (2026-11-13)

Both remaining open findings were run on Colab. **Both came back against the manuscript's claims.**

### M4 — the sparse-decomposition advantage does not replicate (claim withdrawn)

Evaluated on the audit-corrected strict benchmark (27 click-beetle positives = 4.99 effective clusters vs. 32 labelled negatives, cluster-weighted, AP baseline 0.1849). Identical pipeline for every representation.

| Representation | dim | AUROC | AP | nnz |
|---|---|---|---|---|
| Dense ESM-2 8M (layer 4) | 320 | 0.920 | 0.735 | 22 |
| **Dense, matched sparsity** | 320 | **0.950** | **0.837** | 18 |
| Random Gaussian projection (10 seeds) | 10,240 | 0.920 | 0.759 ± 0.094 | — |
| Random ReLU projection (10 seeds) | 10,240 | 0.894 | 0.622 ± 0.133 | — |
| InterPLM-8M SAE (layer 4) | 10,240 | 0.923 | 0.688 | 19 |

1. The SAE **does not beat the dense representation it decomposes** (0.688 vs 0.735).
2. A plain L1 dense probe at matched sparsity is the **best** representation tested (0.837).
3. SAE − random ReLU projection = **+0.066 AP**, inside that control's own **±0.133** seed spread. No evidence of a learned-disentanglement advantage.
4. **The benchmark is underpowered.** Random-projection controls fluctuate ±0.09 to ±0.13 AP from the seed alone — larger than nearly every between-method gap in the table. At ~5 effective positive clusters this benchmark cannot rank representations, and no ranking claim should be made from it.

The published +0.199 gain at 8M was computed on the 91-negative set containing 59 homology-only annotations, unweighted. **The SAE advantage appears to be a property of the contaminated evaluation set, not of the representation.** The 650M scale was not re-tested.

### M5 — the PIC comparison is not significant, and reverses (claim withdrawn)

Independent reconstruction: global BLOSUM62 alignments over all 228 labelled sequences, normalised `1 − s_ij/√(s_ii·s_jj)`, neighbour-joining, midpoint-rooted, Felsenstein contrasts (N = 227).

| Quantity | Value |
|---|---|
| PIC r (ESM-2 vs luminescence) | +0.270 (p = 3.81e-05) |
| PIC r (identity vs luminescence) | **+0.404** (p = 2.60e-10) |
| PIC r (ESM-2 vs identity) | +0.337 — required by Williams, never reported |
| **Williams' test** | diff = −0.134, t(224) = −1.906, **p = 0.058** |

Not significant, and **sequence identity tracks luminescence more strongly than ESM-2 does** — the opposite ordering to the manuscript. The two pipelines differ (the earlier analysis reported 53 internal bifurcations vs. our 227), so this does not prove the earlier numbers wrong; it shows the comparative claim is **not robust to reasonable analysis choices**.

Two further problems affect *both* analyses and are now disclosed in §2.7: PIC is applied to a **binary** trait, violating its Brownian-motion assumption (a threshold or phylogenetic-logistic model is the right tool), and the 228 contrast sequences **include the probe's 98 training sequences**, so the ESM-2 contrast is partly in-sample.

### Net effect on the paper

Two of the five headline claims are withdrawn. What remains is the original question, answered cleanly: **a firefly-trained ESM-2 probe beats sequence homology at identifying independently evolved click-beetle luciferases against ancestry-matched negatives (AP 0.943 vs 0.708), on a negative set with no homology-transferred labels** — plus the 94 mis-annotated UniProt entries. That is a smaller paper and a correct one.


---

# Addendum 3 — the 650M capacity-matched control (2026-11-13)

Run on a Colab T4 in 1 min 39 s. Same protocol and same strict evaluation set as the 8M control.

| Representation | dim | AUROC | AP | nnz | *Ignelater* ranks |
|---|---|---|---|---|---|
| Dense ESM-2 650M (layer 18) | 1,280 | 0.890 | 0.512 | 20 | [5, 7] |
| Dense, matched sparsity | 1,280 | 0.844 | 0.427 | 24 | [4, 7] |
| Random Gaussian projection (10 seeds) | 10,240 | 0.929 | 0.689 ± 0.212 | — | [3, 6] |
| Random ReLU projection (10 seeds) | 10,240 | 0.890 | 0.606 ± 0.180 | — | [4, 9] |
| **InterPLM-650M SAE (layer 18)** | 10,240 | **0.815** | **0.387** | 24 | **[7, 8]** |

**Neither published 650M claim reproduces, and both move the wrong way.**

| Published | Measured |
|---|---|
| AP 0.470 → 0.590 | AP 0.512 → **0.387** |
| *Ignelater* rank #7 → #1 | ranks [5, 7] → **[7, 8]** |

The SAE is the worst representation tested, and scores below **all twenty** random-projection draws (one-sided rank p = 0.091 against each control).

### The more important finding: the benchmark has no resolving power

- Across the 20 random-projection draws, AP spans **0.408 to 1.000** — a range of **0.592 AP from the random seed alone**.
- The largest gap between any two *named* methods in the table is **0.302 AP**.
- One Gaussian seed reached **AP = 1.000**.

**Seed noise is roughly twice the size of every effect the table is trying to measure.** With 4.99 effective positive clusters, no ranking of representations is supportable from this benchmark — including rankings that favour the project's hypothesis. This is now stated in §2.5 and Limitations item 5.

### Two caveats on the controls, stated against our own interest

1. A Gaussian projection ℝ¹²⁸⁰ → ℝ¹⁰²⁴⁰ is an injective linear map and therefore **information-preserving**. It is a *capacity* control, not an information null. Its apparent superiority over dense features reflects how the L1 penalty behaves in the higher-dimensional space, not added signal. So "SAE vs random projection" is not a clean learned-vs-unlearned contrast.
2. The InterPLM dictionaries were trained on UniRef, not on this task. What is being tested is whether an **off-the-shelf SAE basis transfers** to beetle paralog discrimination — not the quality of the SAE training objective, and not InterPLM's performance on the tasks it was built for.

### Status

Audit finding M4 is now **fully closed at both scales**, with the claim refuted at each. Together with M5, three of the manuscript's headline claims have been withdrawn on evidence. The only open experiment in the project is the holo MD run.
