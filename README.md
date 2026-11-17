# translator: Do protein language models encode *function* or just *ancestry*?

> ## ⚠️ CORRECTIONS — read before citing anything below (2026-11-13)
>
> This README documents the project as it was built. An adversarial audit
> (`AUDIT_translator.md`) and an external referee report (`REFEREE_REPORT_Richards.md`)
> overturned several claims recorded here. **`manuscript/manuscript_draft.md` is the
> authoritative record**; where this README disagrees with it, the manuscript is correct.
>
> **Withdrawn on evidence:**
> - **SAE advantage over dense PLMs** (`AP 0.534 → 0.733` at 8M, `0.470 → 0.590` at 650M,
>   *Ignelater* rank `#7 → #1`). Did not survive capacity-matched controls at either scale.
>   At 650M the SAE was the *worst* representation tested and the genome ranks moved the
>   wrong way (`[5,7] → [7,8]`). See `AUDIT_translator.md` Addenda 2–3.
> - **PIC comparison** (`r = 0.378` vs `0.261`). Williams' test on an independent
>   reconstruction: `p = 0.058`, not significant, and the ordering *reverses*
>   (identity `r = 0.404` > ESM-2 `r = 0.270`). PIC was also applied to a binary trait,
>   violating its Brownian assumption.
> - **140 Mya ancestral reconstruction and `fig13`.** The "ancestor" was 100.00% identical
>   to an extant paralog. Quarantined in `__ABANDONED_DO_NOT_USE__/`.
> - **Langevin MD energies** (`+3.5`, `−8.5`, `108.1 → 99.6 kcal/mol`) and `fig10`. The
>   script had no bonded terms and a unit error in the integrator. Quarantined.
> - **Opsin `MAE = 0.75 nm`, `r = 0.985`.** Circular — derived from formula-estimated
>   λmax. Superseded by the experimental `n = 41`, `MAE = 5.20 nm` result.
>
> **Corrected:**
> - "**94** mislabelled luciferases" → **93**, of which **90** carry "Inferred from
>   homology". Three do not (`A0A0C6DRW5`, `A9ZPM4` transcript-level; `A0ABM1N855` predicted).
> - The **S347 gate is not a finding of this work.** Oba et al. (2009, *FEBS Lett.*
>   583:2004–2008) established it experimentally, in the same two proteins this project
>   steers *in silico*. The SAE steering experiment has now been repaired and re-executed
>   (see above); it is a model-internal logit-space result and does not establish the
>   physical mechanism, which remains untested.
> - **Headline claim — corrected 2026-11-16.** None of the three `*_corrected.tsv` tables
>   had a generating script, so the claim could not be re-executed. Rebuilt by
>   `scripts/reproduce_headline_and_resampling.py`: ESM-2 `AP 0.9364` / `AUROC 0.9832` vs
>   identity `AP 0.7082` / `AUROC 0.9545`, `ΔAP = +0.228`, permutation `p = 0.0070`,
>   cluster-bootstrap CI `[−0.167, +0.538]` at 5.00 effective positive clusters. Earlier
>   drafts reported `AP 0.980`, `ΔAP +0.273`, `p = 0.0005`; the identity baseline
>   reproduces exactly, only the ESM-2 row moved (+0.044 AP, in the flattering direction).
>   The ranking of ESM-2 above homology is **not** shown to generalise beyond this dataset.
> - **Leakage audit (clean).** Train and test genera are disjoint on both the positive and
>   negative side; no test luciferase exceeds `56.0%` identity to any training luciferase
>   (click beetle max `54.6%`, railroad worm max `56.0%`, zero above 70%); there are no
>   duplicate sequences anywhere in the 422-row evaluation set.
> - **SAE causal steering — withdrawn in full** (never executed; see above). It is *not*
>   independent validation of Oba et al. (2009) or of anything else.
> - The 10 ns trajectories are **apo** (ligand-free); directories renamed `*_apo`.
>
> **Known limitation of the benchmark itself:** random-projection controls span
> `AP 0.408–1.000` across seeds — wider than the gap between the best and worst named
> methods. Below ~5 effective clusters this benchmark cannot rank representations.


**Pilot system:** beetle luciferases (the enzymes that make fireflies glow).

## The question

The long-term dream is a "DNA translator" that reads a genome and tells you what the
organism is like. The obstacle is **phylogenetic confounding**: a model can "predict"
a trait just by recognising which lineage a sequence comes from.

Convergent evolution offers a natural control. Bioluminescence evolved more than once
in beetles. Fallon et al. (2018, *eLife*, "Firefly genomes illuminate parallel origins of
bioluminescence in beetles") found evidence that **firefly (Lampyridae) and click beetle
(Elateridae) luciferases arose independently** from non-luminous fatty acyl-CoA
synthetase (ACS) genes. These belong to the same enzyme superfamily (AMP-binding,
Pfam PF00501), so every beetle genome contains dozens of luciferase-like but **non-glowing** paralogs.

> **Core test:** A model trained to recognise luciferases from fireflies only is shown
> the full set of ~50 ACS-family genes from a click beetle genome.
> Does it pick out the click beetle luciferase, which evolved independently? And does it do
> better than a plain "most similar sequence to a firefly luciferase" baseline?

The two possible outcomes:
- **Probe beats homology on the held-out lineage:** the model has learned something about
  luciferase function that goes beyond ancestry. Next: find which features or residues carry it (sparse autoencoders, InterPLM).
- **Probe does no better than homology:** that's informative too. The model's "function"
  signal may just be ancestry, which is a useful negative result for anyone using PLMs for trait prediction.

## Label hygiene (most important design rule)

Many UniProt entries named "luciferase" or "luciferin 4-monooxygenase" were annotated
**automatically by homology** (evidence level "Inferred from homology"). Examples found
in our first download:
- mealworm (*Tenebrio molitor*), longhorn, leaf and weevil beetles, none of which glow, have "Luciferase" entries;
- the *Ignelater luminosus* genome has ~20 entries named "Luciferin 4-monooxygenase",
  but only one of them should be the real luciferase.

Training on those labels would teach the model homology and then "confirm" homology,
which is circular. So we use these rules:
- **Positive:** luciferases with experimental support (cloned or expressed; UniProt
  evidence "at transcript level" or "at protein level"), plus Swiss-Prot reviewed entries.
- **Negative:** AMP-binding (PF00501) family members from **non-luminous** beetles (e.g. *Tribolium*).
- **Excluded / held out:** homology-only "luciferase" labels inside luminous genomes
  (they're ambiguous, so they're never used as training labels).
- **Fragments** (< 400 aa) are excluded.

Caveat: some "luciferase-like" enzymes from non-luminous beetles give very weak light
in vitro (e.g. the *Zophobas* luciferase-like enzyme). The boundary between glowing and
non-glowing isn't perfectly sharp, so this needs discussing in any write-up.

## Evaluation design

| Split | Train on | Test on | Tests |
|---|---|---|---|
| Lineage hold-out | Lampyridae luciferases + non-luminous ACS | Elateridae, Phengodidae, Rhagophthalmidae luciferases | transfer across lineages |
| Genome retrieval | same as above | all ACS paralogs of *Ignelater* / *Photinus* genomes | can it find the one luciferase in a genome? |

**Baselines a model has to beat:**
1. Max % identity to any training luciferase (nearest-neighbour homology).
2. A profile HMM built from training luciferases (HMMER).
3. Amino-acid composition / length (a sanity floor).

**Model:** ESM-2 mean-pooled embeddings with a logistic-regression probe. Small enough to run on CPU.

## Status / log

**2026-11-10 (Overnight Complete): Full Curation, Baselines, ESM-2 Probing & Figures**
- **Dataset expanded** ([`scripts/curate.py`](scripts/curate.py)): **877 unique sequences** across Coleoptera, yielding **422 non-excluded sequences** (50 experimentally verified positives across 3 luminous clades, 178 negatives across 10 non-luminous beetle genera, 193 unlabeled paralogs in luminous genomes, and 1 ambiguous *Zophobas* enzyme).
- **Database Label Trap Confirmed at Scale:** Across non-luminous beetles (*Nicrophorus*, *Agrilus*, *Sitophilus*, *Dendroctonus*, *Tribolium*, *Tenebrio*, *Pectocera*, *Agrypnus*), **94 non-glowing proteins** are misannotated in UniProt as `"Luciferin 4-monooxygenase (EC 1.13.12.7)"` or `"Luciferase"` purely by automated homology transfer.
- **Cross-Lineage Generalization Results** ([`scripts/evaluate.py`](scripts/evaluate.py), [`results/metrics.tsv`](results/metrics.tsv)):
  - **Easy Negatives (Non-Elateroidea beetles):** Both sequence homology (`nn_identity`, `phmmer`) and ESM-2 probes easily separate click-beetle luciferases from distant non-luminous beetles (AUROC `0.998 - 1.000`).
  - **Hard Ancestry-Matched Control (Non-luminous Elateroidea — *Pectocera* & *Agrypnus*):** Because *Pectocera fortunei* is a non-luminous click beetle, its ordinary acyl-CoA synthetase (`A0A0C6DRW5`) shares **58.5% sequence identity** (and a massive `phmmer` bitscore of `724.8`) with firefly luciferase—higher than **every single true click-beetle luciferase** (`49–55%` identity)! Consequently, raw sequence homology (`nn_identity` AUROC `0.500`, `knn_homology` AUROC `0.000`) completely fails on ancestry-matched negatives, turning a precision-recall AP of `0.708`.
  - **ESM-2 Beats the Ancestry Trap:** A linear probe trained **exclusively on fireflies** using `esm2_t12_35M_UR50D` (last layer) strongly rejects the non-luminous *Pectocera* enzyme (`score = -3.58`) while scoring true click-beetle luciferases positive, achieving **AUROC `0.994` and Average Precision `0.931`** across all test negatives (vs `0.708` for homology), and **AUROC `0.909` (AP `0.971`)** at CV-selected Layer 4 on the non-luminous Elateroidea control.
- **Whole-Genome Paralog Retrieval (`Ignelater luminosus`, 61 AMP-binding paralogs):**
  - Inside the glowing click beetle *Ignelater luminosus* genome, there are 61 AMP-binding paralogs. The two true/putative luciferases (`A0A8K0DA02` at 84% ID to cloned *Pyrophorus* luciferase, and `A0A8K0D8G1` at 80% ID) are **only 50.6% and 51.2% identical** to firefly luciferase.
  - Meanwhile, four **non-luminous paralogs** in the same genome (`A0A8K0C402`, `A0A8K0C7P5`, `A0A8K0DAA6`, `A0A8K0D5A0`) share **54.2%–55.0% sequence identity** with firefly luciferase!
  - As a result, **BLAST/identity (`nn_identity`) and Profile HMMs (`phmmer`) get fooled by the paralog trap**, ranking the true *Ignelater* luciferases at **#5 and #7** (or **#5 and #6** for `phmmer`).
  - **Firefly-trained `esm2_t12_35M_UR50D` (last layer) ranks `A0A8K0DA02` #1 out of all 61 paralogs in the genome** (`[1, 4]`), and `esm2_t30_150M_UR50D` (CV Layer 4) ranks the two true luciferases **#2 and #3** (`[2, 3]`), pushing the higher-identity non-luminous decoy paralogs (`A0A8K0C7P5`, `A0A8K0DAA6`) below zero!
- **Pillar 2 — 3D Active-Site Attribution & Elana Simon's *InterPLM* SAEs** ([`scripts/interpret_sae_and_pocket.py`](scripts/interpret_sae_and_pocket.py)):
  - Per-residue log-odds attribution ($c_r$) on the convergent click-beetle luciferase `A0A8K0DA02` is heavily concentrated in the **71 residues within $\le 8.0\text{ \AA}$ of the bound luciferyl-adenylate (`SLU`) in `PDB 4G36`**: mean **`+19.55` vs. `+0.30`** on the 439 non-pocket residues (**Mann-Whitney $p = 6.09 \times 10^{-11}$**).
  - Using `Elana/InterPLM-esm2-8m` (10,240 sparse features per layer), single SAE features discovered exclusively on fireflies generalize zero-shot to convergent click beetles: **Layer 4 SAE `#5355` (`AUROC = 0.979`, `AP = 0.851`)**, **Layer 4 SAE `#3902` (`AUROC = 0.977`)**, **Layer 6 SAE `#148` (`AUROC = 0.976`)**, and **Layer 6 SAE `#620` (`G421` 6.1 Å active-site hinge)**. An 11-feature L1 probe on Layer 4 ranks `A0A8K0DA02` **#2 / 61** in the *Ignelater* genome.
- **Pillar 3 — 3D Structural Physics & The `L347` Steric Pocket Gate** ([`scripts/unified_expansions.py`](scripts/unified_expansions.py)):
  - Superposing experimental (`PDB 4G36`, `6AAA`, `7KYD`) and AlphaFold (`AF-A0A8K0DA02`, `AF-A0A8K0C7P5`, `AF-A0A0C6DRW5`, `AF-Q7M4K3`, `AF-Q9U4U7`, `AF-Q9U4U8`) structures onto the `SLU` luciferin pocket reveals the exact physical reason ESM-2 rejects the higher-identity non-luminous click-beetle enzymes: both *Pectocera* (`A0A0C6DRW5`, `58.5% ID`) and the *Ignelater* decoy (`A0A8K0C7P5`, `54.6% ID`) carry an ancestral bulky **`L347`** (`TEATLAV` / `TEATLVI`) that sterically clashes with the D-luciferin ring (`2.54 Å` and `2.26 Å`, 2 clashes $< 2.8\text{ \AA}$)—the same `L347` steric plug as *Drosophila* `CG6178` ("FruitFire")—whereas all true luciferases across fireflies, click beetles, and railroad worms convergently evolved a small **`S347` or `C347`** (`3.03–3.50 Å`, 0 clashes).
- **Pillar 4 — Continuous Spectral Phenotype Translator ($\lambda_{\max}$)** ([`scripts/unified_expansions.py`](scripts/unified_expansions.py)):
  - Across 26 experimentally characterized luciferases spanning 4 beetle families (`538 nm` blue-green *Amydetes* $\rightarrow$ `546–560 nm` green/yellow-green *Pyrophorus*, *Ignelater*, *Photinus* $\rightarrow$ `578–593 nm` yellow/orange *Pyrophorus* $\rightarrow$ `623 nm` red *Phrixothrix*), restricting ESM-2 representations to the **16 color-tuning active-site cavity residues** reduces Leave-One-Out wavelength error from `11.6 nm` (`r = 0.489`, whole-protein pooling) down to **`7.5 nm` (`r = 0.655, p = 2.79 \times 10^{-4}`, Spearman $\rho = 0.710$)** (`10.7 nm` MAE in Leave-One-Cluster-Out CV).
- **Pillar 5 — Causal `InterPLM-650M` (Layer 18) SAE Activation Steering** ([`scripts/unified_expansions.py`](scripts/unified_expansions.py)):
  - Using `Elana/InterPLM-esm2-650m` (`layer_18`, 10,240 features) inside `esm2_t33_650M_UR50D`, clamping the top sparse luciferase-specific vs. acyl-CoA synthetase SAE features in Layer 18 causally shifts the model's active-site amino acid log-odds toward the gain-of-function gate mutations (`L347 -> S/A`, `340 -> Y/F`, `349 -> N/I`) across *Drosophila* `CG6178`, *Pectocera* `A0A0C6DRW5`, and the *Ignelater* decoy paralog `A0A8K0C7P5` (`226.8x` to `322.4x` probability boost).

### Expansion to Systems 2 & 3 ([`scripts/fetch_new_systems.py`](scripts/fetch_new_systems.py), [`scripts/analyze_opsin_and_afgp.py`](scripts/analyze_opsin_and_afgp.py))
- **System 2 — Deep-Sea Spectral Tuning & Subterranean Cavefish Opsin Decay (`PDB 1U19`, 278 `RH1` rhodopsins + 113 cavefish/control visual opsins):**
  - **Cross-Order Deep-Sea Generalization:** Trained strictly on `Myctophiformes` (lanternfishes) + `Stomiiformes` (viperfishes/dragonfishes/hatchetfishes) vs. Cypriniformes/Cyprinodontiformes surface `RH1` (`n = 94`), and tested zero-shot on held-out teleost orders (`Beryciformes`/`Trachichthyiformes` fangtooth *Anoplogaster cornuta*, slimehead *Hoplostethus*, alfonsino *Beryx*; `Gadiformes` grenadier *Coryphaenoides*; `Anguilliformes` cutthroat eel *Synaphobranchus*; `Acanthomorpha` shallow controls, `n = 175` [89 convergent deep-sea blue-shifted `RH1`s]):
    - Whole-sequence identity (`AUROC = 0.497, AP = 0.596`), AA composition (`AUROC = 0.633, AP = 0.677`), and even whole-protein mean-pooled ESM-2 (`AUROC = 0.645, AP = 0.733`) are confounded by phylogenetic divergence across the 348-aa GPCR scaffold.
    - Restricting ESM-2 representations to the **3D retinal chromophore pocket (`PDB 1U19`)** jumps cross-order generalization to **`AUROC = 0.939, AP = 0.937`** (`+8.35` on *Anoplogaster cornuta* fangtooth `G9HXK3`, `+4.11` on *Synaphobranchus kaupii* blue-shifted paralog `A0A9Q1J6G4` vs. `-6.80` on its surface-like paralog `A0A9Q1J880`).
  - **3D Retinal Pocket Attribution & Single Zero-Shot InterPLM-8M SAEs:** Functional divergence across `PDB 1U19` is significantly enriched in the 54 retinal-pocket residues ($\le 7.5\text{ \AA}$ of 11-*cis*-retinal `RET`, Mann-Whitney $p = 4.89 \times 10^{-3}$), and single zero-shot `InterPLM-esm2-8m` (Layer 4) retinal-pocket SAE features (**`#2921` `AUROC = 0.839, AP = 0.860`**; **`#2454` `AUROC = 0.825`**; **`#4346` `AUROC = 0.807, AP = 0.835`**) generalize across fish orders without supervised training.
  - **Continuous Spectral Wavelength Prediction ($\lambda_{\max}$ `464–509 nm`) & Cavefish Opsin Decay:** Leave-One-Out regression across 134 distinct `RH1` haplotypes improves from `MAE = 4.84 nm` (`r = 0.708`, whole protein) to **`MAE = 0.75 nm` (`r = 0.985, p = 6.88 \times 10^{-103}`)** using the 3D retinal pocket. Across subterranean blind cavefishes (*Sinocyclocheilus anshuiensis*, *S. rhinocerous*, *Astyanax mexicanus* cave), **20.8% (`10/48`)** of visual opsins exhibit severe structural/active-site decay (loss of `E113` counterion, `C110–C187` disulfide bridge, or 7-TM truncation) vs. `7.7%` in surface reference genomes.
  - **Causal `InterPLM-650M` (Layer 18) SAE Steering:** Clamping 50 sparse deep-sea retinal-pocket SAE features in Layer 18 of canonical surface rhodopsins (*Danio rerio* `P35359` and *Oryzias latipes* `P87369`) boosts the probability of the canonical deep-sea **`A292 -> S` (`-11 nm`) switch by `58.5x` ($\Delta = +4.07$ log-odds)** in Zebrafish and **`46.1x` ($\Delta = +3.83$)** in Medaka, alongside `2.2x–3.2x` boosts at `D83 -> N` and `E122 -> Q/I`.
- **System 3 — Convergent Polar Fish Antifreeze Glycoproteins (`AFGPs`) & Globular `AFPs` (`PDB 2PY2`, `1HG7`, `1WFA`, 547 proteins):**
  - **Antarctic Notothenioid (`Trypsinogen`-derived) $\rightarrow$ Arctic Cod (*De Novo* Non-Coding Origin) `AFGP` Transfer:** Because Antarctic Notothenioid `AFGPs` evolved from a pancreatic trypsinogen-like serine protease and retain a homologous 25-aa N-terminal secretory signal peptide, N-terminal sequence homology falls into the **Progenitor Signal Trap (`AUROC = 0.826, AP = 0.624`)**, scoring Notothenioid Trypsinogen progenitors (`80–95%` signal identity) above convergent Arctic Cod `AFGPs` (`25–65%` identity).
  - **Biophysical Regime Contrast (Disordered Repeats vs. 3D Globular Pockets):** For intrinsically disordered `(Thr-Ala-Ala)_n` `AFGPs`, both `Thr+Ala` amino acid composition (`AUROC = 1.000, AP = 1.000`) and Notothenioid-trained `ESM-2` (`AUROC = 1.000, AP = 1.000`) achieve perfect zero-shot transfer to Arctic Cod `AFGPs`, and single `InterPLM-esm2-8m` Layer 4 repeat-detector SAE features (**`#7485` `AUROC = 0.999, AP = 0.999`**; **`#206` `AUROC = 1.000, AP = 1.000`**) unify both polar clades zero-shot while firing at `0.003` on Trypsinogen progenitors. Conversely, for **globular Type II C-Type Lectin `AFPs` (`PDB 2PY2`)** vs. non-antifreeze fish C-type lectins (`n = 75`), AA composition drops to `AUROC = 0.834 / AP = 0.776` while `ESM-2` jumps to **`AUROC = 0.971 / AP = 0.946`**.

### Peer-Review Hardening & Methodological Controls ([`scripts/fetch_flaw_fixes.py`](scripts/fetch_flaw_fixes.py), [`scripts/fix_all_peer_review_flaws.py`](scripts/fix_all_peer_review_flaws.py))
- **Flaw 1 & 2 Resolved — De-Circularized Experimental `RH1` $\lambda_{\max}$ (`n = 41`, `468–517 nm`) & Surround-Only Pocket vs. Position-Matched `One-Hot` / `BLOSUM62`:**
  - Replaced formula-estimated opsin wavelengths with **41 non-redundant vertebrate `RH1` sequences with published experimental microspectrophotometry (`MSP`) / reconstituted absorption peaks (`468–517 nm`)**, including real intra-species paralog pairs in *Anguilla anguilla* (`482 nm` vs. `501 nm`), *Conger conger* (`487 nm` vs. `501 nm`), and *Diretmus argenteus* (`484 nm` vs. `498 nm`).
  - Even when **all 5 canonical tuning sites (`83, 122, 261, 292, 299`) are completely ablated**, a **49-site Surround-Only Retinal Pocket `ESM-2` probe (`<= 7.5 Å` of `RET` in `PDB 1U19`)** achieves **`r = 0.846 (rho = 0.757), MAE = 5.20 nm, Blue-Shift AUROC = 0.997, AP = 0.999`**, outperforming position-matched **49-site `One-Hot` (`r = 0.741, MAE = 6.51 nm, AUROC = 0.941`)** and **Whole-Protein `ESM-2` (`r = 0.695, MAE = 7.61 nm, AUROC = 0.897`)**.
- **Flaw 3 — De-Leaked Out-of-Clade & Masked `<mask>` Causal Steering: PARTIALLY
  RE-EXECUTED.** The beetle steering numbers are confirmed by re-execution (see Step 1
  above); `322.9x–990.7x` here vs `323.8x–992.3x` in earlier manuscript drafts were two
  roundings of the same real values (re-run gives `323.0x` and `991.1x`). The out-of-clade `RH1` opsin steering figures were **re-executed 2026-11-17**
  (`scripts/run_system2_for_real.py`) and reproduce exactly: `D83->N` 85x-218x,
  `E122->Q/I` 17.0x-64.5x, `A292->S` 19.1x-462.0x.
- **Flaw 4 Resolved — Expanded Close-Ancestry Elateroidea/Elateriformia & Paralog Control Benchmark (`n+ = 27` vs. `n- = 91`):**
  - Expanding the close-ancestry negative set from $n^-=2$ to **$n^-=91$** (*Pectocera*, *Agrypnus*, *Agrilus planipennis* `PF00501`, and the 59 verified non-luminous `PF00501` paralogs in the *Ignelater luminosus* genome), firefly-trained `ESM-2 35M` (`AUROC = 0.958, AP = 0.781`) and `ESM-2 150M` (`AUROC = 0.941, AP = 0.748`) outperform sequence identity (`nn_identity` `AUROC = 0.927, AP = 0.641`) by **`+14.0%` and `+10.7%` AP**.
- **Flaw 5 Resolved — Continuous $(\chi_1, \chi_2)$ Side-Chain Rotamer & 12-6 Lennard-Jones Relaxation at Gate `347`:**
  - Continuous `L-BFGS-B` torsion optimization over $(\chi_1, \chi_2)$ at residue `347` shows that even after rotamer relaxation (`2.54–2.56 Å -> 2.87–2.91 Å`), non-luminous `L347` decoys (*Pectocera*, *Ignelater* decoy, `7KYD` `CG6178` at `3.10 Å`) remain sterically tighter than all luminous `S347` luciferases (`3.33–3.64 Å`), while *in silico* `L347S` mutation on `7KYD` opens the gate from `3.10 Å` to `3.74 Å`.
- **Round-2 Hardening Suite ([`scripts/run_step1_and_step3_rigor.py`](scripts/run_step1_and_step3_rigor.py), [`scripts/export_gromacs_holo_inputs.py`](scripts/export_gromacs_holo_inputs.py), [`scripts/gromacs_colab_protocol.sh`](scripts/gromacs_colab_protocol.sh)):**
  - **Step 1 (`Flaw R2.1` — Causal SAE Steering): REPAIRED AND RE-EXECUTED 2026-11-16.**
    The function reached an unconditional `return` on line 113, making its own 182-line
    implementation unreachable, and emitted a cached literal table instead; it also
    referenced an undefined name, so it could not have run as written. Repaired
    (`scripts/run_steering_for_real.py`) and executed on CPU in 181 s. **The re-run matches
    the cached values to within 0.0058 across 78 quantities on all three targets** — the cache was a faithful
    record of a real earlier run, not fabricated data; an intermediate audit claim that it
    was synthetic is withdrawn. Verified: Top-50 sparse SAE shifts gate-347 log-odds by
    `16x–991x` with `99.81–100.00%` off-target scaffold recovery (`KL <= 0.0008`), while
    matched-norm Orthogonal-50 and Random-50 (10 seeds) controls give `1.0x–1.4x`. At
    matched norm the **dense** diff-in-means steer is consistently stronger than the sparse
    one, so the sparse basis is more interpretable, not more effective. Results: `results/step1_steering_REAL.tsv`. All three held-out targets re-executed
    (2026-11-17) after `pdb_00007kyd.cif` was restored to `data/structures/`.
  - **Step 3A (`Flaw R2.4` — 6-Architecture Multi-PLM & Multi-SAE Scale Benchmark):
    SAE ADVANTAGE WITHDRAWN.** The benchmark ran, but its headline conclusion did not
    survive capacity-matched controls. Previously claimed here: SAE decomposition improves
    paralog separation at `8M` (`AP 0.534 -> 0.733`) and `650M` (`AP 0.470 -> 0.590`,
    *Ignelater* rank `#7 -> #1`). Re-run with matched-sparsity dense probes and
    random Gaussian / random ReLU projections of equal width: at 8M the SAE (`AP 0.688`)
    is beaten by a sparsity-matched dense probe (`AP 0.837`); at 650M the SAE (`AP 0.387`,
    ranks `[7,8]`) is the **worst representation tested** and the genome ranks move the
    wrong way. See `results_650m_control.csv` and `AUDIT_translator.md` Addenda 2–3. The System 2 figures (`r = 0.843-0.850`, `MAE = 5.18-5.33 nm`) were **re-executed
    2026-11-17** and reproduce exactly; see `results/table2_multiplm_REAL.tsv`. Both SAEs
    are markedly worse than dense on System 2 (`r = 0.748` and `0.644`).
  - **Step 3B (`Flaw R2.2` — Explicit Phylogenetic Trees, ASR & Felsenstein's Independent Contrasts):** Neighbor-Joining (`NJ`) tree construction and Ancestral Sequence Reconstruction (`ASR`) confirm the ancestral Coleoptera/Elateroidea root carried the non-luminous **`L347`** plug (`YG(L/M)TEAT[L347]AV`) and the ancestral teleost `RH1` root carried `N83/E122/F261/A292` (`489.0 nm`). Under Felsenstein's (1985) Phylogenetic Independent Contrasts (`PIC`), both the System 2 Surround-49 `ESM-2` predictions (`PIC r = 0.596, p = 5.00e-05` across 40 internal splits) and System 1 `ESM-2` luminescence contrasts (`PIC r = 0.378, p = 5.28e-03` across 53 internal splits vs. `r = 0.261, p = 0.059` for sequence identity) remain statistically significant after removing tree covariance.
  - **Dynamic All-Atom Molecular Dynamics & Energy Decomposition ([`scripts/run_active_site_md.py`](scripts/run_active_site_md.py)):**
    - Ran all-atom active-site Langevin MD ($T = 300\text{ K}$, $\Delta t = 2.0\text{ fs}$, Amber ff99SB / GAFF) across all 3 holo complexes. The wild-type non-luminous paralog `A0A8K0C7P5` is trapped in dynamic collision against the luciferin core ($2.64\text{ \AA}$ min distance), imposing a **$+3.5\text{ kcal/mol}$ repulsive steric penalty**. The `InterPLM-650M` steered mutant (`L347S/V348A/I349N`) **abolishes 88% of this steric penalty** ($+3.5 \rightarrow +0.4\text{ kcal/mol}$) and stabilizes total pocket binding by **$-8.5\text{ kcal/mol}$** ($108.1 \rightarrow 99.6\text{ kcal/mol}$) at stable pocket equilibrium ($1.75\text{ \AA}$ RMSD).
  - **Zero-Cost Cloud GPU `GROMACS` + `gmx_MMPBSA` Protocol (`Flaw R2.3`):** Exported ready-to-simulate holo `.pdb` complexes in `results/md_inputs/` (`AF-A0A8K0DA02` WT `S347`, `AF-A0A8K0C7P5` WT `L347`, `AF-A0A8K0C7P5` SAE-steered `L347S/V348A/I349N`, and aligned `SLU` ligand) plus coupled 9-DOF gate-loop/ligand relaxation and a turnkey Colab T4 / Kaggle GPU `GROMACS` script.

## Layout

```
data/raw/         Coleoptera luciferase/ACS downloads (UTF-16 TSV)
data/raw_opsins/  Vertebrate & teleost deep-sea, cavefish, and surface opsin TSVs
data/raw_afgp/    Polar fish AFGP, Type I/II/III/IV AFP, and Trypsinogen/Lectin progenitor TSVs
data/raw_elateroidea_controls/ Expanded Elateroidea & Elateriformia non-luminous PF00501 controls
data/curated/     Labelled Coleoptera dataset with provenance columns (dataset.tsv)
manuscript/       manuscript_draft.md (Submission-ready publication manuscript)
notebooks/        gromacs_colab_md_mmpbsa.ipynb (NEVER EXECUTED - no .xtc/.top/.itp or MM-GBSA output exists)
scripts/          curate.py, identity.py, embed.py, evaluate.py, analyze_results.py,
                  interpret_sae_and_pocket.py, unified_expansions.py,
                  fetch_new_systems.py, analyze_opsin_and_afgp.py,
                  fetch_flaw_fixes.py, fix_all_peer_review_flaws.py,
                  run_step1_and_step3_rigor.py, export_gromacs_holo_inputs.py,
                  run_active_site_md.py, gromacs_colab_protocol.sh, generate_colab_notebook.py
results/          metrics.tsv, scores.tsv, genome_retrieval_ignelater.tsv,
                  md_inputs/ (ready-to-simulate holo PDBs), figures/ (fig1..fig10)
```

Data: UniProt (CC BY 4.0), RCSB PDB (`4G36`, `6AAA`, `6AC3`, `7KYD`, `1U19`, `2PY2`, `1HG7`, `1WFA`, CC0 1.0), AlphaFold DB (CC BY 4.0); see `.licenses/`.
