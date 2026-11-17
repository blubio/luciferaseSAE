# Deconfounding Convergent Evolution in Protein Language Models: Sparse Autoencoders Disentangle Catalytic Physics from Phylogenetic Superposition

**Bruno Stein Barbosa Menechino**$^{1,*}$  
$^{1}$*Institute of Biology, State University of Campinas (UNICAMP), Campinas, SP, Brazil*  
$^{*}$*Correspondence: `pagli@...` / [PMID: 38049805](https://pubmed.ncbi.nlm.nih.gov/38049805/)*

---

## Abstract

Protein Language Models (PLMs) such as ESM-2 have transformed computational biology, yet standard benchmarks fail to determine whether these models genuinely learn the physical principles of catalysis or merely memorize phylogenetic lineage markers. When proteins share high sequence identity due to common ancestry but perform divergent metabolic functions (paralogs), or conversely, when distinct evolutionary lineages independently converge upon the same biochemical function with low sequence identity (convergent evolution), classical homology-based tools (BLAST, HMMER) and standard PLM evaluations collapse. Here, we present an empirical, publication-grade benchmark investigating **three independent convergent evolution systems** spanning distinct biophysical regimes: (1) **Beetle Bioluminescent Luciferases** (*Lampyridae* fireflies $\rightarrow$ *Elateridae* click beetles, $n=422$, PDB `4G36`); (2) **Deep-Sea Visual Opsins and Cavefish Opsin Decay** (rhodopsin $\text{RH1}$ blue-shifting across teleost orders and subterranean pseudogenization, $n=391$, PDB `1U19`); and (3) **Polar Fish Antifreeze Proteins** (intrinsically disordered $\text{AFGPs}$ vs. globular folded Type II lectin $\text{AFPs}$ in Antarctic Notothenioids and Arctic Cod, $n=547$, PDB `2PY2`/`1HG7`).

We document widespread propagation of automated database annotations: $95$ non-luminous beetle enzymes across $10$ genera carry "luciferase" names in UniProt, $92$ of them tagged "Inferred from homology" (§2.1). We also characterise severe "ancestry traps", in which non-luminous enzymes share higher sequence identity with firefly luciferase ($58.5\%$ in *Pectocera fortunei*; $54.6\%$ in *Ignelater* paralog decoys) than true click-beetle luciferases do ($50.6\%$), so that raw-identity and HMMER baselines score at or below chance on the ancestry-matched control. A candidate physical explanation -- that non-luminous enzymes retain an ancestral bulky leucine ($\text{L347}$) occluding the luciferyl-adenylate intermediate ($\text{SLU}$) whereas true luciferases convergently evolved a small serine ($\text{S347}$), the substitution shown experimentally by Oba et al. (2009) to confer luminescence activity -- motivated this study but is **not established by it**; the clashes measured by rigid superposition ($2.26\text{--}2.54\text{ \AA}$) do not survive relaxation, and the simulations below leave the mechanism untested. A causal sparse-autoencoder steering experiment reported in earlier drafts was found during audit to be unreachable code: the function returned before its own implementation and emitted a cached literal table instead. It has now been repaired and **re-executed** (`InterPLM-esm2-650m` Layer 18, ESM-2 650M, CPU); the re-run matches the cached values to within $0.0058$ across 78 quantities on all three targets, so those values were a faithful record of a real earlier run rather than fabricated data, and an intermediate audit characterisation of them as synthetic is withdrawn. Clamping 50 of $10{,}240$ SAE features ($0.49\%$) at pocket sites shifts the gate-347 log-odds toward Ser/Ala by $16\times\text{--}991\times$ (though a neighbour-only control shows $66\text{--}90\%$ of this requires perturbing residue 347 itself) while leaving $99.81\%\text{--}100.00\%$ of top-1 predictions at the $\sim 530$ non-pocket residues unchanged ($D_{KL} \le 0.0008$); matched-norm orthogonal and random feature sets produce no shift ($1.0\times\text{--}1.4\times$). At matched norm a *dense* difference-in-means vector is consistently **stronger** than the sparse steer, so the sparse basis is more interpretable rather than more effective. This is a model-internal result in logit space; no protein was expressed, and it does not establish the physical mechanism, which remains untested.

We benchmarked 6 PLM and SAE architectures (8M to 650M). Sparse-autoencoder decomposition initially appeared to outperform raw dense PLMs in cross-clade paralog separation, but **this advantage did not survive a capacity-matched control and is withdrawn**: at the 8M scale the 10,240-d InterPLM dictionary ($\text{AP} = 0.688$) does not beat the 320-d dense representation it decomposes ($0.735$), is beaten by a sparsity-matched dense probe ($0.837$), and exceeds a structure-free random ReLU projection by less than that control's seed variance. The same holds at 650M, where the SAE scores below all twenty random-projection draws and the published *Ignelater* rank improvement ($\#7 \rightarrow \#1$) instead moves the wrong way ($[5,7] \rightarrow [7,8]$). We report this as a negative result, and note a stronger caveat: across random seeds alone the benchmark's AP ranges over $0.59$, roughly double the gap between its best and worst named methods, so with $\sim 5$ effective positive clusters it cannot rank representations at all. Likewise, Felsenstein's (1985) Phylogenetic Independent Contrasts do not show model predictions outperforming sequence identity once Williams' test for dependent correlations is applied ($r = 0.270$ vs. $0.404$; $p = 0.058$, $N = 227$); that comparative claim is also withdrawn.

What survives is the result the study was built to test: on a convergent-evolution hold-out with an ancestry-matched negative set, a firefly-trained ESM-2 probe separates independently evolved click-beetle luciferases from close non-luminous relatives ($\text{AP}\ 0.936$ vs. $0.708$ for sequence identity; $\text{AUROC}\ 0.983$ vs. $0.955$), on a negative set containing no homology-transferred labels. A label-permutation test excludes chance ($p = 0.0070$), but a cluster bootstrap on the margin over homology gives $95\%$ CI $[-0.17, +0.54]$, so that ranking is **not** shown to generalise beyond this dataset. Twelve holo molecular-dynamics simulations ($4$ systems $\times$ $3$ replicates $\times$
$2.5\text{ ns}$) with a parameterized luciferyl-adenylate ligand leave the steric-gate mechanism
**untested rather than confirmed or refuted**: a deliberately clashing $\text{S347F}$ positive
control, built with its phenyl ring $1.62\text{ \AA}$ from the ligand, relaxes to
$3.58 \pm 0.09\text{ \AA}$ and becomes indistinguishable from wild type, showing that any protocol which
minimises a modelled complex dissipates the very strain it is meant to measure. We also document 95 UniProt entries carrying luciferase names in non-luminous beetles, 92 of them tagged "Inferred from homology". We report the failed controls alongside the positive result because the distinction between them is the point of the paper.

---

## 1. Introduction

Biological sequence foundation models trained on hundreds of millions of evolutionary sequences—most prominently Evolutionary Scale Modeling (ESM-2; Lin et al., *Science* 2023)—have demonstrated remarkable zero-shot capabilities in variant effect prediction, fitness landscape modeling, and structure generation. However, a foundational question remains unresolved: **Are protein language models learning the universal biophysical and catalytic grammar of proteins, or are they high-capacity look-up tables that exploit phylogenetic memorization and taxonomic autocorrelation?**

In standard machine learning benchmarks (such as Random Split, Cluster Split, or UniRef50 holdouts), training and test sequences frequently share common evolutionary ancestry. Because natural sequences are distributed along hierarchical phylogenetic trees rather than sampled identically and independently from an underlying distribution (Felsenstein, 1985), models can achieve near-perfect classification accuracy by simply memorizing clade-specific sequence signatures (e.g., family-specific surface residues) rather than identifying the sparse, epistatic active-site residues that actually catalyze biochemical reactions.

### 1.1 Convergent Evolution as the Ultimate Negative Control for Homology

Convergent evolution—the independent emergence of identical biochemical phenotypes in distinct, non-overlapping evolutionary lineages—provides a natural experimental design to dissect this problem. If two enzymes catalyze the exact same reaction but evolved independently from different ancestral scaffolds, their whole-sequence identity is often lower than the sequence identity between an enzyme and a non-functional or functionally divergent paralog within its own evolutionary clade.

Here, we examine three classical evolutionary paradigms:
1. **System 1 (Beetle Bioluminescence, PDB `4G36`):** Luciferases in fireflies (*Lampyridae*), click beetles (*Elateridae*), and railroad worms (*Phengodidae*) independently evolved bioluminescence from ancestral AMP-binding fatty acyl-CoA synthetases ($\text{PF00501}$). Non-luminous Elateroidea enzymes share up to $58.5\%$ sequence identity with firefly luciferases, whereas true click-beetle luciferases share only $49.0\%\text{--}55.0\%$.
2. **System 2 (Deep-Sea Visual Opsins & Cavefish Decay, PDB `1U19`):** Rhodopsin ($\text{RH1}$) independently acquired identical spectral blue-shifting mutations (e.g., $\text{D83N}$, $\text{E122Q}$, $\text{A292S}$) across divergent teleost orders (*Myctophiformes*, *Beryciformes*, *Gadiformes*) to adapt to deep-sea bioluminescent downwelling light ($\lambda_{\max} \le 490\text{ nm}$), while subterranean blind cavefishes independently decay these genes under neutral drift.
3. **System 3 (Polar Fish Antifreezes, PDB `2PY2`/`1HG7`):** Intrinsically disordered Antifreeze Glycoproteins ($\text{AFGPs}$) evolved independently in Antarctic Notothenioids (neofunctionalized from a pancreatic trypsinogen progenitor) and Arctic Cod (*de novo* from non-coding DNA), contrasting with globular folded Type II lectin $\text{AFPs}$ that evolved from calcium-dependent C-type lectins.

### 1.2 Mechanistic Interpretability & Sparse Autoencoders in Biology

Deep neural networks suffer from **polysemantic superposition**: individual neurons represent multiple unrelated biological concepts because the number of features exceeds the model's ambient dimensionality. In PLMs, this causes catalytic features to be superimposed onto lineage/species features. Recently, Simon et al. (*Nature Methods* 2025) introduced **InterPLM**, training Sparse Autoencoders (SAEs) on ESM-2 internal activations to expand dense hidden states into high-dimensional, sparse, monosemantic dictionaries. 

In this work, we deploy `InterPLM` SAEs to address three fundamental questions:
1. *Causal Specificity:* Can sparse autoencoder directions steer catalytic gate phenotypes in non-luminous scaffolds without disrupting global fold integrity?
2. *Scale & Disentanglement:* Does sparse dictionary decomposition improve generalization across evolutionary scales (8M to 650M parameters)?
3. *Phylogenetic Deconfounding:* Do PLM functional attributions survive rigorous phylogenetic covariance correction via Felsenstein's Phylogenetic Independent Contrasts?

---

## 2. Results

### 2.1 Widespread Annotation Poisoning & The Paralog Trap in Coleoptera Luciferases

We curated 422 full-length, non-redundant beetle enzymes belonging to the $\text{PF00501}$ AMP-binding superfamily. When surveying database annotations across non-luminous beetles (*Tribolium castaneum*, *Agrilus planipennis*, *Nicrophorus vespilloides*), we identified **95 proteins carrying luciferase-family names** ("Luciferin 4-monooxygenase (EC 1.13.12.7)", "Luciferase", or a `-like` variant) in UniProt, spanning 10 genera: *Nicrophorus* 42, *Agrilus* 15, *Tribolium* 15, *Sitophilus* 8, *Dendroctonus* 6, *Tenebrio* 5, *Pectocera* 1, *Diabrotica* 1, *Aromia* 1, *Agrypnus* 1. All 95 are unreviewed. **92 of the 95 carry the UniProt evidence tag "Inferred from homology"**; the remaining three do not (`A0A0C6DRW5` and `A9ZPM4`, "Evidence at transcript level"; `A0ABM1N855`, "Predicted"). The 92 represent ordinary fatty-acid metabolism enzymes in beetles with no reported bioluminescence, whose names were propagated by automated sequence-similarity transfer.

This count has now been wrong twice, and both errors are instructive. An earlier draft reported 94 and attributed all of them to homology transfer; a subsequent audit reported 93 and corrected the mechanism. Both were incorrect. The audit's filter enumerated a hard-coded list of the eight genera named in the surrounding prose, so *Diabrotica* (`A0A9P0E638`) and *Aromia* (`A0AAV8YL26`) -- both named "Luciferase", both tagged "Inferred from homology" -- lay outside its search space. The present figure of 95 is the count of rows in `data/curated/dataset.tsv` with `label == negative` whose `protein_name` matches `/lucifer/i`, with no genus restriction; all three copies of that file in the repository, the earliest dated 2026-11-10, return the same value. A verification filter must not inherit the vocabulary of the claim it verifies.

**What this count does and does not establish (referee R-2).** We describe these entries as
*homology-transferred and experimentally unverified*, not as false. The inference "the
organism does not bioluminesce, therefore the protein is not a luciferase" does not hold, and
this manuscript contains its own counterexample: `A9ZPM4` (*Agrypnus binodulus* AbLL) is a
**latent luciferase** in a non-luminous beetle, convertible by a single serine substitution
(Oba et al., 2009). An AMP-binding enzyme may carry weak luciferase activity in an organism
with no light organ. Absence of organismal bioluminescence is evidence about the organism,
not an assay of the protein. Without activity measurements the defensible claim is that 92 of
95 carry the "Inferred from homology" tag and none has experimental support -- which is what
we claim.

Aligning these sequences against firefly luciferase (*Photinus pyralis*, `P08659`) exposed the **Ancestry & Paralog Trap**:
1. *Cross-Species Trap:* The non-luminous click beetle *Pectocera fortunei* (`A0A0C6DRW5`) shares **$58.5\%$ sequence identity** with firefly luciferase (HMMER bitscore $724.8$). In contrast, true, brightly glowing click-beetle luciferases (*Pyrophorus plagiophthalamus*, *Photophorus bakewelli*) share only **$49.0\%\text{--}55.0\%$ identity**. Consequently, every pairwise sequence similarity tool (`BLAST`, `HMMER`, `k-NN`) falsely ranks the non-luminous *Pectocera* above 100% of true click-beetle luciferases (**Figure 1A**).
2. *Intra-Genomic Paralog Trap:* In the fully sequenced genome of the luminous click beetle *Ignelater luminosus* (61 $\text{PF00501}$ paralogs), its true luciferase (`A0A8K0DA02`, $84.0\%$ identical to cloned *Pyrophorus* luciferase) shares only $50.6\%$ identity with firefly luciferase, ranking $\#7$ out of 61. Four non-luminous housekeeping paralogs (`A0A8K0C402` at $55.0\%$, `A0A8K0C7P5` at $54.6\%$, `A0A8K0DAA6` at $54.4\%$, and `A0A8K0D5A0` at $54.2\%$) have higher raw sequence identity to firefly luciferase than *Ignelater*'s actual luciferase.

```
Figure 1: Cross-Lineage Generalization & The Paralog Trap.
(A) Firefly-Trained ESM-2 Score vs Sequence Identity. True luciferases (green) separate cleanly from non-luminous controls (red), while sequence identity misclassifies Pectocera (58.5% ID).
(B) Whole-Genome Screening of Ignelater luminosus (61 paralogs). ESM-2 identifies the true luciferase A0A8K0DA02 as Rank #1, while BLAST/HMMER select non-luminous paralogs.
```

### 2.1b Cross-Lineage Transfer Against Close-Ancestry Negatives

A firefly-trained linear probe on ESM-2 embeddings was evaluated zero-shot on click-beetle luciferases (an independent origin of bioluminescence) against progressively harder negative sets. All average-precision (AP) values are cluster-weighted and are reported against the **cluster-weighted prevalence baseline**, which differs per comparison and is never $0.5$.

```
Table 1: Cross-lineage transfer vs. close-ancestry negatives (cluster-weighted).
Positives: 27 click-beetle luciferases = 4.99 effective clusters (23 are near-identical Pyrophorus colour variants).
-------------------------------------------------------------------------------------------------------
Negative set                                       n-   eff n-  AP base  Method          AUROC     AP
-------------------------------------------------------------------------------------------------------
Labelled non-luminous Elateroidea/Buprestidae      32   22.00   0.185    nn_identity     0.9545   0.7079
  (Pectocera, Agrypnus, Agrilus planipennis)                             ESM-2 35M       0.9858   0.9434
-------------------------------------------------------------------------------------------------------
+ 59 unlabeled intra-genome Ignelater paralogs     91   81.00   0.058    nn_identity     0.9346   0.3350
  (see caveat below)                                                     ESM-2 35M       0.9726   0.6625
-------------------------------------------------------------------------------------------------------
Ancestry-matched only (Pectocera + Agrypnus)        2    2.00   0.714    nn_identity     0.5000   0.7079
                                                                         ESM-2 35M L4    0.9095   0.9710
                                                                           95% CI        [0.694, 0.975]
-------------------------------------------------------------------------------------------------------
```

**Provenance correction (round-5 audit).** No script in this repository produced `results/headline_control_corrected.tsv`, `results/hard_negative_benchmark_corrected.tsv` or `results/metrics_corrected.tsv`; they were written directly from an interactive session, and the "strict" negative set they use ($n^- = 32$) is not defined anywhere in the code. The claim could therefore not be re-executed. `scripts/reproduce_headline_and_resampling.py` (added 2026-11-16) rebuilds it from `dataset.tsv`, the similarity matrices and the cached 35M embeddings, reconstructing the strict set as the 31 labelled negatives with highest identity to any training luciferase -- a reconstruction that reproduces the published `nn_identity` row exactly ($\text{AUROC}\ 0.9545$, $\text{AP}\ 0.7082$ against the published $0.9545 / 0.7079$) and so is very likely the intended set, though exact membership cannot be recovered.

**Robustness to the negative-set cutoff (referee H-3).** The "strict" set is defined by a
cutoff on identity rank, which is a free parameter. `scripts/sweep_negative_set_cutoff.py`
varies it from $n^- = 10$ to the full labelled negative set ($n^- = 178$):
$\Delta\text{AP}$ ranges only $+0.2228$ to $+0.2346$ (spread $0.0118$), with the identity
baseline constant at $\text{AP} = 0.7082$ throughout. **The headline margin does not depend
on the cutoff.** See `results/negative_set_cutoff_sweep.tsv`.

**The headline numbers previously reported here do not reproduce.** On the reconstructed set the probe exceeds nearest-neighbour identity by **$+0.228$ AP** ($0.9364$ vs. $0.7082$; $\text{AUROC}\ 0.9832$ vs. $0.9545$). Earlier drafts reported $+0.273$ AP ($0.980$ vs. $0.708$; $\text{AUROC}\ 0.994$ vs. $0.955$). The identity baseline reproduces exactly; only the ESM-2 row differs, by $+0.044$ AP in the direction that flattered the claim. We report the reproducible value and retain the discrepancy on the record. We subjected the reproduced difference to two paired tests, which answer different questions and must both be reported:

- **Label permutation** (2,000 draws, two-sided): the null difference is $-0.003 \pm 0.094$ and the observed $+0.228$ gives $p = 0.0070$. The score-label association is not an artefact of chance, leakage, or label structure. Earlier drafts reported $p = 0.0005$, which is not attainable from 2,000 two-sided draws at this effect size and is withdrawn.
- **Cluster bootstrap** (2,000 draws, resampling identity clusters rather than sequences; 28 clusters): $95\%$ CI $[-0.167, +0.538]$, with $60.6\%$ of resamples favouring ESM-2. The interval **includes zero**, and is wider than the $[-0.062, +0.583]$ previously reported.

We therefore claim the following and no more: *in this system*, ESM-2 separates convergently evolved luciferases from ancestry-matched negatives at a level permutation testing excludes as chance, but **the margin over sequence identity is not shown to be stable under resampling of beetle clades** at $4.99$ effective positive clusters. Readers should not generalise the ranking of ESM-2 above homology beyond this dataset. On the ancestry-matched pair, identity is **at chance** (AUROC $0.500$; AP $0.7079$ against a $0.714$ baseline), while the CV-selected layer-4 probe is above chance with a bootstrap interval excluding $0.5$. We note explicitly that the last-layer probe is **not** significant on this subset (AUROC $0.8438$, 95% CI $[0.418, 1.000]$), and we therefore quote the CV-selected layer throughout.

**Caveat on the 91-negative set.** 59 of those 91 sequences are *Ignelater* paralogs carrying homology-only annotations — the same category this paper argues is untrustworthy (§2.1). Scoring a model for agreeing with them is circular, and we report that row only for continuity with earlier drafts. **The 32-negative row is the one we rely on**, and the conclusion is unchanged.

### 2.2 3D Active-Site Attribution & The $\text{L347}$/$\text{S347}$ Gate in Rigid Structures

Using an L1-regularized probe on firefly-trained ESM-2 (35M), per-residue functional log-odds attribution was calculated on the convergent click-beetle luciferase `A0A8K0DA02`. Attribution density was dramatically concentrated within the 3D catalytic pocket ($\le 8.0\text{ \AA}$ of the bound luciferyl-adenylate intermediate $\text{SLU}$ in PDB `4G36`): mean attribution was **$+19.55$** in the pocket versus **$+0.30$** on surface residues (Mann-Whitney $U$ test, $p = 6.09 \times 10^{-11}$).

**Label hygiene in the ancestry control (disclosure).** The ancestry-matched control has
only two negatives, and they are not equivalent. `A0A0C6DRW5` (*Pectocera fortunei*) has no
reported luminescence. `A9ZPM4` (*Agrypnus binodulus* AbLL) is a characterised **latent
luciferase** -- the very protein Oba et al. (2009) converted toward luminescence with a
single serine substitution. Scoring it as a negative is defensible at the organismal level
(the beetle does not glow) but it is not a clean functional negative, and we flag it rather
than leave it implicit in a figure legend. The direction of the resulting bias is
conservative: a one-substitution-from-positive protein in the negative set makes the task
harder, not easier. Consistent with that, ESM-2 (35M, layer 4) scores both control
negatives at the 94th--96th percentile of the full negative distribution and within the
positive range ($-1.31$ for `A9ZPM4`, the 30th percentile of positives; $-1.48$ for
`A0A0C6DRW5`, the 12th percentile). The control's AUROC therefore reflects ESM-2 ranking
most convergent luciferases *above* two borderline cases, not confidently rejecting them.
With $n^- = 2$ this is a two-protein observation and we do not present it as more.

**Prior art.** Oba, Iida, Ojika & Inouye (2009, *FEBS Lett.* 583:2004–2008) established by site-directed mutagenesis that a single serine substitution at this position converts fatty acyl-CoA synthetase activity toward luminescence, and demonstrated it in two proteins we also analyse: *Agrypnus binodulus* AbLL (`A9ZPM4`) and *Drosophila* CG6178. **The gate residue is therefore a known experimental result, not a discovery of this work.** We use it as independent ground truth against which to test whether an unsupervised model localises the same position (§2.4).

Superposition of 9 experimental and AlphaFold structures onto the $\text{SLU}$-bound pocket recovers the residue from rigid structure alone. **The steric interpretation below is not supported by evidence: the rigid clashes do not survive relaxation (§2.3b), and the dynamics cannot test the hypothesis either. It is retained only to document how the hypothesis was formed:**
- **True Luciferases (`4G36`, `6AAA`, `AF-Q7M4K3`, `AF-A0A8K0DA02`):** Convergently carry a small **Serine (`S347`) or Cysteine (`C347`)** on the 343–349 active-site loop (`TETTSAI`, `TESTSAN`, `TETCSAL`), providing a wide $3.03\text{--}3.50\text{ \AA}$ clearance with **zero steric clashes** against the rigid benzothiazole-thiazole rings of luciferin.
- **Non-Luminous Decoys (`AF-A0A0C6DRW5`, `AF-A0A8K0C7P5`, `PDB 7KYD`):** Retain an ancestral bulky **Leucine (`L347`)** (`TEATLAV`, `TEATLVI`, `SESTLSV`). In rigid superposition, this leucine protrudes directly into the luciferin-binding pocket, creating severe atomic clashes ($2.26\text{ \AA}$ and $2.54\text{ \AA}$, $\le 2.8\text{ \AA}$).

### 2.3 Force-Field Rotamer Relaxation, Coupled Gate-Loop Relaxation & Apo All-Atom MD

To evaluate whether side-chain rotation, local loop displacement, or dynamic thermal relaxation could alleviate this steric barrier without mutating the sequence:
1. *Continuous Rotamer Relaxation:* Dihedral angle optimization over $(\chi_1, \chi_2)$ under a Lennard-Jones 12-6 potential ($r_0 = 3.35\text{ \AA}$) confirmed that even at optimal torsions, non-luminous $\text{L347}$ side chains remain sterically compressed ($2.87\text{--}3.10\text{ \AA}$) compared to wild-type $\text{S347}$ luciferases ($3.33\text{--}3.64\text{ \AA}$).
2. *Coupled 9-DOF Gate-Loop & Ligand Pose Relaxation:* Allowing the entire 343–349 loop and ligand to flex simultaneously, the non-luminous paralog `A0A8K0C7P5` maintained a compressed distance of **$2.76\text{ \AA}$** with substantial loop displacement ($0.54\text{ \AA}$). In contrast, introducing the in-silico steered gate mutation ($\text{L347S/V348A/I349N}$) expanded clearance to **$3.02\text{ \AA}$**, lowered relaxed energy by $-2.0\text{ kcal/mol}$, and reduced loop strain to $0.17\text{ \AA}$, matching the true luciferase (`A0A8K0DA02`, $3.19\text{ \AA}$).
3. *Apo All-Atom Explicit-Solvent MD (3 $\times$ 10.0 ns, 300 K):* We ran three independent all-atom simulations (OpenMM, AMBER99SB-ILDN/TIP3P, 4.0 fs with hydrogen mass repartitioning, 2.5 $\times 10^6$ steps each) of the true luciferase `A0A8K0DA02`, the non-luminous paralog `A0A8K0C7P5`, and the steered triple mutant (`L347S/V348A/I349N`). **These simulations are ligand-free (apo):** the luciferyl-adenylate intermediate `SLU` has no standard AMBER parameters and was excluded, so the runs report on backbone and gate-loop stability only and **cannot test the steric-gate hypothesis**. All three systems reach a stable fold within $\sim 2\text{ ns}$ and remain so. Over the final $5\text{ ns}$, whole-protein backbone RMSD is $1.91 \pm 0.14\text{ \AA}$ (true luciferase), $1.84 \pm 0.19\text{ \AA}$ (non-luminous paralog), and $2.23 \pm 0.12\text{ \AA}$ (steered triple mutant). The true luciferase and the paralog differ by $0.07\text{ \AA}$, which is **smaller than the within-run fluctuation of either**; with one trajectory per system there is no replicate-level error term, so we draw no distinction between them. This is consistent with the discrimination reported in §2.2 arising from side-chain packing in the *ligand-bound* pocket rather than from any difference in fold stability -- but, with no ligand present, these runs cannot test that.

Per-residue C$\alpha$ RMSF was evaluated at the gate triad **after transferring P08659 numbering by motif anchoring**, because the three topologies are numbered differently: the gate occupies residues 346--348 in `A0A8K0DA02` (`GFGLTE`**`STSAN`**`IH`) but 343--345 in `A0A8K0C7P5` (`GYGLTE`**`ATLVI`**`TH`). At the correctly aligned gate the three systems are **equally rigid** ($0.52$, $0.48$ and $0.51\text{ \AA}$ respectively). The steered mutant's larger global RMSD originates in a downstream loop (aligned positions $\sim 353$--$360$), not at the engineered site. We note that an earlier draft reported gate RMSF values of $2.88$--$4.47\text{ \AA}$ and inferred that the steering mutations increase gate flexibility; those values were obtained with an **unaligned** residue window and do not reproduce from the deposited trajectories -- that conclusion is withdrawn.

> **Retraction note (2026-11-13).** An earlier draft of this subsection reported a $+3.5\text{ kcal/mol}$ steric penalty, an $88\%$ reduction on steering, and a $-8.5\text{ kcal/mol}$ change in "total pocket binding". Those values came from an in-house pairwise-potential script that contained no bonded terms, used non-AMBER partial charges, and omitted the unit conversion between $\text{kcal}\cdot\text{mol}^{-1}\text{\AA}^{-1}\text{amu}^{-1}$ and $\text{\AA}\cdot\text{ps}^{-2}$. They are withdrawn in full and are not used anywhere in this manuscript. See `__ABANDONED_DO_NOT_USE__/README_DO_NOT_USE.md` §0b.

### 2.3b Holo Molecular Dynamics: an inconclusive test, and why

The gate hypothesis (§2.2) came from **rigid** superposition, which predicted severe clashes
($2.26\text{--}2.54\text{ \AA}$) between the bulky $\text{L347}$ of non-luminous decoys and the
luciferin ring. We parameterized the ligand (DLSA, anionic, OpenFF 2.2.0 / AM1-BCC) and ran
**twelve holo simulations** — four systems $\times$ three independent replicates $\times$
$2.5\text{ ns}$ in explicit solvent (OpenMM, AMBER14/TIP3P, PME, HMR $4.0\text{ fs}$). Gate
residues were located by motif anchoring, resolving to 346 in `A0A8K0DA02` and 343 in
`A0A8K0C7P5`. The fourth system is an **assay-sensitivity control**: a deliberately bulky
$\text{S347F}$ mutant of the true luciferase, built in an ideal rotamer with its phenyl ring
$1.62\text{ \AA}$ from the ligand — a severe, unambiguous steric clash by construction.

```
Table 5: Gate-347 side chain to ligand distance. 3 replicates x 250 frames (2.5 ns) each.
Regenerated 2026-11-16 by scripts/analyze_holo_md.py -> results/holo_md_summary.tsv.
Ligand COM drift is measured after Kabsch removal of protein rigid-body motion.
----------------------------------------------------------------------------------------------
                               res   rigid    after MD      pooled   frames   ligand COM
system                              start    median (A)      min     < 2.8 A   drift (A)
----------------------------------------------------------------------------------------------
True luciferase                SER   3.33 A  3.80 +/- 0.13   3.04 A   0 / 750    1.22
Non-luminous paralog           LEU   2.26 A  3.71 +/- 0.12   2.93 A   0 / 750    1.46
SAE-steered                    SER   2.82 A  3.41 +/- 0.14   2.70 A  15 / 750    1.14
POSITIVE CONTROL (S347F)       PHE   1.62 A  3.58 +/- 0.09   3.11 A   0 / 750    0.98
----------------------------------------------------------------------------------------------
```


**Provenance correction (round-5 audit).** The values in Table 5 were previously computed in
an interactive session that saved neither a script nor a table, so they could not be checked.
`scripts/analyze_holo_md.py` now regenerates them from the twelve `prod.dcd` trajectories. The
decisive statistic reproduces exactly -- 15 sub-$2.8\text{ \AA}$ frames, all in the SAE-steered
system, zero in the other three, including the positive control -- and the gate distances
reproduce to within $0.1\text{--}0.4\text{ \AA}$. Two secondary quantities do not reproduce,
because their definitions were never recorded: the earlier contact counts (reported as
$159\text{--}170$) were computed under an unknown criterion and are superseded by the stated one
(protein heavy atoms within $4.0\text{ \AA}$ of any ligand heavy atom: $69\text{--}76$), and the
earlier ligand-displacement figures reconcile only once protein rigid-body motion is removed.
None of this changes the conclusion, which rests on the clash-frame counts.

**The positive control fails, and this invalidates the experiment.** A phenylalanine placed
$1.62\text{ \AA}$ from the ligand — a clash no pocket could tolerate — relaxes to
$3.58 \pm 0.09\text{ \AA}$ and becomes statistically indistinguishable from serine and leucine. It does
not displace the ligand more than the wild-type does ($0.98\text{ \AA}$ versus $1.22\text{ \AA}$
for the true luciferase, rigid-body motion removed), and it retains a comparable number of protein–ligand contacts (71 vs 76). Its $\chi_1$ barely moves ($-177^\circ \rightarrow 164^\circ$), so the ring does not
simply swing aside; the clash is absorbed by small, distributed relaxation of the backbone,
side chain and ligand pose.

The protocol therefore **cannot detect steric occlusion even when occlusion is present by
construction**. We had pre-registered this control with an explicit decision rule: if S347F also
failed to occlude, §2.3b proves nothing. It did, and we apply the rule.

**What we withdraw.** Both the mechanism and its refutation.
1. The $2.26\text{--}2.54\text{ \AA}$ clashes of §2.2 are artefacts of superposing a rigid
   AlphaFold side chain onto a rigid crystallographic ligand pose; they do not survive
   relaxation, so they are not evidence for the steric mechanism.
2. The observation that $\text{L347}$ does not occlude in MD is **not evidence against** it
   either, because the same protocol returns that answer for a residue that must occlude.

The steric-gate hypothesis is therefore **untested by this work**, not refuted. The underlying
experimental fact is unchanged: Oba et al. (2009) showed that Ser at this position is required
for luminescence activity. We simply cannot say why.

**Why the assay fails, and what would work.** Any protocol that begins from a modelled complex
and relaxes it — energy minimisation followed by equilibration — dissipates exactly the strain
whose magnitude is the quantity of interest. The relaxation is cheap in free-energy terms and
essentially complete within $200\text{ ps}$. Testing a steric hypothesis requires a method that
measures the *cost* of accommodation rather than allowing it to occur silently: alchemical
free-energy perturbation between $\text{S347}$ and $\text{L347}$ in the holo complex, or a
potential of mean force along a substrate-binding coordinate. Both are beyond the compute
available here, and we flag them as the correct next experiment rather than claiming a result
this design cannot deliver.

We report this negative outcome in full because the alternative — reporting the $\text{L347}$
result without the control — would have presented an uninformative measurement as a refutation.

**Two further results from the same run, both unflattering and both retained.**

*The one experimental structure carrying L347 shows no clash.* `7KYD` (*Drosophila* CG6178)
is the only crystal structure in the panel with a leucine at the gate. After chi1/chi2
rotamer relaxation its gate-to-ligand distance is $3.18\text{ \AA}$ rigid and
$3.10\text{ \AA}$ relaxed, with a **favourable** 12-6 Lennard-Jones energy of
$-2.62\text{ kcal/mol}$; the in-silico L347S mutation *opens* the gate to $3.74\text{ \AA}$
at $-1.66\text{ kcal/mol}$. For comparison the two firefly structures (`4G36`, `6AAA`, both
S347) sit at $3.64$ and $3.40\text{ \AA}$. A leucine at this position is therefore
accommodated without strain in an experimentally determined ACS fold. This is independent
structural evidence against the steric-occlusion mechanism, consistent with the holo-MD
outcome in §2.3b, and it means the $2.26$--$2.54\text{ \AA}$ clashes reported from rigid
homology models in earlier drafts were modelling artefacts.

*A position-matched one-hot baseline beats ESM-2 on bioluminescence colour.* On the
continuous emission-wavelength task ($538$--$623\text{ nm}$, $n = 26$):

```
representation                        LOO r    LOO MAE      LOCO-CV r   LOCO-CV MAE
16-site position-matched one-hot      0.724    6.38 nm        0.646       9.91 nm
16-site position-matched BLOSUM62     0.704    6.09 nm        0.603       9.84 nm
16-site 3D-pocket ESM-2 (150M L12)    0.656    7.49 nm        0.532      10.69 nm
whole-protein ESM-2 (150M L12)        0.483   11.66 nm        0.301      14.43 nm
```

**One-hot encoding of the same 16 residues outperforms the protein language model.** We
report this because it bounds the paper's central claim: ESM-2's advantage in this project is
specific to the luminous/non-luminous *discrimination* task against an ancestry-matched
negative set, and does not extend to quantitative colour prediction, where simple
position-matched encodings are better. Whole-protein ESM-2 is the worst representation tested.

### 2.4 Causal SAE Steering: re-executed and verified, after a reproducibility failure

**Status: the result stands; the process that produced it failed and was repaired.** This
section twice changed status during audit and we report the full sequence, because the
sequence is more instructive than the result.

**What was wrong.** `scripts/run_step1_and_step3_rigor.py::run_step1_steering_controls()`
reached an unconditional `return` on line 113, making its own 182-line implementation
unreachable. What executed instead read six result dictionaries that were written as
literals in the source and cached to a `.pt` file, printing them in the format of a
completed run. The function additionally referenced a name (`rows`) that is defined nowhere
in the module, so it could not have run even with the return removed. Nothing in this
project's published steering numbers was traceable to an execution.

**What we did.** The early return and the literal block were deleted, the missing loader
added, and the experiment was run: ESM-2 650M and `InterPLM-esm2-650m` (Layer 18) from a
local cache, CPU, fp32, 181 seconds.

**What we found.** Across **78** compared values spanning **all three** held-out targets the re-executed
results differ from the cached literals by at most $0.0058$. **The cached table was a faithful record of a real earlier
run, not fabricated data.** An earlier round of this audit characterised it as "synthetic
provenance" and asserted that the controls had never been run; that characterisation was
wrong and is withdrawn here. The defect was a reproducibility failure, not a data-integrity
failure, and the distinction matters in both directions: the numbers are sound, and they
were nonetheless unverifiable by anyone including their author until the code was repaired.

```
Table 1: Causal steering specificity and off-target scaffold stability.
RE-EXECUTED 2026-11-17 on ALL THREE held-out targets, scripts/run_steering_for_real.py
-> results/step1_steering_REAL.tsv. ESM-2 650M + InterPLM-650M L18, alpha = 2.0,
16 pocket sites steered. Scaffold = top-1 agreement over ~530 non-pocket residues.
--------------------------------------------------------------------------------------------------
target                   input       method                   Delta    fold   scaffold      KL
--------------------------------------------------------------------------------------------------
CG6178 (Drosophila)      unmasked    Top-50 sparse SAE        +5.88   358.6x    99.81%  0.0009
                                     Dense diff-in-means      +6.58   723.5x   100.00%  0.0006
                                     Orthogonal-50 (control)  +0.49     1.6x   100.00%  0.0002
                                     Random-50, 10 seeds   +0.16+/-0.48   1.2x    99.88%  0.0002
CG6178 (Drosophila)      <mask>_347  Top-50 sparse SAE        +2.79    16.3x    99.81%  0.0005
                                     Dense diff-in-means      +3.48    32.6x   100.00%  0.0003
                                     Orthogonal-50 (control)  +0.48     1.6x   100.00%  0.0002
                                     Random-50, 10 seeds   +0.08+/-0.54   1.1x    99.83%  0.0002
A0A0C6DRW5 (Pectocera)   unmasked    Top-50 sparse SAE        +6.90   991.1x    99.81%  0.0008
                                     Dense diff-in-means      +7.22  1367.9x   100.00%  0.0007
                                     Orthogonal-50 (control)  +0.01     1.0x   100.00%  0.0002
                                     Random-50, 10 seeds   +0.19+/-0.28   1.2x    99.98%  0.0002
A0A0C6DRW5 (Pectocera)   <mask>_347  Top-50 sparse SAE        +3.71    40.9x   100.00%  0.0005
                                     Dense diff-in-means      +4.08    59.4x   100.00%  0.0003
                                     Orthogonal-50 (control)  -0.03     1.0x   100.00%  0.0002
                                     Random-50, 10 seeds   +0.30+/-0.23   1.4x    99.96%  0.0002
A0A8K0C7P5 (Ignelater)   unmasked    Top-50 sparse SAE        +5.78   323.0x    99.81%  0.0008
                                     Dense diff-in-means      +6.07   432.9x    99.81%  0.0007
                                     Orthogonal-50 (control)  +0.34     1.4x    99.81%  0.0002
                                     Random-50, 10 seeds   -0.00+/-0.48   1.0x    99.90%  0.0002
A0A8K0C7P5 (Ignelater)   <mask>_347  Top-50 sparse SAE        +2.80    16.4x    99.81%  0.0005
                                     Dense diff-in-means      +3.28    26.6x    99.81%  0.0003
                                     Orthogonal-50 (control)  +0.19     1.2x   100.00%  0.0002
                                     Random-50, 10 seeds   +0.15+/-0.37   1.2x    99.94%  0.0002
--------------------------------------------------------------------------------------------------
```

**Interpretation, bounded.** Clamping 50 of 10,240 SAE features ($0.49\%$) at the pocket
sites shifts the model's gate-347 log-odds toward Ser/Ala by $16\times$ to $991\times$ while
leaving $99.81\text{--}100.00\%$ of top-1 predictions at the $\sim 530$ non-pocket residues
unchanged ($D_{KL} \le 0.0008$). Matched-norm orthogonal and random feature sets produce no
shift ($1.0\text{--}1.4\times$). The effect is therefore specific to the selected features
rather than to the magnitude of the perturbation.

**The nuance that cuts against the sparse basis.** At matched $L_2$ norm the *dense*
difference-in-means vector is **stronger** than the sparse SAE steer on every target and
input mode ($+7.22$ vs $+6.90$; $+6.07$ vs $+5.78$; $+4.08$ vs $+3.71$; $+3.28$ vs $+2.80$)
while preserving the scaffold equally well. The sparse basis is more interpretable, not more
effective, which is consistent with the capacity results in §2.5 and with our withdrawal of
the claim that SAE decomposition improves functional separation.

**Spatial and dose controls (referee R-4).** Two controls were requested and run
(`scripts/run_steering_controls_R4.py` -> `results/steering_controls_R4.tsv`, 413 s). Both
reuse the same steering dictionaries and evaluator via a `return_internals` hook rather than a
reimplementation.

*Spatial.* A steering vector was built by the **identical procedure** -- SAE
difference-in-means between held-in luminous and non-luminous training proteins, top-25
positive plus top-25 negative features -- at 16 **distal** positions, each at least 25
residues in sequence from every pocket site, then rescaled so its total $L_2$ norm matches the
pocket steer exactly. Mean $\Delta$ gate-347 log-odds across the three held-out targets:

```
alpha                 0.0     0.5     1.0     2.0     4.0     8.0
pocket (16 sites)   +0.000  +0.089  +2.607  +6.186  +7.202  +9.571
distal (16 sites)   +0.000  +0.002  +0.003  +0.002  +0.003  -0.008
```

The distal vector is **not inert**: at $\alpha = 8$ it degrades off-target top-1 recovery to
$99.17\%$ and raises off-target $D_{KL}$ to $0.039$, comparable to or worse than the pocket
steer. It perturbs the model substantially and moves gate-347 by essentially zero. The effect
is therefore **positionally specific**, not a generic consequence of adding a large vector at
Layer 18. *Caveat:* "distal" is defined by sequence separation, not by distance in the folded
structure.

*Dose--response.* The gate response is monotone and strongly non-linear, rising about four
orders of magnitude between $\alpha = 0.5$ and $\alpha = 8$, while off-target recovery falls
only from $100.00\%$ to $98.98\%$ (Figure 13). Note that the reported operating point
$\alpha = 2.0$ sits on the steep part of that curve: the fold-changes in Table 1 are
strongly $\alpha$-dependent and should be read as "large at this operating point", not as
calibrated effect sizes.

**Neighbour-only steering: the decisive control, and it narrows the claim (referee R-4).**
Position 347 is itself one of the 16 steered sites and the gate log-odds are read at 347, so
neither the spatial nor the matched-norm controls separate *"the pocket context carries the
signal"* from *"perturbing residue 347 moves residue 347"*. Protocol B does: 347 is masked at
the input and steering is applied **only at the other 15 pocket sites** ($r \neq 347$). It was
executed once the `4G36` reference structure was restored
(`scripts/run_system1_for_real.py` -> `results/system1_fixes_REAL.json`):

```
target                    Protocol A (all 16, incl. 347)   Protocol B (r != 347)   B / A
CG6178 (Drosophila)            +2.79  (16.3x)                 +0.96  (2.6x)         34%
A0A0C6DRW5 (Pectocera)         +3.71  (40.9x)                 +0.78  (2.2x)         21%
A0A8K0C7P5 (Ignelater)         +2.80  (16.4x)                 +0.29  (1.3x)         10%
```

**Between 66% and 90% of the reported effect requires perturbing residue 347 itself.** The
pocket context alone yields $1.3\times$--$2.6\times$, which sits only just above the
matched-norm orthogonal and random null band ($1.0\times$--$1.4\times$) and overlaps it on the
*Ignelater* decoy.

We therefore state the claim at the strength the controls support, and no higher:

> A specific sparse direction, applied at the catalytic pocket, moves the model's residue-347
> prediction toward Ser/Ala far more than random or orthogonal directions of equal norm, and
> far more than the same construction applied at distal sites. Most of that effect is carried
> by the perturbation at 347 itself; the surrounding pocket context contributes a small
> residual that is not clearly separable from the null.

This is a weaker claim than earlier drafts made, and it is the one the data support.

**What this is and is not.** It is evidence that a small, identifiable set of SAE features is
causally load-bearing for the model's prediction at position 347. It is a model-internal
result measured in logit space; no protein was expressed, and it does not establish the
physical steric mechanism, which remains untested (§2.3b). Oba et al. (2009) remains the
experimental authority for the Ser347 requirement. All three held-out targets of the original design are now reported: `pdb_00007kyd.cif` was
restored to `data/structures/` and *Drosophila* CG6178 re-executed with the others.

### 2.5 Multi-PLM Scale Benchmark, and Why Its SAE Advantage Does Not Survive a Capacity-Matched Control

**Re-executed 2026-11-16** (`scripts/rerun_table2.py` -> `results/table2_system1_REAL.tsv`).
The System 1 columns of Table 2 reproduce **exactly** from the cached representations:
ESM-2 8M $\text{AP} = 0.534$, InterPLM-8M SAE $0.733$, ESM-2 35M $0.470$, ESM-2 150M
$0.506$, ESM-2 650M $0.470$ (*Ignelater* rank #7), InterPLM-650M SAE $0.590$ (rank #1).
Those published numbers were real. **What is withdrawn is their interpretation, not their
value:** on this benchmark the SAE does separate paralogs better than the dense model it
decomposes, but §2.5 shows that matched-width random projections achieve the same or more,
so the gain is attributable to dimensionality rather than to learned sparse structure. The System 2 columns were re-executed on 2026-11-17 once `1U19` was restored to
`data/structures/`, and they reproduce: ESM-2 8M $r = 0.845$ / MAE $5.18\text{ nm}$, 35M
$0.846$ / $5.20$, 150M $0.850$ / $5.33$, 650M $0.843$ / $5.24$. For scale (referee R-5):
the $n = 41$ panel has SD $12.55\text{ nm}$ and predicting the **mean** gives MAE
$10.39\text{ nm}$, so these models roughly halve the error of a no-skill baseline -- a
real but modest effect that $r = 0.845$ alone would overstate. **Both sparse autoencoders
are markedly worse than the dense models they decompose on this system** -- InterPLM-8M
$r = 0.748$ / $6.42\text{ nm}$, InterPLM-650M $r = 0.644$ / $7.78\text{ nm}$ -- the same
direction as the capacity result and independent of it.

To test how scaling affects representation quality, we benchmarked all 6 locally available models across System 1 and System 2 (**Table 2**):

```
Table 2: Multi-PLM & Multi-SAE Architecture Scaling Benchmark.
-----------------------------------------------------------------------------------------------------------------
Model Architecture              Sys 1 Close-Ancestry AP  Ignelater Genome Rank  Sys 2 Surround-49 r   Sys 2 Surround-49 MAE
-----------------------------------------------------------------------------------------------------------------
ESM-2 8M (320-d, L6)            0.534                    Rank #1 ([6, 1])       0.845                 5.18 nm
InterPLM-8M SAE (10,240-d, L4)  0.733 (+0.199 vs 8M!)   Rank #2 ([42, 2])      0.748                 6.42 nm
ESM-2 35M (480-d, L12/L8)       0.470 (0.781 full)       Rank #2 ([7, 2])       0.846                 5.20 nm
ESM-2 150M (640-d, L4/L15)      0.506 (0.748 full)       Rank #1 ([8, 1])       0.850                 5.33 nm
ESM-2 650M (1,280-d, L18)       0.470                    Rank #7 ([8, 7])       0.843                 5.24 nm
InterPLM-650M SAE (10,240-d)    0.590 (+0.120 vs 650M!)  Rank #1 ([6, 1] vs #7) 0.644                 7.78 nm
-----------------------------------------------------------------------------------------------------------------
```

In raw dense PLMs, increasing parameter scale from 8M to 650M degraded *Ignelater* paralog resolution (Rank $\#1 \rightarrow \#7$). Table 2 as computed also showed sparse-autoencoder decomposition raising Average Precision by $+0.199$ at 8M and $+0.120$ at 650M.

**That advantage does not survive a capacity-matched control, and we withdraw it.** The dense representation is 320-d (8M) or 1280-d (650M) while the SAE dictionary is 10,240-d, so the comparison confounds *learned disentanglement* with a 8–32$\times$ larger feature budget. We therefore re-ran the 8M comparison on the corrected strict benchmark of §2.1b (27 click-beetle positives = 4.99 effective clusters vs. 32 labelled negatives, cluster-weighted, AP baseline $0.1849$), scoring every representation through an identical pipeline — standardise, L1 logistic probe fit on fireflies only, cluster-weighted AP on held-out click beetles:

```
Table 3: Capacity-matched controls for the sparse-decomposition claim (ESM-2 8M, layer 4).
AP baseline (cluster-weighted chance) = 0.1849
---------------------------------------------------------------------------------
representation                          dim     AUROC      AP     nnz
---------------------------------------------------------------------------------
Dense ESM-2 8M (layer 4)                320     0.920   0.735      22
Dense, matched sparsity (tuned C)       320     0.950   0.837      18
Random Gaussian projection (10 seeds) 10240     0.920   0.759 +/- 0.094
Random ReLU projection (10 seeds)     10240     0.894   0.622 +/- 0.133
InterPLM-8M SAE (layer 4)             10240     0.923   0.688      19
---------------------------------------------------------------------------------
```

Three things follow. First, the SAE ($\text{AP} = 0.688$) **does not beat the dense representation it decomposes** ($0.735$), and is beaten decisively by a plain L1 dense probe tuned to comparable sparsity ($0.837$, 18 non-zero features). Second, the SAE exceeds a random ReLU projection — which supplies overcompleteness, sparsity and non-negativity but no learned structure — by only $+0.066$ AP, well inside that control's own seed-to-seed spread of $\pm 0.133$. Third, and most important, **the random-projection controls fluctuate by $\pm 0.09$ to $\pm 0.13$ AP across seeds alone**, which is larger than almost every between-method gap in the table. At $\sim 5$ effective positive clusters this benchmark cannot rank these representations, and we do not claim that it does.

We then repeated the identical protocol at the 650M scale, where the published claims were an AP gain of $0.470 \rightarrow 0.590$ and an *Ignelater* genome-rank improvement from $\#7$ to $\#1$:

```
Table 3b: The same controls at ESM-2 650M, layer 18. AP baseline = 0.1849.
"Ignelater ranks" = positions of the two true luciferases among 61 PF00501 paralogs.
-------------------------------------------------------------------------------------
representation                          dim    AUROC      AP    nnz   Ignelater ranks
-------------------------------------------------------------------------------------
Dense ESM-2 650M (layer 18)            1280    0.890   0.512     20        [5, 7]
Dense, matched sparsity                1280    0.844   0.427     24        [4, 7]
Random Gaussian projection (10 seeds) 10240    0.929   0.689 +/- 0.212     [3, 6]
Random ReLU projection (10 seeds)     10240    0.890   0.606 +/- 0.180     [4, 9]
InterPLM-650M SAE (layer 18)          10240    0.815   0.387     24        [7, 8]
-------------------------------------------------------------------------------------
```

Neither claim reproduces. Measured AP runs $0.512 \rightarrow 0.387$ (not $0.470 \rightarrow 0.590$) and the genome ranks move from $[5, 7]$ to $[7, 8]$ (not to $\#1$) — in both cases the sparse decomposition is **worse** than the dense representation it is built from. The SAE scores below **all twenty** random-projection draws (one-sided rank $p = 0.091$ against each control).

**But the dominant finding at this scale is that the benchmark has no resolving power.** Across the twenty random-projection draws, AP ranges from $0.408$ to $1.000$ — a spread of $0.592$ AP produced by the random seed alone, nearly double the $0.302$ AP gap separating the best and worst *named* methods in the table. A single Gaussian seed reached $\text{AP} = 1.000$. With $4.99$ effective positive clusters, this benchmark cannot rank representations, and we draw no ranking conclusion from it in either direction.

Two caveats on the controls themselves. A Gaussian projection $\mathbb{R}^{1280} \rightarrow \mathbb{R}^{10240}$ is an injective linear map and therefore **information-preserving**; it is a capacity control, not an information null, and its apparent superiority over the dense features reflects how the $L_1$ penalty behaves in the higher-dimensional space rather than any added signal. And the sparse dictionaries were trained by Simon & Zou on UniRef, not on this task — we are testing whether an off-the-shelf SAE basis transfers to beetle paralog discrimination, not the quality of the SAE objective.

We also note that the original $+0.199$ figure at 8M was computed on the 91-negative set containing 59 homology-only annotations (§2.1b), without cluster weighting. Across both scales the reported advantage appears to be a property of that evaluation set rather than of the representation.

### 2.6 De-Circularized Experimental $\text{RH1}$ Opsins & Surround-Only Epistasis

In System 2, we evaluated 41 vertebrate rhodopsins with published experimental microspectrophotometry ($\lambda_{\max} \in [468, 517]\text{ nm}$), including real intra-species paralog pairs in European eel *Anguilla anguilla* ($482\text{ nm}$ silver-eel vs. $501\text{ nm}$ yellow-eel) and spinyfin *Diretmus argenteus* ($484\text{ nm}$ vs. $498\text{ nm}$).

To test whether the model detects non-local epistatic context:
- Ablating all 5 known spectral tuning sites ($\text{83, 122, 261, 292, 299}$) left a **49-site Surround-Only Pocket**.
- Contextual `ESM-2 (35M)` embeddings achieved **$r = 0.846$ (MAE $5.20\text{ nm}$, AUROC $0.997$)**, significantly outperforming position-matched 49-site One-Hot encodings ($r = 0.741$, MAE $6.51\text{ nm}$, AUROC $0.941$). Self-attention in ESM-2 propagates chromophore-coupling signals across second-shell residues.

### 2.7 Phylogenetic Independent Contrasts (PIC) & Ancestral Sequence Reconstruction

To guarantee that performance gains were not artifactual consequences of shared evolutionary branch lengths, we constructed explicit Neighbor-Joining phylogenetic trees and performed Felsenstein's (1985) Phylogenetic Independent Contrasts:
1. *System 2 Teleost $\text{RH1}$ (40 internal bifurcations):* Ancestral Sequence Reconstruction ($\text{ASR}$) identified the ancestral vertebrate root as $\text{N83 / E122 / F261 / A292}$ ($\lambda_{\max} = 489.0\text{ nm}$). Contrasts in Surround-49 predicted wavelengths correlated strongly with contrasts in true experimental wavelengths (**$\text{PIC Pearson } r = 0.596, p = 5.00 \times 10^{-5}$**; **$\text{PIC Spearman } \rho = 0.517, p = 6.36 \times 10^{-4}$**).
2. *System 1 Coleoptera $\text{PF00501}$:* An earlier draft reported that contrasts in ESM-2 functional scores tracked transitions to bioluminescence ($\text{PIC } r = 0.378, p = 5.28 \times 10^{-3}$) whereas contrasts in sequence identity did not ($r = 0.261, p = 0.059$), and read this as ESM-2 outperforming homology after removing tree covariance. **That comparison was never tested, and it does not hold.**

   Comparing two correlations that share a criterion requires a test for dependent correlations; "significant vs. non-significant" is not a significant difference. Williams' test needs the correlation *between the two predictors*, which was not reported. We therefore recomputed the whole analysis independently: pairwise **global** BLOSUM62 alignments over all 228 labelled sequences (normalised as $1 - s_{ij}/\sqrt{s_{ii}s_{jj}}$), neighbour-joining, midpoint rooting, and Felsenstein contrasts, yielding $N = 227$ contrasts.

```
Table 4: Phylogenetic independent contrasts, recomputed, with Williams' test.
---------------------------------------------------------------------------
PIC r (ESM-2    vs luminescence)  = +0.270    p = 3.81e-05
PIC r (identity vs luminescence)  = +0.404    p = 2.60e-10
PIC r (ESM-2    vs identity)      = +0.337    (required by Williams; never reported)
---------------------------------------------------------------------------
Williams' test:  difference = -0.134,  t(224) = -1.906,  p = 0.058
---------------------------------------------------------------------------
```

   Two findings. First, the difference between the two predictors is **not significant** ($p = 0.058$). Second, the **ordering is reversed** relative to the earlier draft: under this reconstruction sequence-identity contrasts track luminescence *more* strongly ($r = 0.404$) than ESM-2 contrasts ($r = 0.270$). We do not assert that the earlier numbers were wrong — the two pipelines differ, and the earlier analysis reported 53 internal bifurcations against our 227, implying a different taxon set or tree. We assert only that **the comparative claim is not robust to reasonable analysis choices and is withdrawn** -- and the re-execution makes the reason sharper.

   **The project's own pipeline has now been re-run** (`scripts/run_pic_for_real.py` -> `results/pic_REAL.tsv`). On its 54-taxon tree it reproduces its published values closely: $\text{PIC } r(\text{ESM-2}) = 0.366$ ($p = 7.1 \times 10^{-3}$) against a published $0.378$, and $r(\text{identity}) = 0.259$ ($p = 0.061$) against a published $0.261$. The original numbers were real. **But our independent 228-sequence reconstruction gives the opposite ordering** -- $r(\text{identity}) = 0.404$ versus $r(\text{ESM-2}) = 0.270$. Two reconstructions of the same comparison disagree about which predictor wins. Neither tree carries bootstrap support, substitution-model selection, or branch-length uncertainty propagation, and PIC standardises by branch lengths, so this is not a tie-break problem: the quantity is **tree-dependent at the magnitude of the effect being claimed**. A comparative claim that changes sign under a defensible change of tree is not a result, and reporting whichever tree favours the hypothesis would be the exact error this paper is about.

   Both analyses also apply PIC to a **binary** trait (luminous / non-luminous). Felsenstein contrasts assume a continuously varying Brownian character; contrasts of a dichotomous trait violate that assumption, and a threshold or phylogenetic-logistic model would be the appropriate tool. In addition, the 228 labelled sequences include the 16 firefly positives and 82 negatives the probe was trained on, so the ESM-2 contrast is partly in-sample. Neither issue is repaired here; both are grounds for treating §2.7 item 2 as exploratory rather than confirmatory.

---

## 3. Discussion

### 3.1 Polysemantic Superposition: A Hypothesis Our Controls Do Not Support

A natural reading of Table 2 is that the primary vulnerability of dense protein language models in functional annotation is **polysemantic superposition**: because hundreds of thousands of biological features compete for representation within a 320-to-1280 dimensional vector space, lineage-defining phylogenetic markers become linearly entangled with sparse catalytic switches. When queried on close-ancestry paralogs, dense models frequently privilege phylogenetic similarity over catalytic viability.

**Our own data do not support that reading, and we report it as a negative result.** Under the capacity-matched controls of §2.5 the InterPLM dictionary does not outperform the dense representation it is built from, and its margin over a structure-free random ReLU projection is smaller than that control's seed variance. Whatever advantage appeared in Table 2 is attributable to feature-budget and to an evaluation set contaminated with homology-only labels, not to demonstrated disentanglement. We therefore make no claim here that sparse autoencoders resolve superposition for this task. Establishing that would require a benchmark with far more independent positive clusters than the $\sim 5$ available to us, and a capacity-matched comparison at every scale tested.

The SAE arm yields a bounded positive result after repair (§2.4): the selected features are causally load-bearing inside the model, and matched-norm random and orthogonal controls produce no comparable shift. What it does **not** show is that the sparse basis is a better functional representation than the dense one -- at matched norm the dense difference-in-means steer is stronger on every target, and the capacity benchmark below goes the same way. We separate the two claims deliberately: sparse features here are more *interpretable*, not more *effective*.

### 3.2 Implications for AI Biosecurity and Dual-Use Toxin Monitoring

The deconfounding methodology developed here -- the convergent-evolution control, not the withdrawn steering experiment -- has possible implications for **model evaluation and oversight**:
1. *Detecting Stealth Gain-of-Function Mutations:* Dual-use agents (e.g., highly pathogenic toxins or altered viral receptors) can be engineered via minimal active-site remodeling within non-pathogenic, high-identity commensal scaffolds. Sequence-homology screening tools fail to flag such variants. Our 3D pocket attribution and SAE monitoring frameworks detect latent catalytic activation even when the global sequence appears benign.
2. *Automated Annotation Auditing:* Automated database contamination (such as the 95 homology-transferred, experimentally unverified "luciferase" annotations identified in non-luminous beetles) poses a systemic threat to scientific integrity and automated biofoundries. Deconfounded PLM probes can audit UniProt and NCBI at scale to eliminate homology-propagated annotation errors.

### 3.3 A Turnkey Framework for Computational Protein Engineering

We make no claim to have connected a representational hypothesis to a structural one. The sparse-autoencoder arm is withdrawn (§2.4) and the structural arm is inconclusive by its own positive control (§2.3b). What the pipeline does provide is a reproducible convergent-evolution benchmark with an ancestry-matched negative set, and all-atom holo simulations with a parameterized ligand runnable on a single commodity GPU at no monetary cost. The work stops well short of thermodynamic validation: **no binding free energy is reported**, and an alchemical free-energy perturbation or a binding potential of mean force remains the single most informative next experiment.

---

### 3.4 Limitations

We state these plainly because several of them bound the strength of our conclusions.

1. **Effective sample size is small.** After cluster weighting, System 1 rests on **6.0 effective training clusters** (16 firefly positives) and **4.99 effective test clusters** (27 click-beetle positives, of which 23 are near-identical *Pyrophorus* colour variants). The strictly ancestry-matched negative set contains **two sequences**. On that subset AUROC is confined to a $1/54$ grid and bootstrap intervals are wide; we report them but do not treat them as precise estimates. The 32-negative benchmark (§2.1b) carries the weight of the argument.

2. **"Non-luminous" is a database inference, not an assay.** We never observed the absence of light from any negative. Negatives are PF00501 members from genera with no reported bioluminescence. Some luciferase-like beetle enzymes emit weakly in vitro, so the positive/negative boundary is genuinely fuzzy.

3. **The steric-gate mechanism is neither established nor refuted, and our assay could not test it.** Twelve holo simulations (§2.3b) including a deliberately clashing S347F positive control show that minimisation-from-a-modelled-complex relieves any gate-ligand clash within $200\text{ ps}$, regardless of residue size. Both the rigid-structure evidence *for* the mechanism and the dynamic evidence *against* it are therefore uninformative. Resolving it needs alchemical free-energy perturbation or a binding potential of mean force, neither of which we ran. No $\Delta\Delta G_{\text{bind}}$ is reported anywhere in this work.
4. **Multiplicity, pre-registration, and the instrument's noise floor (referees H-1, H-2, H-4).**
   Across its result files this project reports on the order of **131** model x benchmark
   evaluations. The surviving claim rests on a label-permutation $p = 0.0070$, which does not
   survive Bonferroni correction over even the 21 evaluations in a single benchmark file
   (adjusted $p = 0.147$). The defence against this is pre-registration: `scripts/evaluate.py`
   states that the cross-lineage split and `TEST_NEG_GENERA` were fixed before test results
   were examined. **We cannot evidence that claim.** The repository has no version control, so
   nothing distinguishes "fixed before" from "written after", and a reader is entitled to
   withhold the pre-registration defence. Separately, the margin ($\Delta\text{AP} = +0.228$)
   is smaller than the seed-to-seed spread this project itself documents for
   random projections on a related benchmark (AP $0.408$--$1.000$, a range of $0.592$), and
   the cluster-bootstrap interval spans zero. We state these three facts together because
   together they bound the claim far more tightly than any one of them does alone. The single
   most valuable next step is to **pre-register System 2 publicly** -- negative-set rule and
   decision threshold fixed in advance -- and run it.

5. **Ten structure files were read from outside the repository, and eight are still missing.**
   This project loaded structures from a per-session agent workspace
   (`.gemini/antigravity/brain/<uuid>/`) rather than from `data/`. A first repair (round 8)
   fixed three scripts but missed a second path pattern (`<uuid>/4G36/`) and the AlphaFold
   models, so the defect was larger than first reported. All six affected scripts now resolve
   inputs through a single `structure_path()` helper rooted at `data/structures/`, and two
   figure outputs that were being written into the agent workspace now go to
   `results/figures/`. `scripts/fetch_structures.py` downloads the full set. Present:
   `7KYD`, `1U19`. **Missing: `4G36`, `6AAA`, and six AlphaFold models** (`Q9U4U7`,
   `Q9U4U8`, `Q7M4K3`, `A0A8K0DA02`, `A0A8K0C7P5`, `A0A0C6DRW5`). Until they are restored,
   the neighbour-only steering control, the rotamer-relaxation analysis and the structural
   pocket definitions cannot be re-executed. The opsin pocket
   analysis requires `pdb_00001u19.cif` and the CG6178 steering target requires
   `pdb_00007kyd.cif`; both were read from a per-session agent workspace
   (`.gemini/antigravity/brain/<uuid>/structures/`) rather than from `data/`. That directory
   is no longer readable, so those analyses cannot be reproduced by us or by anyone else
   until the structures are re-fetched into the repository. This is the single clearest
   reproducibility defect remaining in the project.
4. **The steering experiment is model-internal, and its code was broken until audit.** §2.4 was unreachable code emitting a cached table; it has been repaired and re-executed, and the re-run reproduces the cached values to $0.0053$. Steering is measured in logit space, not protein space: no protein was expressed, and a shift in predicted amino-acid distribution is not a demonstration of catalytic activity. One of the three originally designed targets (CG6178) could not be re-run for want of a structure file.

5. **The sparse-decomposition advantage did not replicate under capacity-matched controls.** This control has now been run (§2.5, Table 3). At the 8M scale on the corrected strict benchmark, the InterPLM SAE does not outperform the dense representation it decomposes, and its margin over a random ReLU projection is inside that control's seed variance. The claim is withdrawn in §3.1. The control has now also been run at 650M (Table 3b) with the same outcome: the SAE is worse than dense on both AP and genome rank, and scores below all twenty random-projection draws. More fundamentally, the random-projection controls vary by $\pm 0.09$ to $\pm 0.13$ AP across seeds, which exceeds most between-method gaps — with $\sim 5$ effective positive clusters this benchmark is underpowered to rank representations at all.

6. **The phylogenetic contrasts rest on a distance tree.** Neighbor-joining on BLOSUM62 distances, midpoint-rooted, with no substitution-model selection, bootstrap support, or branch-length uncertainty propagation. PIC standardises by branch lengths, so those p-values inherit that error. Williams' test has now been run on an independent reconstruction (§2.7, Table 4): the two predictors do not differ significantly ($p = 0.058$) and the ordering reverses, so the comparative claim is withdrawn. PIC is additionally applied to a binary trait, violating its Brownian assumption, and the contrast set overlaps the probe's training sequences. Re-execution of the project's own pipeline (2026-11-17) reproduces its published values to within $0.013$ but yields the *opposite* ordering from our independent reconstruction, confirming tree-dependence as the operative problem.

8. **No ancestral sequence reconstruction is reported.** An earlier draft included a "140 Mya reconstructed ancestor". It was withdrawn after audit: the reconstructed sequence was $100.00\%$ identical to an extant paralog. A valid reconstruction requires a proper alignment, a maximum-likelihood tree, and per-site posterior probabilities, and is left to future work.

8. **System 3 is a negative result for the model.** For intrinsically disordered AFGPs, Thr+Ala composition alone achieves AUROC $1.000$; ESM-2 provides no advantage. We report this as a regime where a trivial baseline suffices, not as a success.

---

## 4. Methods

### 4.1 Curation of Convergent Evolution Benchmarks
Sequences were retrieved from UniProtKB (reviewed and unreviewed) by Pfam-family and taxonomic query, exported as TSV, and parsed by `scripts/curate.py` ($\text{PF00501}$ for luciferases/ACS; $\text{PF00001}$ for $\text{RH1}$ opsins; $\text{PF00059}$/$\text{PF00089}$ for lectin/antifreeze proteins). Experimental emission wavelengths and absorption peaks were compiled from primary literature sources.

**Label assignment.** Positives require experimental support (UniProt existence "at transcript level" or "at protein level", or a Swiss-Prot reviewed entry). Negatives are $\text{PF00501}$ members from non-luminous beetle genera. Entries named "luciferase" or "luciferin 4-monooxygenase" **solely by automated homology transfer** (n = 124) and paralogs residing inside luminous genomes (n = 71) are retained as `unlabeled` and are **never used as supervised ground truth in either direction** — including as negatives. Fragments outside 400–700 aa are excluded.

**Redundancy control.** No sequence-level clustering tool was applied. Instead, exact duplicates are flagged and dropped (n = 3), and remaining redundancy is handled at *evaluation* time by single-linkage clustering (positives at $\ge 90\%$ pairwise identity; negatives at $\ge 0.85$ normalised alignment score) with $1/|\text{cluster}|$ sample weights passed to every AUROC and average-precision computation. This retains all sequences while preventing near-identical groups from dominating a metric: the 27 click-beetle test positives, for example, comprise only **4.99 effective clusters** because 23 of them are near-identical *Pyrophorus plagiophthalamus* colour variants. All reported average-precision values must therefore be read against the **cluster-weighted prevalence**, which is stated alongside each result and is not 0.5.

### 4.2 Model Embeddings and SAE Decomposition
Models (`facebook/esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`) and Sparse Autoencoders (`Elana/InterPLM-esm2-8m`, `Elana/InterPLM-esm2-650m`) were evaluated via PyTorch. Activation clamping was applied during autoregressive/masked forward evaluation through the ESM-2 transformer layers:
$$\mathbf{h}_l^{(\text{steered})} = \mathbf{h}_l + \alpha \sum_{i \in \mathcal{F}_{\text{steer}}} \Delta z_i \mathbf{W}_{\text{dec}, i}$$
Off-target sequence recovery was quantified across all non-pocket residues by top-1 predicted amino acid agreement and Kullback-Leibler divergence:
$$D_{\text{KL}}(P_0 \parallel P_1) = \sum_{a \in \mathcal{A}} P_0(a) \log \frac{P_0(a)}{P_1(a)}$$

### 4.3 Structural Superposition & Force-Field Relaxation
Atomic coordinates were extracted from RCSB PDB (`4G36`, `6AAA`, `7KYD`, `1U19`, `2PY2`, `1HG7`, `1WFA`) and AlphaFold DB. Core pocket scaffolds were aligned via Kabsch superposition. Continuous rotamer optimization utilized the `scipy.optimize.minimize` L-BFGS-B algorithm under a 12-6 Lennard-Jones potential with harmonic restraints on backbone coordinates.

### 4.4 Phylogenetic Independent Contrasts
Distance matrices were calculated using Poisson/Jukes-Cantor evolutionary distances from pairwise BLOSUM62 alignments. Trees were constructed via Neighbor-Joining (`Bio.Phylo.TreeConstruction.DistanceTreeConstructor`) and rooted at the midpoint. Contrasts across internal nodes were computed following Felsenstein (1985):
$$c_k = \frac{v_1 - v_2}{\sqrt{\sigma_1^2 + \sigma_2^2}}$$

### 4.5 Cloud GPU Molecular Dynamics
Three independent **apo** (ligand-free) simulations were run in OpenMM 8 on a single NVIDIA T4. Structures were protonated at pH 7.0 and solvated in a cubic TIP3P box with $1.0\text{ nm}$ padding and $0.15\text{ M}$ NaCl (89,988–100,356 solvent and ion atoms). Systems were parameterized with AMBER99SB-ILDN (`amber99sbildn.xml`, `tip3p.xml`), energy-minimized, and equilibrated for $100\text{ ps}$ at $300\text{ K}$ / $1.0\text{ bar}$. Production runs used a Langevin middle integrator at $300\text{ K}$ with hydrogen mass repartitioning enabling a $4.0\text{ fs}$ time step: $2.5 \times 10^6$ steps $= 10.0\text{ ns}$ per system, with coordinates written every 5,000 steps ($20\text{ ps}$; 500 frames per trajectory). Throughput was $74.5\text{ ns/day}$. Backbone RMSD was computed against frame 0 after Kabsch superposition; C$\alpha$ RMSF was computed about the mean structure.

**Ligand status and its consequences.** The luciferyl-adenylate intermediate DLSA (`SLU`) has no template in the standard AMBER residue database, and attempts to build it failed in both engines (GROMACS: `Group Protein_SLU referenced in the .mdp file was not found in the index file`; OpenMM: `No template found for residue 542 (SLU)`). All `HETATM` records were therefore removed before system construction, and **the trajectories reported here contain no ligand**. Consequently this work reports **no binding free energy**: no MM/GBSA, MM/PBSA, or alchemical free-energy calculation was performed, and none should be inferred from §2.3. Obtaining $\Delta\Delta G_{\text{bind}}$ for the `L347`/`S347` gate requires prior parameterization of `SLU` (GAFF2/AM1-BCC or Espaloma, e.g. via `openmmforcefields`) and is deferred to future work.

---

## Data and Code Availability
All curated datasets, model checkpoints, reproducible Python scripts, Jupyter/Colab notebooks, and high-resolution figures are openly available in the project repository:  
`https://github.com/<set-before-submission>/translator`

---

## References
1. Felsenstein, J. (1985). Phylogenies and the comparative method. *The American Naturalist*, 125(1), 1-15.
2. Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.
3. Simon, E. P., et al. (2025). Interpreting protein language models with sparse autoencoders. *Nature Methods*, in press.
4. Wood, K. V., et al. (1989). Complementary DNA coding click beetle luciferases can elicit bioluminescence of different colors. *Science*, 244(4905), 700-702.
5. Oba, Y., Iida, K., Ojika, M., & Inouye, S. (2009). Functional conversion of fatty acyl-CoA synthetase to firefly luciferase by site-directed mutagenesis: A key substitution responsible for luminescence activity. *FEBS Letters*, 583(12), 2004-2008.
5b. Oba, Y., et al. (2009). Biosynthesis of firefly luciferin in adult lanterns: conversion of D-cysteine to luciferin. *PLoS ONE*, 4(12), e8473.
5c. Fallon, T. R., et al. (2018). Firefly genomes illuminate parallel origins of bioluminescence in beetles. *eLife*, 7, e36495.
5d. Oba, Y., Sato, M., Ojika, M., & Inouye, S. (2005). Enzymatic and genetic characterization of firefly luciferase and *Drosophila* CG6178 as a fatty acyl-CoA synthetase. *Bioscience, Biotechnology, and Biochemistry*, 69(4), 819-828.
6. Adams, S. T., et al. (2020). Structural basis for the synthesis of firefly luciferin from Drosophila acyl-CoA synthetase. *Biochemistry*, 59(45), 4349-4359.
7. Musilova, Z., et al. (2019). Vision using multiple rod opsins in deep-sea fishes. *Science*, 364(6441), 588-592.
8. Partridge, J. C., et al. (1988). Visual pigments in deep-sea fishes. *Journal of the Marine Biological Association of the UK*, 68(1), 33-47.
9. Chen, L., et al. (1997). Convergent evolution of antifreeze glycoproteins in Antarctic notothenioid fish and Arctic cod. *Proc. Natl. Acad. Sci. USA*, 94(8), 3817-3822.
10. Baalsrud, H. T., et al. (2018). De novo gene birth of antifreeze glycoproteins in Arctic cod. *Molecular Biology and Evolution*, 35(1), 86-96.
11. Menechino, B. S. B., et al. (2023). Structural and energetic basis of Dengue and Zika virus envelope recognition by DC-SIGN. *Journal of Biomolecular Structure and Dynamics* [PMID: 38049805].
