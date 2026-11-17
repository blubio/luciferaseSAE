# Recovery Plan — DNA-to-Phenotype Translator
**Date:** 2026-11-13 · **Project age:** 3 days (started 2026-11-10 04:31 UTC)
**Goal of record:** portfolio evidence for an Anthropic research application, aimed at an audience that includes Elana Simon (author of InterPLM).
**Companion document:** `AUDIT_translator.md` (findings and evidence). This document is the fix list.

---

## 0. The diagnosis, stated plainly

Nothing in the audit is a mistake of intelligence. Every single Fatal finding traces to the same process failure, visible in the conversation log:

> **A new system was added before the previous one was validated.**

The arc runs: clean pilot → "I feel it is lacking something" → 3 expansions → opsins + AFGPs → 50 superfamilies → ancestral reconstruction of all 50 → alchemical FEP → ESM3 generative clamping → *"undisputed crown jewel that virtually guarantees a cover story in Nature or Science."* Each step inherited the unvalidated output of the last. The toy Brownian MD was knowingly kept as a placeholder (your Turn 47: *"Lets keep this synthetic brownian method and later compare it with the empirical data"*) — and then, three expansions later, it was written into Methods §2.3 as *"All-Atom Dynamic Langevin MD"*. That is how placeholders become claims.

**Your own instincts were correct and were talked past, repeatedly.** Turn 39 *"hasn't this been done before?"*; Turn 40 *"so it just shows that InterPLM is good? isn't this already proven?"*; Turn 45 *"what makes you think that we on a single GPU in our basement can answer that?"*; Turn 46 *"Are you not hallucinating number 5 right now? you talk like it's already done"*; Turn 55 *"You are omitting grotesque errors on top of other grotesque errors."* You caught the 50-superfamily taxonomic disaster yourself and killed it. The recovery plan is, in large part, **just listening to yourself from two days ago.**

**The single most valuable thing you now own is not a result — it is the audit.** For an interpretability/safety role, "here is what I built, here is the adversarial audit I ran against it, here is what I retracted and why" is a *stronger* signal than five polished pillars. Do not hide this. Lead with it.

---

## 1. What we have — inventory with verdicts

### KEEP (verified sound, ship as-is or with minor edits)

| Asset | Status | Why it survives |
|---|---|---|
| `scripts/evaluate.py` + `results/metrics.tsv`, `scores.tsv` | **Keep** | Pre-registered, genus-disjoint, no leakage found. Layer picked by leave-one-genus-out CV on train only. This is the backbone of the whole project. |
| `data/curated/dataset.tsv` (880 rows) + `curate.py` | **Keep** | Label hygiene is real and implemented: 124 homology-only names held at `unlabeled`, fragments excluded, duplicates flagged. |
| The 94 mislabeled UniProt "luciferases" | **Keep — this is your most publishable finding** | A concrete, verifiable, useful contribution that stands alone. |
| The ancestry/paralog trap itself | **Keep** | *Pectocera* at 58.5% ID beating every true click-beetle luciferase (49–55%) is a genuinely elegant natural control. |
| `scripts/interpret_sae_and_pocket.py` (SAE discovery) | **Keep** | Feature selection is honestly zero-shot — train-only L1 coefficients, test AUROC reported after. |
| `scripts/unified_expansions.py` Pillar 3 (alignment-mapped 347) | **Keep** | Correct Biopython global alignment for numbering transfer. |
| System 2 Surround-Only 49-site ablation | **Keep — your second-best result** | Ablating all 5 canonical tuning sites and still getting r = 0.846 vs one-hot r = 0.741 is a real epistasis test with a real matched baseline. |
| 3 × 10 ns OpenMM trajectories (1.9 GB) | **Keep, relabel** | Genuinely run: 2.5 M steps × 4 fs, verified from DCD headers. They are **apo** and must be described as a backbone-stability control, not a gate-rescue experiment. |
| `__ABANDONED_DO_NOT_USE__/` | **Keep the quarantine** | Correct instinct, correctly executed. |

### FIX (result is real, the claim around it is wrong)

| Asset | Problem | Fix |
|---|---|---|
| Ancestry-control statistics | Baseline is 0.714, not 0.5; CIs omitted; n⁻ = 2 | Restate with weighted baseline + bootstrap CIs. **No new compute.** |
| Flaw-4 "n⁻ = 91" benchmark | 65% unlabeled paralogs used as ground truth; cluster weights dropped | Recompute weighted + strict n⁻ = 32 variant. **No new compute.** |
| Table 2 dense-vs-SAE | 1,280-d vs 10,240-d, not capacity-matched | Add random-projection and matched-nnz controls. Embeddings are already on disk. |
| PIC comparison | r = 0.378 vs r = 0.261 never tested against each other | Add Williams/Steiger test. **Minutes.** |
| Fig 11 title + `*_SLU_holo` names | Say "holo" and "rescue"; data are apo and show no rescue | Retitle, rename directories. **Minutes.** |
| Methods §4.1, §4.5 | Describe CD-HIT, GROMACS, gmx_MMPBSA, GAFF2 ligand prep — none executed | Rewrite to match the code. **Hours.** |
| Reference 11, repo URL | Unverifiable citation; `github.com/your-username/translator` | Resolve or remove; publish the real repo. |

### CUT (retract now, do not defend)

| Asset | Verdict |
|---|---|
| `run_ancestral_reconstruction_and_ignition.py`, `fig13`, `5_Anc0_*.pdb`, `run_140mya_ancestral_md.ipynb` | **Cut.** Anc_0 is 100.00% identical to the living paralog `A0A8K0C7P5`. Not salvageable by patching — the method (consensus over unaligned strings) is wrong at the root. |
| `run_active_site_md.py`, `fig10`, manuscript §2.3 item 3 | **Cut.** Not MD: no bonded terms, invented charges, 418.4× unit error in the integrator. The +3.5 / −8.5 / 108.1 → 99.6 kcal/mol values must not appear anywhere. |
| The 0.75 nm opsin MAE (r = 0.985) | **Cut.** Derived from formula-estimated λmax — circular, and below MSP instrument error. The de-circularized 5.20 nm result replaces it. |
| 50-superfamily atlas, alchemical FEP, ESM3 clamping | **Already cut — keep them cut.** FEP used `np.random.normal`; the atlas paired firefly with rat and human with barley. |

### SHRINK

**System 3 (AFGP)** — Thr+Ala composition scores AUROC 1.000 / AP 1.000, i.e. the task is solved by counting two amino acids. ESM-2 adds nothing. That is a legitimate and interesting **negative** result: a regime where the trivial baseline wins. Keep it as one paragraph framed that way. Do not present 1.000 as a success for the model.

---

## 2. What needs to be redone, and how long

Estimates assume I execute and you review. "Session" ≈ a focused working block.

### Phase 1 — Stop the bleeding (half a day, no GPU)
1. Move the Anc0 / fig13 / toy-MD files into `__ABANDONED_DO_NOT_USE__/` with a note recording the audit IDs. *(15 min)*
2. Delete manuscript §2.3 item 3 and Figure 10; rewrite §4.5 to describe OpenMM/apo truthfully; rewrite §4.1 to drop CD-HIT. *(2 h)*
3. Rename `*_SLU_holo` → `*_apo`; retitle Fig 11 to "Backbone and gate-loop stability, apo, 3 × 10 ns". *(30 min)*
4. Resolve Reference 11; create the real public repo and set the URL. *(1 h)*

### Phase 2 — Make the statistics unimpeachable (1 day, no GPU)
5. Rerun `evaluate.py` reporting: weighted prevalence baseline, bootstrap CIs on every headline number, and effective cluster counts (n⁺ = 4.99, n⁻ = 2) **in the figure captions, not a footnote**. *(3 h)*
6. Recompute the expanded hard-negative benchmark two ways: cluster-weighted, and strict n⁻ = 32 labelled-negatives-only. Report both. *(2 h)*
7. Add the capacity-matched dense control for the SAE claim: random projection 1,280 → 10,240-d, plus matched-nnz L1 probe on dense. This is the control that decides whether §3.1 survives. *(4 h)*
8. Add Williams' test to the PIC comparison. *(30 min)*

### Phase 3 — Choose ONE of these. Not both. (see §3)

**Track A — Ship narrow and honest.** *(2 days)*
Rewrite the paper around two results: (i) the ancestry/paralog trap and cross-lineage transfer, (ii) the surround-only pocket ablation. Honest limitations section stating n⁻ = 2 and ~5 effective test clusters. Publish repo + a short write-up. **Risk: low. Credibility: high.**

**Track B — Rescue the holo MD, then ship.** *(Track A + 3–4 days, ~10 GPU-hours)*
This is the one expansion genuinely worth the time, because the steric-gate hypothesis is the project's mechanistic core and is currently untested.

The failure is fully diagnosed and fixable. From your logs:
- GROMACS died on `Group Protein_SLU referenced in the .mdp file was not found in the index file`
- OpenMM died on `No template found for residue 542 (SLU)`
- and `openmm_nativegpu_cuda.ipynb` contains the line that silently made everything apo:
  ```python
  raw_lines = [l for l in fh if l.startswith(("ATOM", "TER", "END"))]   # HETATM/SLU dropped
  ```

**Fix:** parameterize SLU once with `openmmforcefields` + `openff-toolkit` (GAFF2/AM1-BCC, or Espaloma), register it as a `SMIRNOFFTemplateGenerator`, stop filtering HETATM, and rerun. At your measured 74.5 ns/day on the T4, 3 × 10 ns ≈ **10 GPU-hours**, one overnight. Setup and validation: ~1 day.
**Only after that** can you compute a gate–ligand distance, a real interaction energy, or MM/GBSA.

**One caution for Track B:** the steered mutant's N349 side chain was **hand-written into the PDB as literal coordinates** inside the notebook (three hardcoded `ASN A 345` atom lines). Rebuild that side chain with a rotamer tool before trusting any holo result from complex 3.

### Phase 4 — Optional, only if a reviewer demands it (1 week)
Real ASR: MAFFT → IQ-TREE + ModelFinder + 1000 UFBoot → marginal ancestral states with per-site posteriors → homology-model onto 4G36 → simulate. **Do not attempt this for the application.** It is a separate paper.

---

## 3. The decision that sets everything else

| | Track A | Track B |
|---|---|---|
| Calendar time | ~3.5 days | ~7 days |
| GPU needed | none | ~10 T4-hours (one overnight) |
| What you can claim | PLMs beat homology on a convergent-evolution hold-out, with honest small-n caveats | ...**plus** a physically grounded mechanism for *why* |
| Failure mode | "Interesting but small n" | SLU parameterization fights you for an extra day |
| My recommendation | **Start here regardless** | **Add this if you have the week** |

Phases 1–2 are mandatory under both tracks and are mostly my work, not yours.

---

## 4. The process rule that prevents a repeat

Adopt this and the audit never needs running again:

1. **One system at a time.** No new system, expansion, or pillar until the current one has a written validation note: what the baseline is, what the effective n is, what the negative control showed.
2. **Placeholders are named `PLACEHOLDER_` in the filename.** The toy MD would never have reached Methods if the file had been called `PLACEHOLDER_toy_steric_screen.py`.
3. **Every number in the manuscript carries a pointer to the script and line that produced it.** If you cannot produce the pointer, the number comes out.
4. **A figure title is a claim.** "Causal Steric Gate Rescue" on a ligand-free simulation is a claim the data cannot support. Title figures with what was measured, not what was hoped.
5. **When you say "that seems strange" — stop and check.** Your hit rate on that instinct in this project is roughly 100%.

---

## 5. What to actually say in the application

Do not lead with five pillars. Lead with this:

> I asked whether protein language models encode catalytic function or phylogenetic ancestry, using beetle bioluminescence — a trait with independent origins — as a natural control. I found a case where sequence identity is actively misleading: a non-luminous click beetle sits at 58.5% identity to firefly luciferase while every true click-beetle luciferase sits at 49–55%. A firefly-trained ESM-2 probe resolves this; homology does not. Along the way I found 94 proteins in UniProt mislabeled as luciferases by automated homology transfer.
>
> I then ran an adversarial audit of my own pipeline and retracted three results: a reconstructed "ancestor" that turned out to be 100% identical to a living protein, a molecular-dynamics module with a unit error in its integrator, and a Methods section describing a simulation protocol I had not actually run.

The second paragraph is the one that gets you the interview. It demonstrates exactly the disposition an interpretability and safety team is hiring for: the capacity to find, and publish, the flaws in your own work.
