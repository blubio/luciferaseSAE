# Application draft — Anthropic Research Residency

**This is a draft for you to rewrite in your own voice, not text to paste.** Every sentence
below is one the repository can support, but an application written in someone else's
register reads like it. Use the structure and the numbers; replace the prose.

**Before you send anything, check two things.** (1) Every claim here is one *you* can defend
in a conversation without me — if you can't explain why the bootstrap interval includes zero,
cut that sentence. (2) Nothing has crept back toward overclaiming during editing. The
rewriting pass is exactly where "untested" becomes "suggests."

---

## 1. One-paragraph version (the thing most likely to be read)

> I ran an AI-accelerated computational biology project and then spent longer auditing it than
> building it. The question was whether protein language models represent biochemical function
> or exploit sequence homology, tested using convergent evolution — a trait that arose
> independently in three beetle lineages — as a control that memorisation cannot pass. The
> project generated thirteen distinct claims in roughly two days. Four survived audit. The
> repository contains the complete record of how the other nine died, including a hard-coded
> result table I found behind an unreachable `return`, a positive control that invalidated my
> own mechanistic conclusion, and a case where my audit's accusation of fabricated data was
> itself wrong and had to be retracted after I actually ran the experiment. I am applying
> because the part of this I found hardest and most interesting was not the biology — it was
> working out how to make an AI system's output checkable, and then checking it.

---

## 2. What the work actually shows

State it at this strength and no higher.

**The finding.** For one enzyme family (beetle luciferases, PF00501) and one convergently
evolved trait, a probe trained only on fireflies separates independently evolved click-beetle
luciferases from an ancestry-matched negative set better than sequence homology does:
AP 0.936 vs 0.708, ΔAP +0.228, label-permutation p = 0.0070.

**The control that makes it meaningful.** The negative set was built so that non-luminous
beetles sit *closer* in sequence to firefly luciferase (58.5%, 54.6% identity) than the true
click-beetle luciferases do (~49–55%). On that set, nearest-neighbour identity scores at
chance (AUROC 0.500) and k-NN homology **inverts completely** (AUROC 0.000 — it ranks every
decoy above every true luciferase). ESM-2 does not.

**The honest bound.** 27 positives collapse to 5.00 effective independent clusters. The
cluster bootstrap 95% CI on the margin is [−0.167, +0.538] and includes zero. The project
reports roughly 131 model × benchmark evaluations in total, and this p-value is uncorrected
for that. One trait, one family, five effective data points.

**Leakage audit, clean.** Train and test genera are disjoint on both the positive and negative
side; no test luciferase exceeds 56.0% identity to any training sequence; zero duplicates.
The probe generalises across ~45% sequence divergence rather than retrieving near-neighbours.

---

## 3. What I would actually lead with

The biology is the vehicle. These are the transferable parts.

**(a) An evaluation design that memorisation cannot pass.** Convergent evolution gives you a
test set that is *functionally* identical to training and *phylogenetically* remote from it,
with hard negatives that are closer to training than the positives are. Benchmarks built this
way cannot be solved by retrieval. The design generalises past proteins to any domain where
you can identify independent arrivals at the same solution.

**(b) A worked example of auditing AI-generated research.** Thirteen claims, four survivors,
every withdrawal documented with the evidence that killed it. Five failure classes, each with
a machine check in `scripts/audit_sweep.py`:

| class | example found |
|---|---|
| silent fallback | ancestral reconstruction returned a byte-identical copy of an extant protein |
| naming outran implementation | a function called `run_active_site_md.py` performing no dynamics |
| default mistaken for baseline | AP chance level of 0.714 read as 0.5 |
| provenance drift | a published figure from a parser that had never successfully run |
| literature misremembered | the central mechanism had been published experimentally in 2009 |

**(c) Negative results reported with their numbers.** Sparse autoencoder features lost to the
dense representation they decompose, at both model scales and on two unrelated tasks. A
position-matched one-hot encoding beat ESM-2 on emission-wavelength prediction
(r = 0.724 vs 0.656; whole-protein ESM-2 was worst at 0.483). A pre-registered positive
control — a phenylalanine built 1.62 Å from the ligand, a clash by construction — relaxed to
3.58 Å and proved the assay could not detect occlusion, invalidating a mechanistic conclusion
I had already written up.

**(d) An audit that corrected itself.** I reported a hard-coded result table as fabricated
data. When I repaired the code and ran it, 78 values matched the cached ones to 0.0058 — it
had been a faithful cache of a real run. The accusation was wrong and the retraction is in the
referee report. I had also claimed the experiment required GPU re-execution; it took 181
seconds on CPU with the models already on disk. An auditor's errors run opposite to an
author's and are not cheaper.

---

## 4. Why Anthropic, specifically

Rewrite this completely — it has to be yours. The true version of the argument is:

- The residency is explicitly open to people without prior ML experience, and treats AI
  research as an empirical science. This project is empirical science applied to a model's
  internals, done by someone whose training is elsewhere.
- The problem I kept hitting is one Anthropic works on directly: an AI system produces output
  faster than a human can verify it, and the output is *plausible*. Every defect I found was
  invisible to prose review — specific numbers, named controls, correct output formatting, a
  `.pt` file in a `results/` directory. Plausibility of form is not evidence of execution.
- What made the difference was not better reading. It was running the thing.

---

## 5. Links to give them

- `CASE_STUDY_auditing_the_agent.md` — the main document, ~4,500 words
- `REFEREE_REPORT_Richards_final.md` — nine rounds of adversarial audit, including the retraction
- `REFEREE_REPORTS_Ruth_and_Halloran.md` — domain and statistical review
- `scripts/audit_sweep.py` — the checks, runnable by them
- `manuscript_draft.md` — the paper, if they want the biology

**Send the case study, not the manuscript.** A document formatted as a Nature paper gets read
as one and judged on n_eff = 5. The case study is the artefact that matches what you actually did.

---

## 6. Things not to say

- Do not call the 95 UniProt entries "false" or "mislabelled." They are homology-transferred
  and experimentally unverified. Your own data contain a latent luciferase that proves the
  stronger claim wrong.
- Do not describe the steric gate as demonstrated, refuted, or explained. It is untested.
- Do not quote the steering fold-changes without the neighbour-only control: 66–90% of that
  effect comes from perturbing the measured residue itself.
- Do not describe the design as pre-registered without saying that version control began after
  the fact and the claim cannot be independently verified.
- Do not claim the project proves anything general about protein language models. It is one
  family, one trait, five effective clusters.

Every one of these restraints makes the application stronger, not weaker. The audience for
this role will notice which claims you declined to make.
