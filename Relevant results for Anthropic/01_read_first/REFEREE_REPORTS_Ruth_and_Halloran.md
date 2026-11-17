# Independent Referee Reports — Reviewers 3 and 4

**Manuscript:** *Convergent Evolution as a Control for Phylogenetic Confounding in Protein
Language Models* (v13, 9,009 words)
**Date:** 2026-11-17
**Note:** Both reviewers were asked to find what the preceding reviewers did not. Reviewer 1
(second model) flagged duplicated coordinates; Reviewer 2 (Richards, rounds 4–9) audited code,
provenance and reproducibility. Neither examined the **biology** or the **inferential
structure**. These reports do.

---

# Reviewer 3 — Ruth
### Molecular evolution and protein biochemistry

I have reviewed in this exact area for two decades. I will start with the finding that
matters most, because it is not a criticism.

## R-1 (MAJOR STRENGTH, UNCLAIMED) — your own data support your premise, and you never say so

The entire design rests on bioluminescence having arisen **independently** in Lampyridae,
Elateridae and Phengodidae. The manuscript asserts this by citing Fallon et al. (2018). It
is a contested question in this field — single-origin-with-losses has serious proponents —
and a referee is entitled to ask why your control is a control at all.

**You have the evidence internally and do not present it.** From your own similarity matrix
(median nearest-neighbour normalised alignment score):

| from → to | firefly luc | click-beetle luc | non-luminous Elateroidea |
|---|---|---|---|
| **click-beetle luc** | 0.493 | — | **0.602** |
| **non-luminous Elateroidea** | 0.556 | **0.692** | — |

Click-beetle luciferases are **more similar to their own non-luminous relatives (0.602) than
to firefly luciferases (0.493)**, and the non-luminous Elateroidea are closest of all to the
click-beetle luciferases (0.692). Under a single ancestral luciferase retained in both
lineages, the luciferases should group with each other against the non-luminous background.
They do not. The topology is what independent recruitment from a *local* fatty acyl-CoA
synthetase predicts.

**Add this to §2.1 as a figure.** It converts your premise from a citation into a result,
and it is the cheapest substantial improvement available to this paper.

## R-2 (MAJOR) — "misannotation" is not established, and your own data argue against it

You report 95 UniProt entries carrying luciferase-family names in non-luminous beetles and
characterise them as wrong. The inference is: *the organism does not glow, therefore the
protein is not a luciferase.* That does not follow, and **your manuscript contains the
counterexample.**

`A9ZPM4` (*Agrypnus binodulus* AbLL) is a **latent luciferase**: Oba et al. (2009) showed a
single serine substitution confers luminescence, and it has measurable activity in a
non-luminous beetle. You correctly disclose this for the ancestry control. You do not apply
the same logic to the 95.

An AMP-binding enzyme can possess weak luciferase activity in an organism with no light
organ. Absence of organismal bioluminescence is evidence about the *organism*, not a
functional assay of the *protein*. Without activity measurements you can claim that the
annotations are **unverified homology transfers** — which is defensible, interesting, and
supported by the 92/95 "Inferred from homology" tags — but not that they are **false**.

**Required:** replace "mislabelled" / "false luciferases" / "annotation poisoning" with
"homology-transferred and experimentally unverified" throughout, including the abstract and
the Conclusions, where "94 false luciferases" still appears in the biosecurity paragraph.

## R-3 (MAJOR) — the ancestry control is two proteins

$n^- = 2$. AUROC 0.910 on 27 positives versus two negatives is a statement about two
proteins, one of which (R-2) is a latent luciferase whose negative status is a definitional
choice. Your disclosure is honest but insufficient: a reader scanning the abstract sees a
decisive-looking separation.

**Required:** report $n^-$ in the abstract wherever the ancestry-control AUROC appears, and
move the headline to the strict benchmark ($n^- = 31$), which is a real evaluation.

## R-4 (MAJOR) — the steering experiment measures the residue it perturbs

`pocket_steer_sites = [245, 247, 314, 315, 340, 343, 345, 346, 347, 348, 349, 422, 423, 529,
535, 538]`. **Position 347 is itself steered**, and the gate log-odds are then read at 347.
The "off-target scaffold recovery" window excludes exactly these 16 sites.

So the headline — "clamping 50 SAE features shifts the gate prediction $16\times$–$991\times$"
— is in substantial part the statement that *adding a vector to a residue's hidden state
changes that residue's own prediction.* The direction of the effect is close to guaranteed.

What rescues the experiment is the matched-norm controls: orthogonal-50 and random-50 at the
**same sites** give $1.0\times$–$1.4\times$. That is a real result, but it supports a much
narrower claim than the one made:

> A particular direction in SAE space, applied at the pocket residues, moves the model's
> prediction at 347 toward Ser/Ala far more than random directions of equal norm do.

It does **not** establish that the selected features "encode the catalytic gate," and the
manuscript's language should be brought back to the first statement.

Two controls are missing and both are cheap:
- **Spatial control.** Steer 16 residues *distal* to the pocket with the same procedure and
  measure at 347. If the effect persists, the pocket specificity claim fails. I find no such
  control anywhere in the code.
- **Dose–response.** $\alpha = 2.0$ is fixed in the reported analysis. An $\alpha$ sweep
  exists in `unified_expansions.py` (`[0, 5, 10, 15, 20, 25]`) but is not the reported
  experiment. A monotone dose–response at the gate with a flat scaffold response would be
  far more persuasive than a single operating point.

## R-5 (MODERATE) — the opsin result is weaker than "r = 0.845" sounds

$\lambda_{max}$ panel: $n = 41$ analysed, range 468–517 nm, SD 12.55 nm.

| predictor | MAE |
|---|---|
| predict the mean | 10.39 nm |
| predict the median | 9.39 nm |
| **ESM-2 Surround-49** | **5.18–5.33 nm** |

A 50% reduction in MAE over the mean, on $n = 41$, with leave-one-out ridge in a
high-dimensional representation. That is a positive result and I would not reject it, but
`r = 0.845` invites the reader to imagine something much stronger. **Report the mean-baseline
MAE alongside every reported MAE.** Without it the number is uninterpretable.

The SAE comparison here is the most informative thing in that table and is buried:
InterPLM-8M $r = 0.748$ / 6.42 nm and InterPLM-650M $r = 0.644$ / 7.78 nm against
$0.843$–$0.850$ / 5.18–5.33 nm for the dense models. **The sparse decompositions are clearly
worse on a continuous phenotype.** Promote this.

## R-6 (MINOR)

- Ancestral-state reconstruction in §3B reports a root gate loop `YG(L/M)TEAT[L347]AV` from
  a single NJ tree with no bootstrap support and no substitution model. State it as
  illustrative or drop it; after the 140 Mya episode this paper cannot afford another
  unsupported ancestral claim.
- *Ignelater luminosus* genome retrieval uses "$\ge 80\%$ identity to a cloned *Pyrophorus*
  luciferase" as ground truth for "putative luciferase." That is a homology criterion used
  to score a method whose selling point is beating homology. Flag the circularity explicitly.

## Ruth's recommendation

**Major revision.** The core experiment is sound and the premise is better supported than the
manuscript realises (R-1). The problems are overclaiming at three specific points — R-2, R-3,
R-4 — each fixable by narrowing language to what was measured, plus two cheap missing
controls. I would review a revised version.

---

# Reviewer 4 — Halloran
### Statistical methodology and empirical evaluation

My concern is not any individual number. It is the **inferential structure** of the project
as a whole, which no previous reviewer has examined.

## H-1 (MAJOR) — the surviving p-value is not corrected for the search that produced it

The project evaluates, across its result files:

| file | reported evaluations |
|---|---|
| `metrics_corrected.tsv` | 42 |
| `hard_negative_benchmark_corrected.tsv` | 21 |
| `table2_multiplm_REAL.tsv` | 36 (6 models × 6 metrics) |
| `step1_steering_REAL.tsv` | 24 |
| `pic_REAL.tsv` | 8 |
| `holo_md_summary.tsv` | 12 |
| **total** | **≈ 131** |

The surviving claim rests on a label-permutation $p = 0.0070$.

| correction over | adjusted $p$ | survives $\alpha = 0.05$? |
|---|---|---|
| 21 tests (one benchmark file) | 0.147 | **no** |
| 63 tests (the two benchmark files) | 0.441 | **no** |
| 131 tests (project-wide) | 0.917 | **no** |

Bonferroni over all 131 is unfairly harsh — those are not 131 independent tests of one
hypothesis. But the honest position is not "no correction either," and the paper currently
applies none and discusses none.

**The defence available to you is pre-registration**, and that brings us to H-2.

## H-2 (MAJOR) — the pre-registration claim cannot be verified

`scripts/evaluate.py` opens with:

> *"Pre-registered design (fixed before looking at test results)"*

with `TEST_NEG_GENERA` annotated *"Fixed a priori."* If true, the headline comparison needs
no multiplicity correction and H-1 largely dissolves.

**There is no version control in this repository.** No `.git`, no commit history, no
timestamps that distinguish "fixed before" from "written after." The claim is therefore
unfalsifiable as it stands, and a referee cannot grant it.

This is not an accusation. It is a statement about what the artefact can support. Note the
asymmetry: the project's most valuable methodological asset — a genuinely pre-registered
cross-lineage split — is the one claim it cannot evidence, purely for want of `git init`.

**Required:** either (a) state plainly in Methods that pre-registration is asserted but not
independently verifiable, or (b) pre-register the *next* system publicly before running it.
Option (b) is worth more than anything else in the revision.

## H-3 (MAJOR) — the strict negative set is a researcher degree of freedom

The headline uses a "strict" negative set of $n^- = 32$. **That set is defined nowhere in
the code.** It was recovered during audit by reconstruction — the 31 highest-identity
labelled negatives reproduce the published `nn_identity` row exactly — but "the 31 (or 32)
negatives with highest identity to training" is a *choice*, and the threshold is a free
parameter that directly controls the headline effect size.

A reader cannot tell whether $n^- = 32$ was fixed in advance or selected. Given H-2, neither
can the authors.

**Required:** report the headline as a function of the cutoff ($n^- = 20, 30, 40, 50, \ldots$,
through the full labelled negative set). If the margin is stable across cutoffs, the concern
evaporates and the result is strengthened. If it peaks near 31, that is important to know.

## H-4 (MAJOR) — the claimed effect is smaller than the benchmark's own noise floor

The project's own control, which I commend, found that random projections of the same
embedding span **AP 0.408 to 1.000** across 20 seeds on the 650M capacity benchmark — a
range of 0.592 from the seed alone, against a 0.302 spread across all named methods.

The headline margin is $\Delta\text{AP} = +0.228$, with a cluster-bootstrap 95% CI of
$[-0.167, +0.538]$.

The paper reports these in different sections and never confronts them. **An effect of 0.228
measured on an instrument whose seed-to-seed noise spans 0.592 is not a measurement.** The
bootstrap CI already says this; the random-projection result says it independently. Put them
in the same paragraph and state the implication.

## H-5 (MODERATE) — nothing in the paper could have come out negative

I can identify no outcome of the surviving experiment that the authors would have reported as
a refutation. The withdrawn claims were withdrawn because of *audit*, not because a
pre-specified test failed. The one genuine falsification test in the project — the S347F
positive control — is excellent, and is the model the rest should follow.

**Required:** for the surviving claim, state the pre-specified threshold that would have
counted as failure. If none was set, say so.

## H-6 (MODERATE) — related work is thin where it matters most

The premise is that benchmarks conflate function with homology. This is an active literature —
sequence-sampling bias in PLM training sets, phylogenetic signal in PLM embeddings, homology
leakage in protein benchmarks. The manuscript cites none of it and positions the observation
as novel. It is not; the *convergence-based control* may well be, and that is the claim to
defend.

**Required:** a related-work paragraph that separates "PLMs are confounded by homology"
(known) from "convergent traits are a clean control for it" (your contribution).

## H-7 (MINOR)

- Report $n$ and $n_{\text{eff}}$ beside every AUROC/AP in every table, not only in the text.
- The genome-retrieval rank statistic (#1/61) has no confidence interval and no null.
- `composition` is used as a sanity floor; with AP up to 0.43 on the strict set it is not a
  floor, it is a weak baseline, and should be described as one.

## Halloran's recommendation

**Major revision.** The empirical work is more careful than most I see, and the audit trail is
exemplary. But the paper reports a single surviving $p$-value from a project that ran on the
order of 131 evaluations, defends it with a pre-registration claim its own artefact cannot
support, on a negative set whose definition is absent from the code, with an effect smaller
than the instrument's documented noise floor. Each of these is fixable — three of the four by
writing down what was already done, and H-3 by one additional sweep.

My single recommendation, worth more than the rest: **pre-register System 2 publicly, with
the negative-set rule and the decision threshold fixed in advance, and run it.** That converts
this from a paper defending one result into a paper with a method.

---

## Where the four reviewers agree

Independently, and from different directions:

1. **The ancestry trap is the contribution** and it works (all four).
2. **The sparse-autoencoder advantage does not survive** capacity-matched controls (Richards),
   and the SAEs are worse than dense on a continuous phenotype too (Ruth).
3. **$n_{\text{eff}} \approx 5$ is the ceiling**, not the audit (Richards, Halloran).
4. **The language overclaims relative to what was measured** at three specific points
   (Ruth: R-2, R-3, R-4) — each correctable without new experiments.
5. **`git init`** would retire one fatal-class defect (Richards), one unverifiable claim
   (Halloran H-2), and the provenance failures behind both.
