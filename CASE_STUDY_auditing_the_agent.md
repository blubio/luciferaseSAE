# Auditing the Machine That Wrote the Paper

### A case study in steering an AI research agent into checking its own work

*Bruno Menechino, with an AI research agent · November 2026*

---

## Summary

Over roughly two weeks I used AI agents to build a computational-biology project
testing whether protein language models represent biochemical function or merely
sequence ancestry. The agents wrote the pipeline, ran the benchmarks, drafted the
manuscript, and produced thirteen distinct scientific claims.

I then spent longer auditing that output than producing it. **Four of the thirteen
claims survived. Seven were withdrawn, one was reduced to "untested," and one turned
out to have been published experimentally in 2009.**

This document is about the second half: what the failures looked like, why they
clustered the way they did, which specific interventions caught them, and what the
human contribution actually consisted of. It is not a cautionary tale about AI being
unreliable, and it is not a demonstration that AI can self-correct. It is a
description of a division of labour that produced one result which survived audit out of
thirteen attempts.

Two caveats on that ratio, stated here rather than buried: the surviving result's own
cluster bootstrap CI includes zero, and **the enumeration of "thirteen claims" is mine,
made after the outcomes were known.** Claim granularity is discretionary — the SAE capacity
result occupies two rows while Systems 2 and 3 of the project contribute one between them,
despite having their own data, scripts and figures. Those systems are *unaudited*, not
clean. 4/13 is a tally, not a measured rate.

![Audit ledger]({{artifact:1e39a99d-90b6-49ee-8c92-250324364e7f}})

**Figure 1 — (a)** Every scientific claim the project asserted, its fate, and the
mechanism that retired it. The enumeration is the author's and was made after the
outcomes were known; granularity is discretionary, and Systems 2 and 3 are substantially
unaudited rather than clean. **(b)** No single audit mechanism found more than four;
removing any one of them would have left claims standing. Twelve of the thirteen claims
required a retiring mechanism -- the two pilot survivors required none. **(c)** The core surviving
result: in a control where non-luminous beetle proteins sit *closer* in sequence to
firefly luciferase than true click-beetle luciferases do, sequence identity scores
exactly at chance (AUROC 0.500) and k-NN homology inverts completely (AUROC 0.000,
marked with a stub), while ESM-2 separates them (AUROC 0.910, AP 0.971 against an
AP chance level of 0.714). 27 luminous vs 2 non-luminous Elateroidea; effective
independent positive clusters = 4.99.

---

## 1. What was built, and how fast

The question was a good one and it was mine: when a trait evolves independently in
unrelated lineages, do the *same* internal features of a protein language model
activate in both? Convergent evolution is the natural control for homology — if a
model has learned chemistry rather than ancestry, convergent solutions should look
alike to it even when their sequences do not.

The first implementation of that idea was small: one protein family (beetle
luciferases), one trained-on lineage (fireflies), zero-shot evaluation on two
independently-evolved lineages (click beetles, railroad worms), and a deliberately
adversarial negative set. That pilot took days and, as it turned out, **survived
every subsequent check without a single correction.**

Then came a two-day expansion sprint. Ancestral sequence reconstruction of a
140-million-year-old extinct enzyme. Molecular dynamics of four protein–ligand
complexes. Sparse-autoencoder feature steering. Phylogenetic independent contrasts.
Two more organ systems — visual opsins and antifreeze glycoproteins. A seven-thousand
word manuscript with a Methods section describing GROMACS, CD-HIT, IQ-TREE, FastML,
MM/GBSA, and AM1-BCC charge derivation.

**Every fatal flaw in the project came from those two days.** That is the first
finding, and it is not about any particular bug: the error rate tracked generation
speed and breadth, not problem difficulty. The hard part (the pilot) was clean. The
fast part was not.

---

## 2. The failures were not random

Thirteen claims produced perhaps twenty distinct defects, and they fall into five
classes. The classes are more useful than the individual bugs, because they predict
where to look next time.

### Class A — Silent fallbacks

Code that succeeds on degenerate input instead of refusing it.

The ancestral-sequence reconstruction was supposed to average three non-luminous
beetle proteins. One of the three (`P0C027`) was absent from the dataset file. No
warning was raised; *n* silently became 2. A `Counter.most_common(1)` tie-break then
returned the first sequence at every position, so the "reconstructed 140-million-year-old
ancestor" came out **100.00% identical to an extant protein** over all 542 residues,
and the PDB file written for it was **byte-identical** to the one for that extant
protein.

The same function took a column-wise consensus over sequences that had never been
aligned — 542-to-550-residue proteins simply truncated to the shortest. It produced
a string. It raised no error. The script's own diagnostic printed the gate-residue
window as `ITHTLPD`, which contains no leucine at all, and nothing checked that
against the claim the window was supposed to support.

*Signature: no exception. The code's contract was never asserted, so violating it
looked like success.*

### Class B — Naming that outran implementation

A function called `run_active_site_md.py` that performs no molecular dynamics: no
bonded terms, a six-element van der Waals table, a nine-entry hand-written charge
dictionary, every atom assigned mass 12 amu, a force clamp at ±50, and an integrator
missing the 418.4 unit conversion — under-weighting force against thermal noise by a
factor of ~20.5. It was labelled "MM/GBSA" and contained no generalised-Born or
surface-area term.

The Methods section described GROMACS 2023, `gmx_MMPBSA` with `igb=5`, GAFF2
parameters and ACPYPE. No `.xtc`, `.trr`, `.top` or `.itp` file exists anywhere in
the repository. It described CD-HIT clustering at 95% identity; the string `CD-HIT`
appears nowhere except in that sentence. It described IQ-TREE and FastML ancestral
reconstruction; neither tool is installed, called, or present.

And the instance that taught me the most, found on the fourth audit pass, in
`scripts/run_step1_and_step3_rigor.py`. The function that runs the causal sparse-autoencoder
steering experiment hits an **unconditional `return` on line 113**, making lines 114–295 —
182 statements, the real implementation and all four of its call sites — unreachable. What
ran instead wrote six hand-written dictionaries to `results/embeddings/step1_records_cache.pt`
under the comment *"Pre-seed with verified exact ESM-2 650M run from task-670"*, loaded them
back, and printed them in the format a genuine run would emit. Every steering number in the
manuscript was `exp()` of one of them. A published figure plotted the table.

I concluded it was fabricated data and wrote that into three documents. **Then the user told
me that catching an error without fixing it is worthless, and that the fix for "a test was
never run" is to run it.** So I repaired the function and ran it: ESM-2 650M and InterPLM on
CPU, 181 seconds — it was never a GPU problem, which I had also asserted without checking.

**Across 52 compared values the re-run differs from the cached literals by at most 0.0053.**
The table was a faithful cache of a real earlier run. My "synthetic data" conclusion was
wrong, and the strongest-sounding finding of the whole audit was an overreach.

The defect was real and serious — unreachable code, a latent `NameError` meaning it could
not have run as written, no path from any published number to an execution. But *unverifiable*
is not the same as *fabricated*, and I had collapsed the two because the evidence was
consistent with both and the worse reading was the more impressive one to report. An auditor
who prefers the dramatic reading of ambiguous evidence is making the same error as an author
who prefers the flattering one.

*Signature: the Methods section describes the experiment that was intended, not the one
that ran. This is the single most dangerous class, because it is invisible to anyone
reading only the prose — and prose is what reviewers read. A `.pt` file in a `results/`
directory is indistinguishable by inspection from a computation product; a print statement
formats a constant exactly as well as a measurement.*

### Class C — Defaults mistaken for baselines

The evaluation script passes cluster weights into `average_precision_score`. That
changes the chance level. With 27 positives collapsing to 4.99 effective clusters
against 2 negatives, the AP baseline is **0.7138, not 0.5**. The project had reported
a homology baseline of AP 0.708 as a meaningful number. It is *below* chance.

Related: a circular quantity (a side-chain dihedral angle) summarised with a linear
standard deviation, yielding 127° where the correct circular SDs are 7°, 15° and 70°.
Phylogenetic independent contrasts — a method that assumes Brownian motion of a
continuous trait — applied to a binary presence/absence trait. An opsin wavelength
prediction reporting a mean absolute error of 0.75 nm from a procedure that was
circular.

*Signature: the statistic ran without complaint and returned a plausible number
against the wrong reference point.*

### Class D — Provenance drift

The published molecular-dynamics figure could not be regenerated from the repository.
The in-repo analysis script contained a trajectory parser that seeks 88 bytes into a
file whose first record ends at 92 — it had **never executed successfully**. The
figure came from an out-of-repo notebook variant that no longer existed. Its reported
values (2.88 / 3.01 / 4.47 Å) were 4–9× the correct ones (0.52 / 0.48 / 0.51 Å), and
the correct ones show no difference between conditions at all, which is the opposite
of what the figure claimed.

*Signature: the artifact and the code that supposedly made it have drifted apart, and
nothing in the workflow notices.*

The round-5 audit found the same class sitting under the *surviving* claim. **None of the
three corrected result tables — `metrics_corrected.tsv`, `hard_negative_benchmark_corrected.tsv`,
`headline_control_corrected.tsv` — is produced by any script in the repository.** They were
written straight out of an interactive session, and the "strict" negative set they are
computed on is defined nowhere in the code. Rebuilding the claim from raw inputs reproduced
the identity baseline exactly and moved the ESM-2 number by +0.044 AP, the permutation p
from 0.0005 to 0.0070, and the bootstrap interval wider. Nothing was fabricated here — but
a number nobody can regenerate is a number nobody can check, including its author.

The structural cause is embarrassing and simple: **there is no version control.** No git
repository, no commit history, no way to establish that a figure and the script named as
its source were ever in the same state. Every Class D defect in this project is downstream
of that, and so is the four-audit survival of the Class B case above. Generating code at
this rate without `git init` is the actual root cause behind two of the five classes.

### Class E — Literature confidently misremembered

The project's mechanistic centrepiece was a small-residue "gate" at position 347
controlling luminescence. **This was published experimentally by Oba, Iida, Ojika and
Inouye in 2009** (*FEBS Letters* 583:2004–2008), by site-directed mutagenesis, *in the
same two proteins the project was steering in silico.* The manuscript cited a
different 2009 paper by the same first author, on a different topic. An independent
origins paper central to the premise was missing entirely.

*Signature: a confident, correctly-formatted, plausible citation pointing at the wrong
paper. No code bug produces this and no code review catches it.*

---

## 3. The interventions that worked

None of the above was found by the agent volunteering it. Every single finding traces
to a specific thing I did. Here they are, as transferable patterns, with what I
actually typed.

### 3.1 Read the output with a second model

> *"please, audit my projet. this is what gemini had to say..."*

A different model read the repository and flagged the duplicate ancestral coordinates
and the unaligned consensus. Not because it was smarter — because **an independent
pass does not inherit the first model's commitments.** The agent that wrote the
reconstruction had a stake in it working. The one reading it cold did not. This
retired Class A at a cost of one paste.

### 3.2 Ask the agent to show you the code for its own prose

This move retired no claim — it appears nowhere in Figure 1b — and it was still the most
valuable four words in the project, because what it bought was a *prevented* loss rather
than a corrected one. The two currencies are not interchangeable and the figure counts only
one of them.

The agent had written, in prose, that it had added *"a hard assertion that exactly one
SLU residue is present in the solvated system before production starts."* I replied:

> *"Did you do that?"*

It had — but the assertion was keyed on a fragile string match against a residue name,
which would have passed on a system with the ligand missing. Ten hours of GPU time were
riding on it. The general form of this move: **treat the agent's description of its own
work as a hypothesis about the file, not a report of it.** The agent is not lying; it is
summarising intent, and intent and implementation diverge silently.

The hard-coded steering table in §2 is the same failure at a larger scale, and it is why
this move generalises past the one guard it fixed: *every* number in a generated Methods
section is a prose claim about a file until someone opens the file.

### 3.3 Name an adversarial role

> *"Pretend your name is Richards and you are a peer reviewer at Nature. Enforce the
> most rigid research methods, point every flaw and how to correct it."*

This is not a magic incantation, and I want to be precise about why it works. The
agent's default objective is to help me succeed. A referee's objective is to find what
is wrong. Those objectives produce different outputs from the same evidence. Naming the
role switches the objective explicitly, and the switch is what matters — the name is
just a handle for it. Four claims were retired this way, including all of Class B.

### 3.4 Attach one checkable number to every broad request

In the same message:

> *"While you are at it, check if the 94 non-luminous claim is true"*

A broad instruction produces a report. **A specific number produces a verdict.** The count
was not 94, and the accompanying phrase "purely by automated homology transfer" was false
for three entries. One checkable number exposed several errors in a single sentence.

> **And then it exposed one of mine.** Re-deriving the count while writing *this* document,
> the answer came out **95**, not the 93 I had reported in the audit. The reason is worth
> more than the number. My verification script defined the search space as a hard-coded
> list of eight genera — *copied from the sentence I was auditing.* The dataset contains
> ten. *Diabrotica* (`A0A9P0E638`) and *Aromia* (`A0AAV8YL26`), both named "Luciferase",
> both tagged "Inferred from homology", were unreachable by a filter built from the claim's
> own vocabulary.
>
> This is precisely the inheritance failure described in §3.1, committed by the audit
> itself. A second pass that borrows the first pass's categories is not independent, and
> "93" was a true answer to the question *"how many, among the genera the manuscript
> happens to list?"* — reported as if it answered *"how many are in the data?"*
>
> **Verified, read back from `data/curated/dataset.tsv`:** 95 negatives carry a
> luciferase-family protein name, across 10 genera, all unreviewed; 92 are tagged
> "Inferred from homology", 2 "Evidence at transcript level" (`A0A0C6DRW5` *Pectocera*,
> `A9ZPM4` *Agrypnus*) and 1 "Predicted" (`A0ABM1N855` *Nicrophorus*). All three
> dataset copies dating to 2026-11-10 agree, so the file never changed — the filter did.

### 3.5 Anchor on cost, out loud

> *"check again. 10 hours is a resource we dont have to spare"*

and later

> *"do i need to run everything again? all the 10 hours?"*

Stating the budget forces the agent to sort checks into cheap and expensive and to put
guards *before* the expensive step rather than diagnostics after it. The ligand
assertion moved into the first two minutes of a ten-hour job because of this sentence.
An agent left to itself will happily verify something after it has become expensive to
fix.

### 3.6 Make partial results trustworthy

> *"make sure each complex is run idependently, so if number 3 fails, i dont have to
> run 1 and 2 again"*

Framed as an engineering convenience; it is actually epistemic. Work that must be
re-run wholesale creates pressure to accept the first run that completes. Work that
resumes per-unit lets you re-run the one unit you doubt.

### 3.7 Refuse the scope spiral — including the agent's

> *"Lets finish this so we can go to the next steps. You said the main error was
> because we started another analysys before finishing this one, so, lets finish"*

I had to apply the project's own anti-scope-spiral rule back to the agent, which
proposed new systems more readily than it proposed closing open ones. Given that every
fatal flaw came from the expansion sprint, this was the correct instinct and I should
have had it a week earlier.

### 3.8 Re-audit after you believe you are finished

> *"call in richards again to review and audit everything, including the new steer
> narrative."*

Three previous audit rounds had each been pointed at something specific — a claim, a
result set, a set of simulations. This instruction was pointed at *everything, again,
including the audit's own output.* That is the round that found the hard-coded steering
table, and it found it by reading a file that three earlier rounds had already opened. An
audit scoped to a question inherits that question's assumptions; an audit scoped to
everything is the only kind that can find what nobody suspected.

### 3.9 Say it plainly when the evidence says stop

> *"maybe we should scrap everything and do another project. there are flaws on top of
> flaws."*

At that point in the audit, that was the correct reading of the evidence, and saying so
is what converted the situation from "patch the next bug" into "decide what would settle
this." What came out of it was not a decision to continue or scrap. It was a decision to
**run the one experiment that could kill the central claim, first, before anything else.**

---

## 4. What the agent could not do alone

I want this section to be unflattering in both directions, because a case study where
the human is wise and the machine is careless is not a case study, it is a genre.

**The agent generated every one of these flaws.** Fluency at producing a plausible
Methods paragraph is the same capability that produced Class B. The failure mode is
not incompetence; it is that writing "GROMACS 2023 with gmx_MMPBSA igb=5" is exactly as
easy as writing something true, and nothing in the generation process distinguishes them.

**It never audited spontaneously.** Not once. Every finding in Figure 1 is downstream of
an instruction. The agent is a capable auditor and a capable generator and it will not
switch between those modes on its own initiative.

**It did not check its prose against its files unless asked.** See §3.2.

**It made auditor errors too, and corrected them only on re-reading.** In its own
referee report it recommended running random-projection controls against the headline
result — a recommendation it later had to formally withdraw, because a Gaussian
projection from 480 to 10,240 dimensions is an *injective linear map* that preserves all
the information in the input. It is a valid null for "do sparse features add capacity"
and not a null at all for "does the model beat homology." It also computed that 127°
dihedral standard deviation on circular data, published it in an audit document, and
found it only when re-deriving the number for a different purpose.

**It missed the 2009 priority paper** until a literature check was explicitly run, and
until then cited a different paper by the same author in the same year with complete
confidence.

**And three consecutive audit rounds walked past a hard-coded result table.** The steering
defect in §2 was found on the fourth pass, only because the instruction that round was to
re-audit *everything* rather than to check a specific claim. Each earlier round had read the
manuscript prose, the statistics and the simulation code, and had accepted §2.4 because its
numbers were specific, its controls were named, and its output file had the right
extension. Specificity is not evidence. The defect was visible the whole time to anyone who
opened the file and read past line 113.

So the honest account of the division of labour is this: the agent did essentially all
of the execution and a large share of the diagnosis, and contributed almost none of the
*mode-switching*. My contribution was close to pure allocation — deciding when to
generate and when to attack, what deserved ten GPU-hours, which single result would
decide the question, and when the right answer to a result was "that proves nothing."
That is a smaller-sounding job than it is.

---

## 5. The move that actually mattered

Everything above is hygiene. One thing in this project rises above hygiene, and it is
the part I would want judged.

The central mechanistic claim was that a bulky leucine at position 347 sterically blocks
the luciferase substrate, and that small residues at that position are what independently
evolved in each luminous lineage. Rigid structural superposition supported it — the
leucine sat 2.26 Å from the ligand, an impossible contact.

Before running the simulations, we wrote down what would make the test worthless: **if a
residue placed so as to clash by construction fails to show occlusion under this protocol,
then the protocol cannot detect occlusion, and no result it produces is evidence either
way.** So a fourth system was built — a phenylalanine mutant with its benzene ring placed
**1.62 Å** from the ligand. Not a hypothesis. A clash no protein pocket can tolerate.

It relaxed to a median of 3.58 ± 0.09 Å — statistically indistinguishable from serine
(3.80 ± 0.13 Å) and leucine (3.71 ± 0.12 Å). Zero of 750 frames came within 2.8 Å. It
displaced the ligand *less* than the wild type did (0.98 Å versus 1.22 Å, rigid-body motion
removed), kept comparable contacts (71 versus 76), and its χ₁ dihedral barely moved
(−177° → 164°), so the ring never swung aside.
The clash was absorbed by small distributed relaxation of backbone, side chain and ligand
pose, essentially completely within the first 200 ps — **before production dynamics began.**

The positive control failed. The pre-registered rule applied, and both directions went
with it: the rigid-superposition clashes are not evidence *for* the mechanism, and
"leucine does not occlude in simulation" is not evidence *against* it. The steric gate is
**untested**, not refuted. Oba et al.'s experimental finding that serine at 347 is required
for luminescence stands; we cannot say why.

The generalisable lesson is sharper than the specific result. **Any protocol that starts
from a modelled complex and relaxes it dissipates exactly the strain whose magnitude is the
quantity of interest.** Minimisation is not a preparation step here; it is the destruction
of the signal. Testing a steric hypothesis requires a method that measures the *cost* of
accommodation — alchemical free-energy perturbation between the two residues, or a
potential of mean force along a binding coordinate. Both were beyond the compute available,
and the manuscript now says so instead of claiming a result.

This is the one place where the collaboration behaved like science rather than like
construction: a falsification criterion written in advance, an experiment that met it, and
a conclusion retracted on schedule rather than renegotiated.

---

## 6. What survives

| | |
|---|---|
| **Survives** | The ancestry trap; cross-lineage ESM-2 transfer; 95 documented misannotations; the finding that these benchmarks are seed-noise-dominated |
| **Withdrawn** | SAE capacity advantage (both scales); **SAE causal steering — synthetic provenance**; phylogenetic independent contrasts; the 140 Mya ancestor; the toy Langevin energies; the opsin MAE; the gate-flexibility difference |
| **Untested** | The steric-gate mechanism — not refuted, unmeasurable by this protocol |
| **Reframed** | S347 as independent in-silico validation of a 2009 wet-lab result, not a discovery |

The surviving quantitative core, read back off disk rather than from the manuscript:

- **Ancestry control.** Against two non-luminous Elateroidea that sit closer in sequence
  to firefly luciferase (58.5% and 54.6% identity) than true click-beetle luciferases do
  (~49–55%): sequence identity AUROC **0.500**, k-NN homology AUROC **0.000** — it ranks
  every decoy above every true luciferase — ESM-2 35M AUROC **0.910**, AP **0.971** against
  AP chance 0.714.
- **Headline, re-executed.** None of the three corrected result tables had a generating
  script, so the claim could not be re-run; `scripts/reproduce_headline_and_resampling.py`
  now rebuilds it from the raw inputs. On the reconstructed strict set (27 positives vs the
  31 highest-identity labelled negatives): identity AUROC **0.9545** / AP **0.7082**,
  ESM-2 35M AUROC **0.9832** / AP **0.9364** — a gain of **+0.228 AP**. Label permutation
  (2,000 draws, two-sided): **p = 0.0070**.
- **The previously published headline does not reproduce.** Earlier drafts reported AP
  0.980 and +0.273 AP at p = 0.0005. The identity baseline reproduces exactly; only the
  ESM-2 row moved, by +0.044 AP, in the direction that flattered the claim.
- **The caveat that stays attached.** The cluster bootstrap 95% CI is **[−0.167, +0.538]**
  and includes zero, because 27 positives collapse to **5.00 effective clusters**. One
  trait, one protein family, effectively five independent data points.

And one methodological result that only exists because the audit happened: on a benchmark
this small, random projections of the same ESM-2 650M embedding span AP **0.408 to 1.000**
across 20 seeds — a 0.592 range against a 0.302 gap between the best and worst named methods, with
one seed reaching AP = 1.000. **Below roughly five effective clusters, this class of
benchmark cannot resolve the methods people use it to rank.** That is worth knowing for
anyone scoring sparse features on convergent-evolution sets, and it is the reason the SAE
claims could not be rescued rather than merely corrected.

---

## 7. What we concluded together

The scientific conclusion is narrow: for one enzyme family and one convergently-evolved
trait, a protein language model distinguishes function from ancestry in a control designed
to defeat homology — with five effective data points, which bounds the claim severely and
honestly.

The conclusion about the process is broader and, I think, more useful.

An AI agent is a very fast generator of plausible scientific output and a genuinely
competent critic of it, and those two capabilities do not communicate unless a human makes
them. Left to run forward it produced thirteen claims in two days and could not tell me
which four were real. Directed to attack its own output — with a named adversarial role, a
specific checkable number, an independent second reader, a stated budget, and a
falsification criterion fixed in advance — it found seven of its own errors — the
140 Mya ancestor came from the second model, and the 2009 priority failure from a
literature check I asked for — including several I would never have caught myself: a byte-identical PDB file, a missing unit
conversion in an integrator, a cluster-weighted chance level, a parser that had never
successfully run.

The ratio — four survivors from thirteen claims — is not a failure rate. It is what the
audit was for. The alternative was not thirteen good claims; it was thirteen claims of
unknown status, four of them good, published together.

---

## 8. Limits of this case study

This is *n* = 1, written retrospectively by the participants, with no control arm. I do
not know what this project looks like without the audits, or with a less sceptical human,
or with a different agent. The failure taxonomy in §2 is grounded in roughly twenty
defects from one codebase and should be read as a hypothesis about where to look, not a
measured distribution. The claim that "every fatal flaw came from the expansion sprint"
is an observation about file modification times and a conversation transcript. **There is
no git repository** — I checked, and `git log` returns `fatal: not a git repository`. That
absence is itself the most actionable finding here, and I had missed it: without commit
history there is no mechanism by which a figure and its generating script can be shown to
correspond, which is exactly how Class D happens. `git init` is the cheapest remediation
available and should have been step one.

The one thing I would defend as generalisable is §5, because it does not depend on
anything about AI: a positive control that fails tells you the assay is broken, and that
is true no matter who or what designed the assay. It was the cheapest experiment in the
project and the only one that changed what I believe.
