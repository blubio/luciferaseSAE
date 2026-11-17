# Referee Report — Richards, on the holo molecular dynamics (§2.3b)

I re-derived every number in §2.3b from the deposited DCD files rather than from the authors'
figure. The headline conclusion — that the steric-occlusion mechanism is not supported — **is
sound and I endorse it**. But the section as written overstates part of what the data license,
and omits four controls a referee will demand. Details below.

---

## 1. What I verified independently, and confirm

**1.1 The ligand stayed bound in all nine runs.** This is the first thing that could have made
the result vacuous and it does not. Ligand centre-of-mass drift relative to the protein is
1.34–2.47 Å; ligand internal RMSD 1.10–1.99 Å; protein–ligand heavy-atom contacts within 4 Å
average 147–182 per frame and never fall below 128 in any frame of any run. The ligand is in the
pocket throughout. "No clash" is therefore a real observation, not a dissociation artefact.

**1.2 The systems are equilibrated.** First-half versus second-half median distances are
statistically indistinguishable (paired t across replicates: p = 0.479, 0.719, 0.160). No drift.

**1.3 The replicates are genuinely independent.** Gate χ₁ means differ between replicates of the
same system by more than one within-replicate standard deviation (e.g. steered system: +56°,
−80°, −36°). They are sampling, not three copies of one trajectory. I had expected this to be a
weakness — the replicates share one minimised structure and differ only in velocity seed — but
the gate region decorrelates.

**1.4 The decisive diagnostic, which the authors should foreground.** Gate–ligand minimum
distance at three stages:

| System | Rigid input PDB | After min + 200 ps equil | Equilibrium |
|---|---|---|---|
| True luciferase (S347) | 3.33 Å | 3.49 Å | 3.51 Å |
| **Non-luminous paralog (L347)** | **2.26 Å** | **3.59 Å** | **3.57 Å** |
| SAE-steered | 2.82 Å | 3.22 Å | 3.37 Å |

The L347 "clash" is gone **before production begins** — relieved by energy minimisation and
200 ps of equilibration. This single row is the cleanest statement of the result in the whole
study and currently appears nowhere in the manuscript. It should be Table 5's first column.

**1.5 An analysis the authors did not do, which strengthens their case.** §2.3b measures the
347–349 *triad*, which mixes residues of different side-chain size and is therefore not a clean
test of the gate residue. Isolating position 347 alone:

| System | res 347 | side-chain heavy atoms | median distance |
|---|---|---|---|
| True luciferase | SER | 2 | 3.90 ± 0.19 Å |
| **Non-luminous paralog** | **LEU** | **4** | **4.11 ± 0.66 Å** |
| SAE-steered | SER | 2 | 3.49 ± 0.19 Å |

The bulky leucine sits **further** from the ligand than serine does. This is the opposite of the
steric hypothesis in its strongest form and should replace, or at least accompany, the triad
measurement.

---

## 2. MAJOR — The claim of "no difference between systems" is not supported by this design

§2.3b reports "no pairwise difference is significant (p = 0.345, 0.197, 0.095)" and the figure
panel is titled "Differences are within replicate scatter". **Absence of significance at n = 3 is
not evidence of equivalence**, and the manuscript edges toward asserting the latter.

Power analysis on the authors' own data: with n = 3 per group and the observed between-replicate
SD of 0.347 Å, the smallest difference detectable at 80% power (α = 0.05) is **1.07 Å**. The
observed S347-vs-L347 difference is 0.22 Å — a factor of five below the detection floor. This
design cannot distinguish "no difference" from "a real difference of up to ~1 Å."

**Correction.** Separate the two claims explicitly, because they have very different evidential
status:

- **Well supported (categorical):** L347 does not grossly occlude the pocket. Zero clash frames
  in 750. By the rule of three, the 95% upper bound on the true clash frequency is **0.40%**.
  A mechanism requiring steric exclusion of the substrate is incompatible with this.
- **Not supported (underpowered):** that S347 and L347 clearances are *equal*. Do not claim it.
  Retitle the figure panel to "No gross occlusion in any system" and state the 1.07 Å floor.

---

## 3. MAJOR — Four missing controls

**3.1 One protonation state.** Only the anionic DLSA was simulated. The authors' own notebook
documents the neutral form as a one-line switch and advises running both "if the gate result is
load-bearing." It is load-bearing — it overturns the paper's mechanism. A reviewer will ask
whether a neutral ligand, with different electrostatics at the sulfamate, packs differently.
**Run three replicates of the neutral form for complex 2 at minimum.**

**3.2 No positive control.** Nothing in this experiment demonstrates that the protocol *can*
detect steric occlusion when it exists. Without that, "we saw no clash" is weak. **Add an
in-silico bulky mutant** (e.g. S347W or S347F in complex 1). If tryptophan also shows no
occlusion, the assay is insensitive and §2.3b proves nothing; if it does occlude, the negative
result for L347 is meaningful.

**3.3 Timescale.** 2.5 ns excludes fast local occlusion only. Loop or domain motions gating
substrate access occur on µs–ms timescales and are entirely unsampled. The text says this; the
abstract does not.

**3.4 Single force field, no crystallographic anchor.** AMBER14/OpenFF 2.2.0 only. More
importantly, two of three systems are **AlphaFold models**, not experimental structures, and the
ligand pose is transplanted by superposition. The authors should report backbone RMSD to the
starting model over the trajectory, and state plainly that a modelled pocket may relax
differently from a real one.

---

## 4. MINOR

1. Table 5 reports "min (Å)" as a mean of per-replicate minima. A minimum over a sample is a
   biased, sample-size-dependent statistic. Report the 1st percentile instead, or the minimum
   over the pooled 750 frames, and say which.
2. §2.3b states the mutant "is the system that comes closest to the ligand." True (2.77 Å), but
   its χ₁ standard deviation is 127° in one replicate — the steered serine is conformationally
   disordered. That is worth a sentence; it may matter more than the distance.
3. The alternative mechanisms offered (transition-state stabilisation, H-bonding, solvation) are
   appropriately flagged as untested, but three untested hypotheses in a results section invites
   the question of why none was tested. One sentence on feasibility would close it.
4. State the production length (2.5 ns) in the abstract, not only in Methods. A reader who sees
   "nine simulations" will assume more sampling than this.

---

## Assessment

The authors set out to test their own central mechanism, built the controls to do it properly
(ligand parameterised, replicates, motif-anchored residue selection, a guard that caught a
real stereochemistry defect), and reported a result that destroys their own prior claim. The
analysis is reproducible from the deposited files — I checked.

The conclusion stands. What needs fixing is calibration: one claim in §2.3b is strongly supported
and one is underpowered, and the section currently presents them with equal confidence. Add the
positive control, run the second protonation state, separate the two claims, and this is a clean
negative result of the kind the field publishes too rarely.

**Recommendation: minor revision.**

**Richards**
