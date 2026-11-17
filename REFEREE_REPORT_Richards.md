# Referee Report — Richards
**MS:** "Do Protein Language Models Encode Convergent Function or Shared Ancestry?"
**Recommendation: Reject as submitted; resubmission invited after major revision.**

I have verified every claim I could reach by re-running the authors' own data. The revised manuscript is markedly more honest than earlier drafts — three claims have already been withdrawn on the authors' own controls, which I credit. But I have found **one issue that is disqualifying in its present form** and which no amount of additional computation will fix: a priority failure. I detail it first.

---

## 1. FATAL — The central mechanistic claim was published experimentally in 2009 and is not cited

The manuscript's §2.2 presents the `L347`/`S347` steric gate as a mechanism the model *revealed*: "Superposition of 9 experimental and AlphaFold structures onto the SLU-bound pocket **revealed the physical mechanism** separating true luciferases from higher-identity decoys." §2.4 then presents SAE steering toward `L347 → S` as a causal prediction recovered from the model.

**Both recapitulate Oba, Iida, Ojika & Inouye (2009), *FEBS Letters* 583:2004–2008, "Functional conversion of fatty acyl-CoA synthetase to firefly luciferase by site-directed mutagenesis: A key substitution responsible for luminescence activity."** That paper states its conclusion in the title. It performed the mutation experimentally in exactly the two proteins this manuscript steers *in silico* — *Agrypnus binodulus* AbLL and *Drosophila* CG6178 — and reported that both "gave luminescence enhancement, suggesting that the serine residue is a key substitution responsible for luminescence activity."

The reference list contains a different Oba 2009 paper (luciferin biosynthesis, *PLoS ONE*). The FEBS Letters paper is absent. So is Fallon et al. (2018) *eLife*, which establishes the independent-origins premise the entire study rests on and which the authors' own README cites.

This is not a missing-citation quibble. As written, the manuscript claims as discovery a result that is sixteen years old, and tests its model against a "held-out" target whose answer is in the literature.

**How to correct.** Invert the framing completely, and the work becomes *stronger*:

> Oba et al. (2009) established by site-directed mutagenesis that restoring Ser347 confers luminescence activity on AbLL and CG6178. We ask whether an unsupervised sparse autoencoder, trained on no luciferase labels, independently localises the same residue. It does: steering the top-50 Layer-18 features shifts position 347 toward Ser/Ala by 324–992×, on proteins held out of training, while leaving 99.8% of the scaffold unchanged.

That is a **validated** interpretability result against a wet-lab ground truth — rare, and far more persuasive than an unverifiable novel claim. It costs the authors nothing but the word "revealed."

---

## 2. FATAL — "94 mislabelled luciferases" does not reproduce, and the mechanism claimed for it is wrong

Abstract and §2.1 claim **94** UniProt entries mislabelled as luciferases in non-luminous beetles "purely by automated homology transfer."

Recomputed from `dataset.tsv` (rows whose `protein_name` matches /lucifer/i, in the eight genera the authors name):

```
Nicrophorus 42 · Agrilus 15 · Tribolium 15 · Sitophilus 8 · Dendroctonus 6
Tenebrio 5 · Agrypnus 1 · Pectocera 1          TOTAL = 93, not 94
```

I could not reconstruct 94 under any variant: excluded rows in these genera = 0, flagged duplicates = 0. The number is off by one and must be corrected or its derivation stated.

**The more serious error is the mechanism.** Of the 93, **three are not homology-inferred**:

| Accession | Organism | UniProt evidence |
|---|---|---|
| `A0A0C6DRW5` | *Pectocera fortunei* | **Evidence at transcript level** |
| `A9ZPM4` | *Agrypnus binodulus* | **Evidence at transcript level** |
| `A0ABM1N855` | *Nicrophorus vespilloides* | Predicted |

"Purely by automated homology transfer" is therefore false for 3/93 entries. Correct to "90 of 93 carry the evidence tag *Inferred from homology*; three do not."

**And this is where the two errors collide.** `A9ZPM4` is AbLL — the Oba et al. protein. `A0A0C6DRW5` is its *Pectocera* counterpart. **These two sequences are the entire n⁻ = 2 ancestry-matched negative set** on which the manuscript's surviving headline result depends. They are not database detritus; they are cDNA-cloned, experimentally characterised enzymes from the exact papers the manuscript fails to cite.

---

## 3. MAJOR — The binary "non-luminous" label is not safe for AbLL

Oba et al. (2009) report that AbLL possesses "luminescence activity, but with extremely low catalytic efficiency," and Adams et al. (cited by the authors) describe AbLL as a **"second latent luciferase."** Oba, Shintani et al. separately show *A. binodulus* lacks luciferin *in vivo*, so the animal does not glow — the negative label is defensible at organism level. But it is not a clean enzymatic negative.

The README acknowledges exactly this fuzziness for the *Zophobas* enzyme and classes it `ambiguous`. The same standard must apply to AbLL. With n⁻ = 2, reclassifying one negative changes the headline AUROC materially.

**How to correct.** State in §2.1b that one of the two ancestry-matched negatives is a characterised latent luciferase, cite Oba 2009 and Adams 2020, and report the control both ways (n⁻ = 2 and n⁻ = 1). If the result survives only with AbLL scored as negative, say so.

---

## 4. MAJOR — Statistical power is below the threshold at which any ranking claim is admissible

The authors have already disclosed this (§2.5, Limitations 5), and their own 650M control is the cleanest demonstration I have seen in a submitted manuscript: across 20 random-projection seeds, AP spans **0.408–1.000**, a range of 0.592, against a 0.302 gap between the best and worst *named* methods. One random seed achieved AP = 1.000.

I endorse the authors' conclusion and push it further: **this applies to the surviving positive result too.** The strict-benchmark gap (ESM-2 AP 0.943 vs identity 0.708) rests on 4.99 effective positive clusters, and the manuscript nowhere demonstrates that *this* gap exceeds what a random 10,240-d projection achieves on the same split.

**How to correct.** Run the identical random-projection control against the §2.1b headline comparison. If ESM-2's margin over homology also sits inside the seed envelope, the paper has no positive result and must be reframed entirely as a methodological cautionary study — which would still be publishable, and arguably more useful.

---

## 5. MAJOR — Unresolved items carried from the authors' own audit

- **PIC on a binary trait.** Felsenstein contrasts assume a continuous Brownian character. The authors disclose this but do not repair it. Use a threshold model (Felsenstein 2005) or phylogenetic logistic regression (Ives & Garland 2010), or remove §2.7 item 2.
- **In-sample contamination in §2.7.** 98 of the 228 contrast sequences trained the probe. The ESM-2 contrast is partly in-sample; the identity contrast is not. The comparison is not like-for-like even before Williams' test.
- **No MSA/profile baseline.** `phmmer` appears in the README but not in the manuscript's reported baselines. A PSSM or HMM profile is the standard comparator for remote-homology function transfer and its absence will be the first thing any reviewer asks for.

---

## 6. MINOR

1. §2.3 is now sound, but the apo MD adds nothing to the argument. Consider moving to Supplementary.
2. Methods §4.2 gives a steering equation with an $\alpha$ coefficient whose value is never reported.
3. Reference numbering: the list jumps after the withdrawn Valenzano entry.
4. `README.md` still advertises the withdrawn claims (+0.199 SAE gain, MAE = 0.75 nm, "94"). It must be reconciled with the manuscript before release.

---

## Assessment

The question is excellent and the convergent-evolution design is the right instrument for it. The authors' willingness to run controls that destroyed three of their own claims is, frankly, better scientific conduct than most submissions I review.

But the manuscript currently claims priority over a 2009 experimental result it does not cite, and its single surviving positive claim has not been subjected to the same random-projection control that demolished its others. Fix the attribution, run that control, and report the outcome honestly — whichever way it falls — and this becomes a solid methods paper about the limits of small convergent-evolution benchmarks, with a genuine SAE-validation result attached.

**Richards**
