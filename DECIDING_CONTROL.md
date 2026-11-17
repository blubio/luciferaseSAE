# The deciding control — result

## What was run
Random-projection controls plus two *paired* significance tests on the surviving headline
comparison (§2.1b): firefly-trained ESM-2 35M vs. nearest-neighbour sequence identity, strict
benchmark, 27 click-beetle positives (4.99 effective clusters) vs. 32 labelled negatives,
cluster-weighted, AP baseline 0.1849.

```
method                            AUROC      AP
Sequence identity (homology)      0.955   0.708
ESM-2 35M dense (headline)        0.994   0.980
Random Gaussian proj (10 seeds)   0.989   0.953 +/- 0.059   range [0.807, 0.998]
Random ReLU proj (10 seeds)       0.942   0.790 +/- 0.156   range [0.520, 1.000]
```

## Correction to the referee's own recommendation

The referee report told the authors to run "the same random-projection control" against the
headline result. **That was the wrong control for this question, and I withdraw the
recommendation as stated.**

A Gaussian projection R^480 -> R^10240 is an *injective linear map*: it preserves all of ESM-2's
information. It is a valid null for "is the SAE's advantage just dimensionality?" (audit M4),
because there the comparison is between two encodings of the same vector. It is **not** a null
for "does ESM-2 beat homology," because the projected features are still ESM-2 features. That
5/10 Gaussian seeds match the dense probe is therefore the *expected* result and carries no
information about the headline claim. Reported here only to prevent the same mistake recurring.

## The correct tests, and their disagreement

| Test | Question it answers | Result |
|---|---|---|
| Label permutation (2,000 draws) | Could this AP gap arise if labels were random? | **p = 0.0005** — no |
| Cluster bootstrap (2,000 draws) | Would the gap replicate on a different draw of clades? | **95% CI [-0.062, +0.583]** — cannot tell |

Observed difference +0.273 AP. The permutation null is centred at -0.001 +/- 0.085, so the
observed gap is >3 SD outside it. But only **63.2%** of cluster-bootstrap resamples favour
ESM-2, and the interval includes zero.

These are not contradictory; they answer different questions. Permutation conditions on the
observed taxa and asks whether the score-label association is real — it is. The bootstrap asks
whether the *margin over homology* is stable under resampling of the clades — at 4.99 effective
positive clusters, it is not.

## What may now be claimed

**Supported:** ESM-2 embeddings carry genuine, non-random information about luciferase function
that survives an ancestry-matched negative control (permutation p = 0.0005). On this dataset the
probe separates convergently evolved click-beetle luciferases from close non-luminous relatives
better than sequence identity does (AP 0.980 vs 0.708, AUROC 0.994 vs 0.955).

**Not supported:** that the margin over homology would replicate on an independent draw of
beetle clades. The cluster-bootstrap interval includes zero and must be reported alongside the
point estimate.

**Required wording change.** "ESM-2 beats homology on convergent evolution" becomes:
"In this system, ESM-2 separates convergent luciferases from ancestry-matched negatives
(permutation p = 0.0005), with a margin over sequence identity of +0.27 AP whose 95% cluster
bootstrap interval, [-0.06, +0.58], does not exclude zero at n = 4.99 effective positive
clusters."

That is a true sentence, and it is the paper's result.
