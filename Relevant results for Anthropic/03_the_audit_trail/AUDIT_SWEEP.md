# Automated audit sweep

**15 findings.**


## A. Unreachable code after return/raise

- clean

## B. Suspiciously small .pt caches

- results\embeddings\step1_records_cache.HANDWRITTEN_NOT_COMPUTED.pt is only 2.7 kB - too small for embeddings; verify it holds computed values

## C. Manuscript numbers with no source file

- 0.0053
- 0.0058
- 0.273
- 0.337
- 0.378
- 0.408
- 0.538
- 0.583
- 0.592
- 0.596
- 0.688
- 0.735
- 0.837
- 19.55

## D. Figures with no generating script

- clean

## E. Tools named in Methods but absent from code

- clean

## F. Result tables with no generating script

- clean

## G. Leakage and data integrity

- clean
