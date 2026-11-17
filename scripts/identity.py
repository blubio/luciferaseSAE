"""Pairwise sequence similarity for the homology baselines and clustering.

Outputs (results/similarity/):
  ids.txt          accession order of the matrices
  normscore.npy    all-vs-all normalised global alignment score
                   s(a,b) / sqrt(s(a,a) * s(b,b)), BLOSUM62, gap open -10 / extend -0.5
  pid_vs_pos.npy   percent identity of every sequence vs every positive
                   (identical aligned positions / length of the shorter sequence)
  pos_ids.txt      column order of pid_vs_pos
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "similarity")

_aligner = None


def make_aligner():
    a = PairwiseAligner()
    a.mode = "global"
    a.substitution_matrix = substitution_matrices.load("BLOSUM62")
    a.open_gap_score = -10
    a.extend_gap_score = -0.5
    # Free end gaps so N/C-terminal length differences are not penalised.
    a.end_gap_score = 0
    return a


def _init():
    global _aligner
    _aligner = make_aligner()


def clean(seq):
    # BLOSUM62 lacks some ambiguity codes; map rare letters to X.
    return "".join(c if c in "ARNDCQEGHILKMFPSTWYVBZX" else "X" for c in seq)


def _score_row(args):
    i, seqs = args
    return i, [_aligner.score(seqs[i], seqs[j]) for j in range(i, len(seqs))]


def _pid(args):
    i, j, a, b = args
    aln = _aligner.align(a, b)[0]
    ident = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        ident += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return i, j, 100.0 * ident / min(len(a), len(b))


def load():
    path = os.path.join(ROOT, "data", "curated", "dataset.tsv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]
    return rows


def main():
    rows = load()
    ids = [r["accession"] for r in rows]
    seqs = [clean(r["sequence"]) for r in rows]
    pos_idx = [k for k, r in enumerate(rows) if r["label"] == "positive"]
    n = len(seqs)
    os.makedirs(OUT, exist_ok=True)
    print(f"{n} sequences, {len(pos_idx)} positives, {os.cpu_count()} CPUs", flush=True)

    t0 = time.time()
    raw = np.zeros((n, n), dtype=np.float32)
    with Pool(initializer=_init) as pool:
        for done, (i, row) in enumerate(pool.imap_unordered(_score_row, [(i, seqs) for i in range(n)], chunksize=4)):
            raw[i, i:] = row
            raw[i:, i] = row
            if done % 50 == 0:
                print(f"  scores: {done}/{n} rows, {time.time() - t0:.0f}s", flush=True)
    d = np.sqrt(np.clip(np.diag(raw), 1e-9, None))
    norm = raw / np.outer(d, d)
    np.save(os.path.join(OUT, "normscore.npy"), norm)

    t0 = time.time()
    pid = np.zeros((n, len(pos_idx)), dtype=np.float32)
    jobs = [(i, jj, seqs[i], seqs[j]) for i in range(n) for jj, j in enumerate(pos_idx)]
    with Pool(initializer=_init) as pool:
        for done, (i, jj, v) in enumerate(pool.imap_unordered(_pid, jobs, chunksize=64)):
            pid[i, jj] = v
            if done % 5000 == 0:
                print(f"  identity: {done}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, "pid_vs_pos.npy"), pid)

    with open(os.path.join(OUT, "ids.txt"), "w") as fh:
        fh.write("\n".join(ids))
    with open(os.path.join(OUT, "pos_ids.txt"), "w") as fh:
        fh.write("\n".join(ids[k] for k in pos_idx))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
