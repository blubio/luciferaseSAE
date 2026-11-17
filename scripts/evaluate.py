"""Cross-lineage evaluation: does a firefly-trained luciferase detector find
independently evolved (click beetle) luciferases better than homology does?

Pre-registered design (fixed before looking at test results):
  TRAIN positives : firefly (Lampyridae) luciferases
  TRAIN negatives : AMP-binding family proteins from non-luminous beetle genera in TRAIN_NEG_GENERA
  TEST positives  : click beetle luciferases (main test) and railroad-worm group (secondary)
  TEST negatives  : all non-luminous Elateroidea (ancestry-matched control: same superfamily
                    as fireflies) + non-luminous genera NOT in TRAIN_NEG_GENERA
  Near-identical positives are clustered (>=90% identity) and down-weighted so each cluster
  counts once (25 Pyrophorus colour variants would otherwise dominate).

Methods
  nn_identity   max % identity to any training luciferase           (homology baseline 1)
  knn_homology  max sim to train positives - max sim to train negs  (homology baseline 2)
  composition   logistic regression on amino-acid composition + length (sanity floor)
  esm_last      logistic regression on mean-pooled last-layer ESM-2 embeddings (main model)
  esm_cvlayer   same, layer chosen by leave-one-genus-out CV on TRAINING data only

Usage: python scripts/evaluate.py esm2_t12_35M_UR50D [esm2_t30_150M_UR50D ...]
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RNG = np.random.default_rng(0)

# Fixed a priori. Agrilus (jewel beetles, Elateriformia) is held out because it is the
# closest non-elateroid relative; the rest of the test genera were chosen to span the tree.
TEST_NEG_GENERA = {"Agrilus", "Tenebrio", "Leptinotarsa", "Nicrophorus"}
AA = "ACDEFGHIKLMNPQRSTVWY"
PID_CLUSTER = 90.0
NEG_CLUSTER_SIM = 0.85  # normalised-score threshold used to cluster negatives (approximate)


def load_rows():
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]


def single_linkage(n, linked):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in linked:
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def cluster_weights(idx, linked_pairs):
    """1/cluster_size weights for the items in idx."""
    pos_of = {g: k for k, g in enumerate(idx)}
    pairs = [(pos_of[i], pos_of[j]) for i, j in linked_pairs if i in pos_of and j in pos_of]
    roots = single_linkage(len(idx), pairs)
    size = defaultdict(int)
    for r in roots:
        size[r] += 1
    return np.array([1.0 / size[r] for r in roots]), roots


def balanced(y, w):
    w = w.copy()
    w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
    return w


def fit_lr(X, y, w):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), y, sample_weight=balanced(y, w))
    return lambda Z: lr.decision_function(sc.transform(Z))


def metric(y, s, w, n_boot=1000):
    auc = roc_auc_score(y, s, sample_weight=w)
    ap = average_precision_score(y, s, sample_weight=w)
    boots = []
    for _ in range(n_boot):
        b = RNG.integers(0, len(y), len(y))
        if len(set(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], s[b], sample_weight=w[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi, ap


def composition(seqs):
    out = []
    for s in seqs:
        c = np.array([s.count(a) for a in AA], dtype=float) / max(len(s), 1)
        out.append(np.append(c, np.log(len(s))))
    return np.array(out)


def main(models):
    rows = load_rows()
    sim_ids = open(os.path.join(RES, "similarity", "ids.txt")).read().split("\n")
    pos_ids = open(os.path.join(RES, "similarity", "pos_ids.txt")).read().split("\n")
    assert sim_ids == [r["accession"] for r in rows], "similarity matrix out of date: rerun identity.py"
    norm = np.load(os.path.join(RES, "similarity", "normscore.npy"))
    pid = np.load(os.path.join(RES, "similarity", "pid_vs_pos.npy"))
    pos_col = {a: k for k, a in enumerate(pos_ids)}
    n = len(rows)

    lab = np.array([r["label"] for r in rows])
    grp = np.array([r["split_group"] for r in rows])
    genus = np.array([r["genus"] for r in rows])
    seqs = [r["sequence"] for r in rows]

    train_pos = np.where((lab == "positive") & (grp == "firefly"))[0]
    train_neg = np.where((lab == "negative") & (grp == "nonlum_other") & ~np.isin(genus, list(TEST_NEG_GENERA)))[0]
    test_neg_elat = np.where((lab == "negative") & (grp == "nonlum_elateroid"))[0]
    test_neg_other = np.where((lab == "negative") & (grp == "nonlum_other") & np.isin(genus, list(TEST_NEG_GENERA)))[0]
    test_pos = {g: np.where((lab == "positive") & (grp == g))[0] for g in ["click_beetle", "railroad_worm_group"]}

    # Cluster weights: positives by % identity, negatives by normalised score.
    all_pos = np.where(lab == "positive")[0]
    pos_pairs = [(i, j) for i in all_pos for j in all_pos if i < j and pid[i, pos_col[rows[j]["accession"]]] >= PID_CLUSTER]
    neg_all = np.where(lab == "negative")[0]
    sub = norm[np.ix_(neg_all, neg_all)]
    neg_pairs = [(neg_all[a], neg_all[b]) for a, b in zip(*np.where(np.triu(sub >= NEG_CLUSTER_SIM, 1)))]
    w = np.ones(n)
    for idx, pairs in [(list(all_pos), pos_pairs), (list(neg_all), neg_pairs)]:
        ww, _ = cluster_weights(idx, pairs)
        w[idx] = ww

    print("== data")
    print(f"train: {len(train_pos)} firefly luciferases ({w[train_pos].sum():.1f} clusters), "
          f"{len(train_neg)} negatives from {sorted(set(genus[train_neg]))}")
    for g, ix in test_pos.items():
        print(f"test positives {g}: {len(ix)} seqs, {w[ix].sum():.1f} clusters")
    print(f"test negatives: {len(test_neg_elat)} non-luminous Elateroidea {sorted(set(genus[test_neg_elat]))}, "
          f"{len(test_neg_other)} other {sorted(set(genus[test_neg_other]))}")

    # Difficulty check: identity of test luciferases to the nearest firefly luciferase.
    tr_cols = [pos_col[rows[i]["accession"]] for i in train_pos]
    nn_pid = pid[:, tr_cols].max(1)
    for g, ix in test_pos.items():
        print(f"  {g}: identity to nearest firefly luciferase {nn_pid[ix].min():.0f}-{nn_pid[ix].max():.0f}%")
    for name, ix in [("non-lum Elateroidea negs", test_neg_elat), ("other test negs", test_neg_other), ("train negs", train_neg)]:
        if len(ix):
            print(f"  {name}: identity to nearest firefly luciferase {nn_pid[ix].min():.0f}-{nn_pid[ix].max():.0f}% (max)")

    # ---- scores for every sequence under each method
    scores = {}
    scores["nn_identity"] = nn_pid
    scores["knn_homology"] = norm[:, train_pos].max(1) - norm[:, train_neg].max(1)
    y_tr = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
    tr = np.r_[train_pos, train_neg]
    C = composition(seqs)
    scores["composition"] = fit_lr(C[tr], y_tr, w[tr])(C)

    layer_report = {}
    for m in models:
        emb = np.load(os.path.join(RES, "embeddings", m + ".npz"))
        assert list(emb["ids"]) == sim_ids, f"embeddings for {m} out of date: rerun embed.py"
        L = emb["layers"].astype(np.float32)
        scores[f"{m}:last"] = fit_lr(L[tr, -1], y_tr, w[tr])(L[:, -1])
        # Layer choice by leave-one-genus-out CV on the training set only.
        cv = []
        tr_genus = genus[tr]
        for layer in range(L.shape[1]):
            oof = np.zeros(len(tr))
            for g in sorted(set(tr_genus)):
                te_m = tr_genus == g
                f = fit_lr(L[tr[~te_m], layer], y_tr[~te_m], w[tr[~te_m]])
                oof[te_m] = f(L[tr[te_m], layer])
            cv.append(roc_auc_score(y_tr, oof, sample_weight=w[tr]))
        best = int(np.argmax(cv))
        layer_report[m] = (best, cv)
        scores[f"{m}:cvlayer{best}"] = fit_lr(L[tr, best], y_tr, w[tr])(L[:, best])

    # ---- metrics
    os.makedirs(RES, exist_ok=True)
    out_rows = []
    print("\n== cross-lineage results (cluster-weighted AUROC [95% bootstrap CI], average precision)")
    comparisons = []
    for g, ix in test_pos.items():
        comparisons += [
            (f"{g} vs ALL test negatives", ix, np.r_[test_neg_elat, test_neg_other]),
            (f"{g} vs non-lum Elateroidea (ancestry control)", ix, test_neg_elat),
            (f"{g} vs other non-lum beetles", ix, test_neg_other),
        ]
    for title, P, N in comparisons:
        if len(P) == 0 or len(N) == 0:
            continue
        print(f"\n-- {title}  (n+={len(P)}, n-={len(N)})")
        idx = np.r_[P, N]
        y = np.r_[np.ones(len(P)), np.zeros(len(N))]
        for meth, s in scores.items():
            auc, lo, hi, ap = metric(y, s[idx], w[idx])
            print(f"   {meth:34s} AUROC {auc:.3f} [{lo:.3f}-{hi:.3f}]  AP {ap:.3f}")
            out_rows.append({"comparison": title, "method": meth, "auroc": f"{auc:.4f}",
                             "ci_lo": f"{lo:.4f}", "ci_hi": f"{hi:.4f}", "ap": f"{ap:.4f}",
                             "n_pos": len(P), "n_neg": len(N)})
    with open(os.path.join(RES, "metrics.tsv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()), delimiter="\t")
        wr.writeheader()
        wr.writerows(out_rows)

    for m, (best, cv) in layer_report.items():
        print(f"\n{m}: training-CV AUROC by layer: " + " ".join(f"{v:.2f}" for v in cv) + f"  -> layer {best}")

    # ---- genome retrieval: rank all Ignelater AMP-binding proteins
    ign = np.where(genus == "Ignelater")[0]
    pyro_cols = [pos_col[rows[i]["accession"]] for i in all_pos if genus[i] == "Pyrophorus"]
    if len(ign) and pyro_cols:
        pid_pyro = pid[ign][:, pyro_cols].max(1)
        print(f"\n== genome retrieval: Ignelater luminosus, {len(ign)} AMP-binding proteins")
        print("   putative luciferase = >=80% identity to an experimentally cloned Pyrophorus luciferase")
        true_mask = pid_pyro >= 80
        for k in np.argsort(-pid_pyro)[:5]:
            print(f"   {rows[ign[k]]['accession']:12s} {pid_pyro[k]:.0f}% to Pyrophorus luc, "
                  f"{nn_pid[ign[k]]:.0f}% to nearest firefly luc, len {rows[ign[k]]['length']}")
        ret_rows = []
        for meth, s in scores.items():
            order = np.argsort(-s[ign])
            ranks = [int(np.where(order == k)[0][0]) + 1 for k in np.where(true_mask)[0]]
            print(f"   {meth:34s} rank of putative luciferase(s): {ranks} / {len(ign)}")
            ret_rows.append({"method": meth, "ranks": ";".join(map(str, ranks)), "n": len(ign)})
        with open(os.path.join(RES, "genome_retrieval_ignelater.tsv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["method", "ranks", "n"], delimiter="\t")
            wr.writeheader()
            wr.writerows(ret_rows)

    # ---- per-sequence scores for inspection
    with open(os.path.join(RES, "scores.tsv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, delimiter="\t")
        wr.writerow(["accession", "label", "split_group", "genus", "weight"] + list(scores))
        for i, r in enumerate(rows):
            wr.writerow([r["accession"], lab[i], grp[i], genus[i], f"{w[i]:.3f}"] + [f"{scores[m][i]:.4f}" for m in scores])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
