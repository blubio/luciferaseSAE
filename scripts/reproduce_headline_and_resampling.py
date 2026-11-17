"""Reproduce the surviving headline claim and its resampling statistics from raw inputs.

Written 2026-11-16 during the round-5 audit, because NO script in the repository
produced results/headline_control_corrected.tsv, results/hard_negative_benchmark_corrected.tsv
or results/metrics_corrected.tsv -- they were written directly from an interactive
session and could not be regenerated.

IMPORTANT CAVEAT ON THE NEGATIVE SET
------------------------------------
The "strict" negative set used by the published tables (n- = 32) is not defined anywhere
in this repository. It is reconstructed here as the 31 labelled negatives with the highest
identity to any training (firefly) luciferase, which reproduces the published
nn_identity row exactly (AUROC 0.9545, AP 0.7082 vs published 0.9545 / 0.7079).
Exact membership of the original set cannot be recovered. Numbers below are therefore a
reconstruction, not a re-execution.

Usage: python scripts/reproduce_headline_and_resampling.py
Writes: results/headline_reproduced.tsv
"""
import csv, os, sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as EV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
N_DRAWS = 2000
SEED = 0


def build():
    rows = EV.load_rows()
    sim = os.path.join(RES, "similarity")
    pos_ids = open(os.path.join(sim, "pos_ids.txt")).read().split("\n")
    norm = np.load(os.path.join(sim, "normscore.npy"))
    pid = np.load(os.path.join(sim, "pid_vs_pos.npy"))
    pos_col = {a: k for k, a in enumerate(pos_ids)}
    lab = np.array([r["label"] for r in rows])
    grp = np.array([r["split_group"] for r in rows])
    genus = np.array([r["genus"] for r in rows])
    n = len(rows)

    all_pos = np.where(lab == "positive")[0]
    pos_pairs = [(i, j) for i in all_pos for j in all_pos
                 if i < j and pid[i, pos_col[rows[j]["accession"]]] >= EV.PID_CLUSTER]
    neg_all = np.where(lab == "negative")[0]
    sub = norm[np.ix_(neg_all, neg_all)]
    neg_pairs = [(neg_all[a], neg_all[b])
                 for a, b in zip(*np.where(np.triu(sub >= EV.NEG_CLUSTER_SIM, 1)))]

    w = np.ones(n)
    roots = np.zeros(n, dtype=int)
    for off, (idxs, pairs) in enumerate([(list(all_pos), pos_pairs), (list(neg_all), neg_pairs)]):
        ww, rr = EV.cluster_weights(idxs, pairs)
        w[idxs] = ww
        for a, r in zip(idxs, rr):
            roots[a] = r + off * 100000

    train_pos = np.where((lab == "positive") & (grp == "firefly"))[0]
    train_neg = np.where((lab == "negative") & (grp == "nonlum_other")
                         & ~np.isin(genus, list(EV.TEST_NEG_GENERA)))[0]
    nn_pid = pid[:, [pos_col[rows[i]["accession"]] for i in train_pos]].max(1)
    test_pos = np.where((lab == "positive") & (grp == "click_beetle"))[0]
    strict_neg = neg_all[np.argsort(-nn_pid[neg_all])][:31]

    L = np.load(os.path.join(RES, "embeddings", "esm2_t12_35M_UR50D.npz"))["layers"].astype(np.float32)
    tr = np.r_[train_pos, train_neg]
    y_tr = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
    esm = EV.fit_lr(L[tr, -1], y_tr, w[tr])(L[:, -1])
    return test_pos, strict_neg, nn_pid, esm, w, roots


def main():
    rng = np.random.default_rng(SEED)
    P, N, nn_pid, esm, w, roots = build()
    idx = np.r_[P, N]
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    sI, sE, ww = nn_pid[idx], esm[idx], w[idx]

    ap_i = average_precision_score(y, sI, sample_weight=ww)
    ap_e = average_precision_score(y, sE, sample_weight=ww)
    d = ap_e - ap_i

    perm = np.array([average_precision_score(yp, sE, sample_weight=ww)
                     - average_precision_score(yp, sI, sample_weight=ww)
                     for yp in (rng.permutation(y) for _ in range(N_DRAWS))])
    pval = (np.sum(np.abs(perm) >= abs(d)) + 1) / (N_DRAWS + 1)

    cl = roots[idx]
    uc = np.unique(cl)
    boot = []
    for _ in range(N_DRAWS):
        sel = np.concatenate([np.where(cl == c)[0] for c in rng.choice(uc, len(uc), replace=True)])
        if len(set(y[sel])) < 2:
            continue
        boot.append(average_precision_score(y[sel], sE[sel], sample_weight=ww[sel])
                    - average_precision_score(y[sel], sI[sel], sample_weight=ww[sel]))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    out = [
        {"quantity": "n_positives", "value": f"{len(P)}"},
        {"quantity": "effective_positive_clusters", "value": f"{ww[y == 1].sum():.2f}"},
        {"quantity": "n_negatives_reconstructed", "value": f"{len(N)}"},
        {"quantity": "identity_auroc", "value": f"{roc_auc_score(y, sI, sample_weight=ww):.4f}"},
        {"quantity": "identity_ap", "value": f"{ap_i:.4f}"},
        {"quantity": "esm2_35M_auroc", "value": f"{roc_auc_score(y, sE, sample_weight=ww):.4f}"},
        {"quantity": "esm2_35M_ap", "value": f"{ap_e:.4f}"},
        {"quantity": "delta_ap", "value": f"{d:+.4f}"},
        {"quantity": "permutation_p_two_sided", "value": f"{pval:.4f}"},
        {"quantity": "permutation_null_mean_sd", "value": f"{perm.mean():+.4f} +/- {perm.std():.4f}"},
        {"quantity": "bootstrap_ci95_lo", "value": f"{lo:+.4f}"},
        {"quantity": "bootstrap_ci95_hi", "value": f"{hi:+.4f}"},
        {"quantity": "bootstrap_frac_favouring_esm2", "value": f"{(boot > 0).mean():.3f}"},
        {"quantity": "n_clusters_resampled", "value": f"{len(uc)}"},
        {"quantity": "n_draws", "value": f"{N_DRAWS}"},
        {"quantity": "seed", "value": f"{SEED}"},
    ]
    path = os.path.join(RES, "headline_reproduced.tsv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["quantity", "value"], delimiter="\t")
        wr.writeheader()
        wr.writerows(out)
    for r in out:
        print(f"  {r['quantity']:32s} {r['value']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
