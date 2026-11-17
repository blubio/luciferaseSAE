"""Robustness of the headline margin to the negative-set cutoff (reviewer H-3).

The 'strict' negative set is defined by a cutoff on identity-rank. This sweeps that cutoff
from n-=10 to the full labelled negative set and reports the ESM-2 minus identity AP margin
at each, so the reader can see whether the headline depends on the choice.

Usage: python scripts/sweep_negative_set_cutoff.py
Writes: results/negative_set_cutoff_sweep.tsv
"""
import csv, os, sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as EV
import reproduce_headline_and_resampling as RH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    P, _, nn_pid, esm, w, _ = RH.build()
    rows = EV.load_rows()
    lab = np.array([r["label"] for r in rows])
    neg_all = np.where(lab == "negative")[0]
    order = neg_all[np.argsort(-nn_pid[neg_all])]

    out = []
    for k in [10, 20, 31, 40, 60, 80, 100, 140, len(neg_all)]:
        N = order[:k]
        idx = np.r_[P, N]
        y = np.r_[np.ones(len(P)), np.zeros(len(N))]
        ai = average_precision_score(y, nn_pid[idx], sample_weight=w[idx])
        ae = average_precision_score(y, esm[idx], sample_weight=w[idx])
        ui = roc_auc_score(y, nn_pid[idx], sample_weight=w[idx])
        ue = roc_auc_score(y, esm[idx], sample_weight=w[idx])
        out.append(dict(n_neg=k, ap_identity=f"{ai:.4f}", ap_esm2=f"{ae:.4f}",
                        delta_ap=f"{ae-ai:+.4f}", auroc_identity=f"{ui:.4f}", auroc_esm2=f"{ue:.4f}"))
        print(f"  n-={k:4d}  AP identity {ai:.4f}  AP ESM-2 {ae:.4f}  dAP {ae-ai:+.4f}")

    dest = os.path.join(ROOT, "results", "negative_set_cutoff_sweep.tsv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        wr.writeheader(); wr.writerows(out)
    d = [float(r["delta_ap"]) for r in out]
    print(f"\nmargin range across cutoffs: {min(d):+.4f} to {max(d):+.4f} (spread {max(d)-min(d):.4f})")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
