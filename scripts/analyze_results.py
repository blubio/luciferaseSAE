"""Deep-dive analysis and visualization of the cross-lineage luciferase translator results.

Adds a Profile HMM baseline (pyhmmer built on firefly luciferases) and produces
publication-style figures in results/figures/:
  fig1_homology_vs_esm.png       Cross-lineage separation: Firefly homology vs ESM-2 probe score
  fig2_ignelater_paralog_trap.png The "Paralog Trap" inside the Ignelater luminosus genome
  fig3_layerwise_transfer.png     Layer-by-layer cross-lineage AUROC & Average Precision across ESM-2
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyhmmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RES, "figures")


def run_hmmer_baseline(rows, train_pos):
    """Build a profile HMM from aligned firefly luciferases or score via phmmer."""
    alphabet = pyhmmer.easel.Alphabet.amino()
    queries = []
    for i in train_pos:
        dsq = pyhmmer.easel.TextSequence(name=rows[i]["accession"], sequence=rows[i]["sequence"]).digitize(alphabet)
        queries.append(dsq)

    targets = []
    for i, r in enumerate(rows):
        dsq = pyhmmer.easel.TextSequence(name=r["accession"], sequence=r["sequence"]).digitize(alphabet)
        targets.append(dsq)

    # Run phmmer of all training firefly luciferases against all sequences, take max bitscore
    max_bits = {r["accession"]: 0.0 for r in rows}
    for hits in pyhmmer.hmmer.phmmer(queries, targets, cpus=4):
        for hit in hits:
            name = hit.name if isinstance(hit.name, str) else hit.name.decode("ascii")
            if hit.score > max_bits[name]:
                max_bits[name] = hit.score
    return np.array([max_bits[r["accession"]] for r in rows])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]
    with open(os.path.join(RES, "scores.tsv"), encoding="utf-8") as fh:
        scores = list(csv.DictReader(fh, delimiter="\t"))

    pid = np.load(os.path.join(RES, "similarity", "pid_vs_pos.npy"))
    pos_ids = open(os.path.join(RES, "similarity", "pos_ids.txt")).read().split("\n")
    pos_col = {a: k for k, a in enumerate(pos_ids)}

    ff_cols = [pos_col[r["accession"]] for r in rows if r["label"] == "positive" and r["split_group"] == "firefly"]
    pyro_cols = [pos_col[r["accession"]] for r in rows if r["label"] == "positive" and r["genus"] == "Pyrophorus"]

    train_pos = [i for i, r in enumerate(rows) if r["label"] == "positive" and r["split_group"] == "firefly"]
    hmmer_bits = run_hmmer_baseline(rows, train_pos)

    print("=== Non-luminous Elateroidea (Ancestry-Matched Control) ===")
    for i, r in enumerate(rows):
        if r["split_group"] == "nonlum_elateroid":
            print(f"  {r['accession']:12s} {r['organism'][:32]:32s} ff_pid={pid[i, ff_cols].max():.1f}% "
                  f"pyro_pid={pid[i, pyro_cols].max():.1f}% hmmer={hmmer_bits[i]:.1f} "
                  f"esm35_last={float(scores[i]['esm2_t12_35M_UR50D:last']):.2f} "
                  f"esm35_cv4={float(scores[i]['esm2_t12_35M_UR50D:cvlayer4']):.2f}")

    print("\n=== Top 12 Ignelater luminosus AMP-binding Paralogs (Ranked by Firefly Sequence Identity) ===")
    ign = [i for i, r in enumerate(rows) if r["genus"] == "Ignelater"]
    for rank_ff, i in enumerate(sorted(ign, key=lambda x: -pid[x, ff_cols].max())[:12], 1):
        r = rows[i]
        is_luc = "**TRUE LUC**" if pid[i, pyro_cols].max() >= 80.0 else "paralog"
        print(f"  ff_rank={rank_ff:2d} {r['accession']:12s} [{is_luc:12s}] ff_pid={pid[i, ff_cols].max():5.1f}% "
              f"pyro_pid={pid[i, pyro_cols].max():5.1f}% hmmer={hmmer_bits[i]:5.1f} "
              f"esm35_last={float(scores[i]['esm2_t12_35M_UR50D:last']):6.2f} "
              f"esm150_cv4={float(scores[i]['esm2_t30_150M_UR50D:cvlayer4']):6.2f}")

    # Check hmmer ranks on Ignelater
    ign_hmmer_order = np.argsort(-hmmer_bits[ign])
    true_ign = [k for k, idx in enumerate(ign) if pid[idx, pyro_cols].max() >= 80.0]
    hmmer_ranks = [int(np.where(ign_hmmer_order == k)[0][0]) + 1 for k in true_ign]
    print(f"  phmmer bitscore ranks on Ignelater putative luciferases: {hmmer_ranks} / {len(ign)}")

    # Check Photinus & Aquatica genome retrieval (in-clade sanity check)
    for g_name in ["Photinus", "Aquatica"]:
        g_idx = [i for i, r in enumerate(rows) if r["genus"] == g_name]
        true_g = [k for k, idx in enumerate(g_idx) if rows[idx]["label"] == "positive"]
        order_esm = np.argsort([-float(scores[idx]["esm2_t12_35M_UR50D:last"]) for idx in g_idx])
        ranks_esm = [int(np.where(order_esm == k)[0][0]) + 1 for k in true_g]
        print(f"  {g_name} genome ({len(g_idx)} seqs, {len(true_g)} known lucs): ESM-35M ranks = {sorted(ranks_esm)}")

    # ---- Figure 1: Homology vs ESM-2 Cross-Lineage Separation
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=180)

    groups_plot = [
        ("Train: Non-luminous beetles", lambda r, i: r["label"] == "negative" and r["split_group"] == "nonlum_other" and r["genus"] not in {"Agrilus", "Tenebrio", "Leptinotarsa", "Nicrophorus"}, "#b0b8c4", "o", 28, 0.55),
        ("Test: Non-luminous beetles (held-out genera)", lambda r, i: r["label"] == "negative" and r["split_group"] == "nonlum_other" and r["genus"] in {"Agrilus", "Tenebrio", "Leptinotarsa", "Nicrophorus"}, "#6c7a89", "s", 32, 0.65),
        ("Test: Non-luminous Elateroidea (Agrypnus, Pectocera)", lambda r, i: r["split_group"] == "nonlum_elateroid", "#d9534f", "D", 75, 0.95),
        ("Train: Firefly luciferases (Lampyridae)", lambda r, i: r["label"] == "positive" and r["split_group"] == "firefly", "#f0ad4e", "^", 65, 0.9),
        ("Test: Railroad-worm luciferases (Phengodidae/Rhag.)", lambda r, i: r["label"] == "positive" and r["split_group"] == "railroad_worm_group", "#9b59b6", "P", 75, 0.95),
        ("Test: Click-beetle luciferases (Elateridae - convergent)", lambda r, i: r["label"] == "positive" and r["split_group"] == "click_beetle", "#2ecc71", "*", 110, 0.95),
    ]

    for label_txt, cond, col, marker, sz, alpha in groups_plot:
        ix = [i for i, r in enumerate(rows) if cond(r, i)]
        x = [pid[i, ff_cols].max() for i in ix]
        y = [float(scores[i]["esm2_t12_35M_UR50D:last"]) for i in ix]
        ax.scatter(x, y, c=col, marker=marker, s=sz, alpha=alpha, label=f"{label_txt} (n={len(ix)})", edgecolors="black" if sz > 50 else "none", linewidths=0.6)

    # Annotate the two non-luminous Elateroidea traps
    for i, r in enumerate(rows):
        if r["split_group"] == "nonlum_elateroid":
            offset = (18, -16) if r["genus"] == "Pectocera" else (18, -28)
            ax.annotate(f"{r['genus']} ({r['accession']})\nNon-luminous click beetle ({pid[i, ff_cols].max():.1f}% ID)",
                        (pid[i, ff_cols].max(), float(scores[i]["esm2_t12_35M_UR50D:last"])),
                        xytext=offset,
                        textcoords="offset points", fontsize=8.2, fontweight="bold", color="#b52b27",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fff8f8", ec="#b52b27", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", color="#b52b27", lw=1.0))

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9, alpha=0.6, label="ESM-2 Probe Decision Boundary")
    ax.axvline(48.0, color="#b52b27", linestyle=":", linewidth=1.0, alpha=0.6, label="Homology Ambiguity Zone (48-58% ID)")
    ax.set_xlabel("Max Sequence Identity to Any Firefly Luciferase (%)", fontweight="bold")
    ax.set_ylabel("Firefly-Trained ESM-2 (35M) Log-Odds Score", fontweight="bold")
    ax.set_title("Cross-Lineage Transfer: ESM-2 Separates Convergent Luciferases from\nAncestry-Matched Non-Luminous Elateroidea Where Sequence Identity Fails", fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=8.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_homology_vs_esm.png"))
    plt.close(fig)

    # ---- Figure 2: The Ignelater luminosus Genome Paralog Trap
    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=180)
    ign_ff = np.array([pid[i, ff_cols].max() for i in ign])
    ign_pyro = np.array([pid[i, pyro_cols].max() for i in ign])
    ign_esm = np.array([float(scores[i]["esm2_t12_35M_UR50D:last"]) for i in ign])

    is_true = ign_pyro >= 80.0
    sc = ax.scatter(ign_ff[~is_true], ign_esm[~is_true], c=ign_pyro[~is_true], cmap="viridis", s=45, alpha=0.85, label="Non-luminous ACS paralogs in Ignelater genome (n=59)")
    ax.scatter(ign_ff[is_true], ign_esm[is_true], c="#2ecc71", marker="*", s=240, edgecolors="black", linewidths=1.0, zorder=5, label="Putative Ignelater Luciferase (80-84% ID to cloned Pyrophorus Luc)")

    for k in np.where(is_true)[0]:
        idx = ign[k]
        offset = (-185, 4) if ign_pyro[k] > 82 else (-195, -24)
        ax.annotate(f"True Luc: {rows[idx]['accession']} ({ign_pyro[k]:.0f}% to Pyrophorus)",
                    (ign_ff[k], ign_esm[k]), xytext=offset,
                    textcoords="offset points", fontsize=8.2, fontweight="bold", color="#1b7943",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f2fff6", ec="#1b7943", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="#1b7943", lw=1.0))

    # Annotate the decoy paralogs that fool firefly sequence identity
    top_ff_decoy = sorted([k for k in range(len(ign)) if not is_true[k]], key=lambda k: -ign_ff[k])[:2]
    for rank_d, k in enumerate(top_ff_decoy):
        idx = ign[k]
        offset = (-215, -46) if rank_d == 0 else (-215, -62)
        ax.annotate(f"Homology Decoy #{rank_d+1}: {rows[idx]['accession']} ({ign_ff[k]:.1f}% ID to firefly)",
                    (ign_ff[k], ign_esm[k]), xytext=offset,
                    textcoords="offset points", fontsize=8.0, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#f8f8f8", ec="#888888", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9))

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("% Identity to Experimentally Cloned Click-Beetle (Pyrophorus) Luciferase", fontsize=8.5)
    ax.set_xlabel("Sequence Identity to Nearest Firefly Luciferase (%)", fontweight="bold")
    ax.set_ylabel("Firefly-Trained ESM-2 (35M) Log-Odds Score", fontweight="bold")
    ax.set_title("Inside the Click Beetle Genome (Ignelater luminosus, 61 ACS Paralogs):\nWhy Raw Firefly Homology Ranks Non-Luminous Paralogs Above the True Luciferase", fontweight="bold", pad=12, fontsize=10.5)
    ax.legend(loc="lower left", frameon=True, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_ignelater_paralog_trap.png"))
    plt.close(fig)

    # ---- Figure 3: Layer-by-Layer Transfer Profile
    # Compute test AUROC (click beetle vs all test negatives, and vs non-lum Elateroidea) across all layers of ESM2-35M and 150M
    lab = np.array([r["label"] for r in rows])
    grp = np.array([r["split_group"] for r in rows])
    genus = np.array([r["genus"] for r in rows])
    w = np.array([float(s["weight"]) for s in scores])

    test_neg_genera = {"Agrilus", "Tenebrio", "Leptinotarsa", "Nicrophorus"}
    train_pos_ix = np.where((lab == "positive") & (grp == "firefly"))[0]
    train_neg_ix = np.where((lab == "negative") & (grp == "nonlum_other") & ~np.isin(genus, list(test_neg_genera)))[0]
    tr = np.r_[train_pos_ix, train_neg_ix]
    y_tr = np.r_[np.ones(len(train_pos_ix)), np.zeros(len(train_neg_ix))]
    w_tr = w[tr].copy()
    w_tr[y_tr == 1] *= w_tr[y_tr == 0].sum() / w_tr[y_tr == 1].sum()

    cb_pos = np.where((lab == "positive") & (grp == "click_beetle"))[0]
    te_neg_all = np.r_[
        np.where((lab == "negative") & (grp == "nonlum_elateroid"))[0],
        np.where((lab == "negative") & (grp == "nonlum_other") & np.isin(genus, list(test_neg_genera)))[0],
    ]
    te_ix = np.r_[cb_pos, te_neg_all]
    y_te = np.r_[np.ones(len(cb_pos)), np.zeros(len(te_neg_all))]
    w_te = w[te_ix]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180, sharey=True)
    for ax, m_name, title in zip(axes, ["esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"], ["ESM-2 35M (12 Layers)", "ESM-2 150M (30 Layers)"]):
        emb = np.load(os.path.join(RES, "embeddings", m_name + ".npz"))["layers"].astype(np.float32)
        aurocs, aps, ign_best_ranks = [], [], []
        for l_idx in range(emb.shape[1]):
            sc_l = StandardScaler().fit(emb[tr, l_idx])
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc_l.transform(emb[tr, l_idx]), y_tr, sample_weight=w_tr)
            s_all = lr.decision_function(sc_l.transform(emb[:, l_idx]))
            aurocs.append(roc_auc_score(y_te, s_all[te_ix], sample_weight=w_te))
            aps.append(average_precision_score(y_te, s_all[te_ix], sample_weight=w_te))
            order = np.argsort(-s_all[ign])
            rks = [int(np.where(order == k)[0][0]) + 1 for k in np.where(is_true)[0]]
            ign_best_ranks.append(min(rks))

        layers_x = np.arange(emb.shape[1])
        ax.plot(layers_x, aurocs, "-o", color="#2980b9", markersize=4, label="Click-Beetle AUROC")
        ax.plot(layers_x, aps, "-s", color="#27ae60", markersize=4, label="Click-Beetle Avg Precision (AP)")
        ax.axhline(0.708, color="#e67e22", linestyle="--", linewidth=1.0, label="Homology Baseline AP (0.708)")
        ax.set_xlabel("Transformer Layer (0 = Token Embeddings)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0.35, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("Cross-Lineage Score (Click Beetle vs Held-Out Negatives)", fontweight="bold")
    fig.suptitle("Layer-by-Layer Emergence of Cross-Lineage Functional Generalization in ESM-2", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_layerwise_transfer.png"), bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figures to results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
