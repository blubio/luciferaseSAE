"""Regenerate fig12 from results/holo_md_summary.tsv.

Round-5 audit fix: the previous fig12 was rendered in an interactive session with no
script and no saved table, so its numbers could not be checked. Run
scripts/analyze_holo_md.py first to produce the table, then this.

Usage: python scripts/plot_fig12_holo.py
Writes: results/figures/fig12_holo_four_systems.png
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["1_A0A8K0DA02_TrueLuc_WT_S347_holo", "2_A0A8K0C7P5_Paralog_WT_L347_holo",
         "3_A0A8K0C7P5_Paralog_SAESteered_S347A348N349_holo",
         "4_A0A8K0DA02_PositiveControl_S347F_holo"]
LABEL = {ORDER[0]: "true\nluciferase\nSer", ORDER[1]: "non-luminous\nparalog\nLeu",
         ORDER[2]: "SAE-steered\nparalog\nSer", ORDER[3]: "positive control\nPhe\n(built clashing)"}
COL = {ORDER[0]: "#1b7837", ORDER[1]: "#b2182b", ORDER[2]: "#4a6fa5", ORDER[3]: "#6a3d9a"}
START = {ORDER[0]: 3.33, ORDER[1]: 2.26, ORDER[2]: 2.82, ORDER[3]: 1.62}  # rigid-model input


def main():
    path = os.path.join(ROOT, "results", "holo_md_summary.tsv")
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    by = {s: [r for r in rows if r["system"] == s] for s in ORDER}
    assert all(len(v) == 3 for v in by.values()), "expected 3 replicates per system"

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    x = np.arange(4)

    ax = axes[0]
    for i, s in enumerate(ORDER):
        d = [float(r["d_median"]) for r in by[s]]
        ax.scatter([i] * 3, d, s=46, color=COL[s], zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(d)] * 2, color=COL[s], lw=2)
        ax.scatter([i], [START[s]], marker="v", s=60, facecolor="none", edgecolor=COL[s], zorder=3)
        ax.annotate("", xy=(i, np.mean(d)), xytext=(i, START[s]),
                    arrowprops=dict(arrowstyle="->", color=COL[s], lw=1.0, alpha=.7))
    ax.axhline(2.8, ls=(0, (4, 2)), lw=1, color="0.45")
    ax.text(3.45, 2.84, "steric clash", fontsize=7.5, ha="right", color="0.35")
    ax.set_ylabel("gate 347 to ligand, min distance (Å)")
    ax.set_title("Minimisation erases every starting clash", loc="left")
    ax.set_ylim(1.3, 4.4)
    ax.text(0.02, 0.04, "▽ rigid model input   ● replicate after MD", transform=ax.transAxes, fontsize=7.5, color="0.35")

    ax = axes[1]
    for i, s in enumerate(ORDER):
        v = [float(r["com_max"]) for r in by[s]]
        ax.scatter([i] * 3, v, s=46, color=COL[s], zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(v)] * 2, color=COL[s], lw=2)
    ax.set_ylabel("ligand COM drift (Å)\nrigid-body motion removed")
    ax.set_title("Phe displaces the ligand no more than Ser", loc="left")
    ax.set_ylim(0, 2.4)

    ax = axes[2]
    for i, s in enumerate(ORDER):
        v = [int(r["n_below_clash"]) for r in by[s]]
        ax.bar([i], [sum(v)], color=COL[s], width=.55)
        ax.text(i, sum(v) + 0.4, str(sum(v)), ha="center", fontsize=8)
    ax.set_ylabel("frames < 2.8 Å (of 750)")
    ax.set_title("No system occludes — including the one built to", loc="left")
    ax.set_ylim(0, 19)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=7.5)
        ax.set_xlim(-0.5, 3.5)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "fig12_holo_four_systems.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
