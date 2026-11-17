"""Regenerate fig13 (spatial + dose-response steering controls) from
results/steering_controls_R4.tsv.  Usage: python scripts/plot_fig13_steering_controls.py"""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POCKET, DISTAL = "pocket Top-50 SAE", "DISTAL control (matched norm)"
CP, CD, GREY = "#1b7837", "#b2182b", "#8c8c8c"


def main():
    src = os.path.join(ROOT, "results", "steering_controls_R4.tsv")
    rows = list(csv.DictReader(open(src, encoding="utf-8"), delimiter="\t"))
    series = defaultdict(list)
    for r in rows:
        series[(r["target"], r["condition"])].append(
            (float(r["alpha"]), float(r["d_logodds"]), float(r["scaffold_pct"])))
    for k in series:
        series[k].sort()
    targets = sorted({r["target"] for r in rows})

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for i, t in enumerate(targets):
        for cond, col, ls, lab in [(POCKET, CP, "-", "pocket (16 sites incl. 347)"),
                                   (DISTAL, CD, "--", "distal 16 sites, matched norm")]:
            s = series[(t, cond)]
            axes[0].plot([x[0] for x in s], [x[1] for x in s], ls, color=col, marker="o",
                         ms=4, lw=1.4, alpha=.85, label=lab if i == 0 else None)
            axes[1].plot([x[0] for x in s], [x[2] for x in s], ls, color=col, marker="o",
                         ms=4, lw=1.4, alpha=.85)
    axes[0].axhline(0, lw=.8, color=GREY)
    axes[0].set_xlabel(r"steering strength $\alpha$")
    axes[0].set_ylabel(r"$\Delta$ gate-347 log-odds")
    axes[0].set_title("Pocket steering is dose-dependent; distal is flat", loc="left")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5, handlelength=1.6)
    axes[0].text(0.97, 0.50, f"{len(targets)} held-out targets\n(one line each)",
                 transform=axes[0].transAxes, ha="right", va="center", fontsize=7, color=GREY)
    axes[1].set_xlabel(r"steering strength $\alpha$")
    axes[1].set_ylabel("off-target top-1 recovery (%)")
    axes[1].set_ylim(98.0, 100.25)
    axes[1].set_title("Distal vector is not inert - it just misses 347", loc="left")
    axes[1].text(0.02, 0.06, "distal perturbation degrades the scaffold as much as\n"
                             r"the pocket one, while moving gate-347 by $\Delta\approx0$",
                 transform=axes[1].transAxes, fontsize=7, color=GREY)
    for ax, L in zip(axes, "ab"):
        ax.text(-0.12, 1.03, L, transform=ax.transAxes, fontweight="bold", fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "fig13_steering_controls.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
