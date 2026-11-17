"""Reviewer R-4 controls: spatial specificity and dose-response for the SAE steering.

Both reuse the exact steering dictionaries and evaluator built by
run_step1_steering_controls(..., return_internals=True); nothing is reimplemented.

SPATIAL CONTROL
  Build a steering vector by the SAME procedure (SAE difference-in-means between held-in
  luminous and non-luminous training proteins, top-25 positive + top-25 negative features)
  at 16 DISTAL positions, each >= MIN_SEP residues in sequence from every pocket site, then
  measure the gate log-odds change at 347. If a distal perturbation moves 347 as much as the
  pocket perturbation does, the pocket-specificity claim fails.
  CAVEAT: "distal" is defined by sequence separation, not 3D distance.

DOSE-RESPONSE
  Sweep alpha over the top-50 pocket steer and record gate log-odds change and off-target
  scaffold recovery at each level. A monotone gate response with a flat scaffold response is
  the expected signature of a specific effect.

Usage: python scripts/run_steering_controls_R4.py
Writes: results/steering_controls_R4.tsv
"""
import csv, glob, os, sys, time
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from transformers import AutoTokenizer, EsmForMaskedLM            # noqa: E402
from Bio import Align                                             # noqa: E402
import fix_all_peer_review_flaws as fpr                           # noqa: E402
import run_step1_and_step3_rigor as R                             # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() - 2))
MIN_SEP = 25
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def main():
    t0 = time.time()
    al = Align.PairwiseAligner(); al.mode = "global"
    al.open_gap_score, al.extend_gap_score = -11, -1
    al.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    snap = glob.glob(os.path.join(ROOT, ".cache", "hf", "hub",
                                  "models--facebook--esm2_t33_650M_UR50D", "snapshots", "*"))[0]
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    mlm = EsmForMaskedLM.from_pretrained(snap, dtype=torch.float32, local_files_only=True).eval()
    sae = glob.glob(os.path.join(ROOT, ".cache", "hf", "hub",
                                 "models--Elana--InterPLM-esm2-650m", "**", "*.pt"), recursive=True)[0]
    sd = torch.load(sae, map_location="cpu", weights_only=False)
    print(f"[{time.time()-t0:.1f}s] models loaded", flush=True)

    _, I = R.run_step1_steering_controls(al, tok, mlm, sd, return_internals=True)
    pocket = I["pocket_steer_sites"]; W_dec = I["W_dec"]
    tp, tn = I["train_pos_hz"], I["train_neg_hz"]
    ev, geth = I["eval_steering_dict"], I["get_h18_z18"]
    print(f"[{time.time()-t0:.1f}s] steering internals rebuilt; pocket sites {pocket}", flush=True)

    # --- choose distal sites: >= MIN_SEP from every pocket site, present in all training maps
    ref_len = len(I["p08659_seq"])
    cand = [r for r in range(20, ref_len - 20)
            if min(abs(r - q) for q in pocket) >= MIN_SEP
            and all(r in p2t for p2t, _, _ in tp + tn)]
    distal = [cand[int(round(i * (len(cand) - 1) / (len(pocket) - 1)))] for i in range(len(pocket))]
    distal = sorted(set(distal))
    print(f"  distal sites (n={len(distal)}, >= {MIN_SEP} aa from any pocket site): {distal}", flush=True)

    # --- build the distal steer by the SAME construction
    steer_distal = {}
    for site in distal:
        zp = np.mean([z[p2t[site] - 1] for p2t, _, z in tp if site in p2t], axis=0)
        zn = np.mean([z[p2t[site] - 1] for p2t, _, z in tn if site in p2t], axis=0)
        dz = zp - zn
        sz = torch.zeros(10240)
        for fid in np.argsort(-dz)[:25]:
            sz[int(fid)] = float(dz[int(fid)])
        for fid in np.argsort(dz)[:25]:
            sz[int(fid)] = float(dz[int(fid)])
        steer_distal[site] = sz @ W_dec.T

    # match the distal steer's total L2 norm to the pocket steer's, so magnitude is controlled
    np_pocket = float(torch.sqrt(sum(torch.sum(v ** 2) for v in I["steer_top50"].values())))
    np_distal = float(torch.sqrt(sum(torch.sum(v ** 2) for v in steer_distal.values()))) + 1e-8
    for k in steer_distal:
        steer_distal[k] = steer_distal[k] * (np_pocket / np_distal)
    print(f"  pocket ||steer|| = {np_pocket:.3f}; distal rescaled to match", flush=True)

    out = []
    for name, seq in I["heldout_targets"]:
        p2t = fpr.map_positions(I["p08659_seq"], seq, al)
        h18, _ = geth(seq, None)
        for aa in ALPHAS:
            lo0, lo1, d, acc, kl = ev(seq, p2t, h18, I["steer_top50"], alpha=aa)
            out.append(dict(target=name, condition="pocket Top-50 SAE", alpha=aa,
                            d_logodds=f"{d:+.4f}", fold=f"{np.exp(d):.2f}",
                            scaffold_pct=f"{acc:.2f}", kl=f"{kl:.5f}"))
            print(f"  {name[:26]:26s} pocket  a={aa:4.1f}  d={d:+6.2f} ({np.exp(d):8.1f}x)  "
                  f"scaffold {acc:6.2f}%  KL {kl:.4f}", flush=True)
        for aa in ALPHAS:
            lo0, lo1, d, acc, kl = ev(seq, p2t, h18, steer_distal, alpha=aa)
            out.append(dict(target=name, condition="DISTAL control (matched norm)", alpha=aa,
                            d_logodds=f"{d:+.4f}", fold=f"{np.exp(d):.2f}",
                            scaffold_pct=f"{acc:.2f}", kl=f"{kl:.5f}"))
            print(f"  {name[:26]:26s} DISTAL  a={aa:4.1f}  d={d:+6.2f} ({np.exp(d):8.1f}x)  "
                  f"scaffold {acc:6.2f}%  KL {kl:.4f}", flush=True)

    dest = os.path.join(ROOT, "results", "steering_controls_R4.tsv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        wr.writeheader(); wr.writerows(out)
    print(f"\n[{time.time()-t0:.1f}s] {len(out)} rows -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
