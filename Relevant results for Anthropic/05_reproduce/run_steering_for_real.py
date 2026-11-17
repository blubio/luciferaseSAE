"""Run the REAL causal SAE steering experiment (round-7 audit).

scripts/run_step1_and_step3_rigor.py::run_step1_steering_controls() was repaired
(unconditional return + hard-coded result table deleted). This driver loads
ESM-2 650M and the InterPLM-650M layer-18 SAE from the local HF cache on CPU and
executes it, writing results/step1_steering_REAL.tsv.

Usage: python scripts/run_steering_for_real.py
"""
import csv, glob, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from transformers import AutoTokenizer, EsmForMaskedLM          # noqa: E402
from Bio import Align                                            # noqa: E402
import run_step1_and_step3_rigor as R                            # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() - 2))


def main():
    t0 = time.time()
    aligner = Align.PairwiseAligner()
    aligner.open_gap_score, aligner.extend_gap_score = -11, -1
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"

    snaps = glob.glob(os.path.join(ROOT, ".cache", "hf", "hub",
                                   "models--facebook--esm2_t33_650M_UR50D", "snapshots", "*"))
    assert snaps, "ESM-2 650M snapshot not found in the local cache"
    name = snaps[0]
    print(f"[{time.time()-t0:6.1f}s] loading ESM-2 650M from {name} (CPU, fp32)...", flush=True)
    tok650 = AutoTokenizer.from_pretrained(name, local_files_only=True)
    mlm650 = EsmForMaskedLM.from_pretrained(name, dtype=torch.float32, local_files_only=True).eval()
    print(f"[{time.time()-t0:6.1f}s] loaded ESM-2 650M", flush=True)

    sae_path = glob.glob(os.path.join(ROOT, ".cache", "hf", "hub",
                                      "models--Elana--InterPLM-esm2-650m", "**", "*.pt"),
                         recursive=True)
    assert sae_path, "InterPLM-650M SAE weights not found in the local cache"
    sd650 = torch.load(sae_path[0], map_location="cpu", weights_only=False)
    if not isinstance(sd650, dict) or "encoder.weight" not in sd650:
        sd650 = getattr(sd650, "state_dict", lambda: sd650)()
    print(f"[{time.time()-t0:6.1f}s] loaded SAE: "
          f"{[(k, tuple(v.shape)) for k, v in sd650.items() if hasattr(v, 'shape')]}", flush=True)

    recs = R.run_step1_steering_controls(aligner, tok650, mlm650, sd650)

    cols = ["target", "inp_mode", "d_top", "acc_top", "kl_top", "d_den", "acc_den", "kl_den",
            "d_ort", "acc_ort", "kl_ort", "d_rnd", "d_rnd_sd", "acc_rnd", "kl_rnd"]
    out = os.path.join(ROOT, "results", "step1_steering_REAL.tsv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        wr.writeheader()
        for r in recs:
            wr.writerow({c: r.get(c) for c in cols})
    print(f"\n[{time.time()-t0:6.1f}s] {len(recs)} records -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
