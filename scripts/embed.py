"""Mean-pooled ESM-2 embeddings (every layer) for all curated sequences.

Runs offline from the local Hugging Face cache in .cache/hf.
Output: results/embeddings/<model>.npz with arrays
  ids     accession order
  layers  float16 array [n_seq, n_layers+1, dim] (index 0 = token embedding layer)
Usage: python scripts/embed.py facebook/esm2_t12_35M_UR50D
"""

import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer, EsmModel  # noqa: E402


def main(model_name):
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["label"] != "excluded" and not r["exact_duplicate_of"]]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).eval()
    torch.set_num_threads(os.cpu_count() or 4)

    out_dir = os.path.join(ROOT, "results", "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, model_name.split("/")[-1] + ".npz")

    feats = []
    t0 = time.time()
    with torch.inference_mode():
        for k, r in enumerate(rows):
            enc = tok(r["sequence"], return_tensors="pt")
            hs = model(**enc, output_hidden_states=True).hidden_states
            # Drop BOS/EOS tokens before mean pooling.
            feats.append(torch.stack([h[0, 1:-1].mean(0) for h in hs]).numpy().astype(np.float16))
            if k % 50 == 0:
                print(f"  {k}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_path, ids=np.array([r["accession"] for r in rows]), layers=np.stack(feats))
    print("saved", out_path, np.stack(feats).shape, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "facebook/esm2_t12_35M_UR50D"))
