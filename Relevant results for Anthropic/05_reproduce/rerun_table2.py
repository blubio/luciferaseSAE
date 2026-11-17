"""Re-execute the Step-3A multi-PLM/SAE benchmark (Table 2) from cached representations."""
import glob, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from Bio import Align
import run_step1_and_step3_rigor as R
torch.set_num_threads(max(1, os.cpu_count()-2))

al = Align.PairwiseAligner(); al.mode="global"
al.open_gap_score, al.extend_gap_score = -11,-1
al.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

def sae(tag):
    p = glob.glob(os.path.join(ROOT,".cache","hf","hub",f"models--Elana--InterPLM-esm2-{tag}","**","*.pt"), recursive=True)
    return torch.load(p[0], map_location="cpu", weights_only=False) if p else None

t0=time.time()
out = R.run_step3a_multi_plm_benchmark(al, None, None, sae("8m"), None, None, None, None, None, None, sae("650m"))
print(f"\n[{time.time()-t0:.1f}s] returned: {type(out).__name__}")
import csv
summary = out[0] if isinstance(out, tuple) else out
cols = ["model","sys1_auroc","sys1_ap","ignelater_best_rank","rank_A0A8K0DA02","rank_A0A8K0D8G1",
        "sys2_r","sys2_mae_nm","sys2_auroc"]
dest = os.path.join(ROOT,"results","table2_multiplm_REAL.tsv")
with open(dest,"w",newline="",encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(cols)
    for row in summary:
        w.writerow([row[0]] + [("NA" if (isinstance(x,float) and x!=x) else (f"{float(x):.4f}" if isinstance(x,float) else int(x))) for x in row[1:]])
print("wrote", dest)
