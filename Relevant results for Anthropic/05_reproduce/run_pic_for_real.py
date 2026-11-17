"""Re-execute Step 3B: phylogeny, ASR and phylogenetic independent contrasts (opsin + System 1)."""
import glob, os, sys, time, csv
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
summary, exp_records, s1_vecs, s2_vecs = R.run_step3a_multi_plm_benchmark(
    al, None, None, sae("8m"), None, None, None, None, None, None, sae("650m"))
print(f"[{time.time()-t0:.1f}s] step3a done; {len(exp_records)} opsin records", flush=True)

pic_op, pic_s1 = R.run_step3b_phylogeny_asr_and_pic(al, exp_records, s2_vecs)
pic_true_op, pic_pred_op, r_op, p_op = pic_op
pheno, esm, ident, r_esm, p_esm, r_id, p_id = pic_s1

dest = os.path.join(ROOT,"results","pic_REAL.tsv")
with open(dest,"w",newline="",encoding="utf-8") as fh:
    w=csv.writer(fh,delimiter="\t"); w.writerow(["analysis","quantity","value"])
    w.writerow(["opsin_lambda_max","n_contrasts",len(pic_true_op)])
    w.writerow(["opsin_lambda_max","pic_pearson_r",f"{r_op:.4f}"])
    w.writerow(["opsin_lambda_max","pic_pearson_p",f"{p_op:.3e}"])
    w.writerow(["system1_luminescence","n_contrasts",len(pheno)])
    w.writerow(["system1_luminescence","pic_r_esm_vs_phenotype",f"{r_esm:.4f}"])
    w.writerow(["system1_luminescence","pic_p_esm_vs_phenotype",f"{p_esm:.3e}"])
    w.writerow(["system1_luminescence","pic_r_identity_vs_phenotype",f"{r_id:.4f}"])
    w.writerow(["system1_luminescence","pic_p_identity_vs_phenotype",f"{p_id:.3e}"])
print(f"\n[{time.time()-t0:.1f}s] wrote {dest}")
print(f"  OPSIN  PIC r = {r_op:+.4f} (p = {p_op:.3e}), n = {len(pic_true_op)} contrasts")
print(f"  SYS1   PIC r(ESM-2) = {r_esm:+.4f} (p = {p_esm:.3e}) | r(identity) = {r_id:+.4f} (p = {p_id:.3e}), n = {len(pheno)}")
