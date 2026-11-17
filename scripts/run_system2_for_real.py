"""Re-execute the out-of-clade RH1 opsin steering (System 2 fixes)."""
import glob, os, sys, time, json
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel
import fix_all_peer_review_flaws as fpr
torch.set_num_threads(max(1, os.cpu_count()-2))

def snap(repo):
    s = glob.glob(os.path.join(ROOT,".cache","hf","hub",f"models--facebook--{repo}","snapshots","*"))
    assert s, repo
    return s[0]

def sae(tag):
    p = glob.glob(os.path.join(ROOT,".cache","hf","hub",f"models--Elana--InterPLM-esm2-{tag}","**","*.pt"), recursive=True)
    return torch.load(p[0], map_location="cpu", weights_only=False)

t0=time.time()
al = fpr.make_aligner()
p8, p35, p650 = snap("esm2_t6_8M_UR50D"), snap("esm2_t12_35M_UR50D"), snap("esm2_t33_650M_UR50D")
tok8  = AutoTokenizer.from_pretrained(p8, local_files_only=True)
mod8  = EsmModel.from_pretrained(p8, local_files_only=True).eval()
tok35 = AutoTokenizer.from_pretrained(p35, local_files_only=True)
mod35 = EsmModel.from_pretrained(p35, local_files_only=True).eval()
tok650= AutoTokenizer.from_pretrained(p650, local_files_only=True)
mlm650= EsmForMaskedLM.from_pretrained(p650, dtype=torch.float32, local_files_only=True).eval()
print(f"[{time.time()-t0:.1f}s] models loaded", flush=True)

out = fpr.run_system2_fixes(al, tok8, mod8, sae("8m"), tok35, mod35, tok650, mlm650, sae("650m"))
print(f"\n[{time.time()-t0:.1f}s] run_system2_fixes returned {type(out).__name__} len={len(out) if hasattr(out,'__len__') else '-'}")
with open(os.path.join(ROOT,"results","system2_fixes_REAL.json"),"w") as fh:
    json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
print("wrote results/system2_fixes_REAL.json")
