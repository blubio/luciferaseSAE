"""Run System-1 fixes: includes Protocol B, neighbour-only steering (r != 347) onto a
masked 347 target -- the control that separates "the pocket context carries the signal" from
"perturbing 347 moves 347". Writes results/system1_fixes_REAL.json."""
import glob, os, sys, time, json
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["HF_HUB_OFFLINE"]="1"
sys.path.insert(0, os.path.join(ROOT,"scripts"))
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel
import fix_all_peer_review_flaws as fpr
torch.set_num_threads(max(1, os.cpu_count()-2))

def snap(repo):
    return glob.glob(os.path.join(ROOT,".cache","hf","hub",f"models--facebook--{repo}","snapshots","*"))[0]

t0=time.time()
al = fpr.make_aligner()
p150, p650 = snap("esm2_t30_150M_UR50D"), snap("esm2_t33_650M_UR50D")
tok150 = AutoTokenizer.from_pretrained(p150, local_files_only=True)
mod150 = EsmModel.from_pretrained(p150, local_files_only=True).eval()
tok650 = AutoTokenizer.from_pretrained(p650, local_files_only=True)
mlm650 = EsmForMaskedLM.from_pretrained(p650, dtype=torch.float32, local_files_only=True).eval()
sd650 = torch.load(glob.glob(os.path.join(ROOT,".cache","hf","hub",
          "models--Elana--InterPLM-esm2-650m","**","*.pt"), recursive=True)[0],
          map_location="cpu", weights_only=False)
print(f"[{time.time()-t0:.1f}s] models loaded", flush=True)

out = fpr.run_system1_fixes(al, tok150, mod150, tok650, mlm650, sd650)
print(f"\n[{time.time()-t0:.1f}s] returned {type(out).__name__}")
with open(os.path.join(ROOT,"results","system1_fixes_REAL.json"),"w") as fh:
    json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
print("wrote results/system1_fixes_REAL.json")
