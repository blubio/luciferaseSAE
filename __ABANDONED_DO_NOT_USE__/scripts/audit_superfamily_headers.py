import os
import json

ROOT = r"c:\Users\pagli\Desktop\translator"
DATA_DIR = os.path.join(ROOT, "data", "superfamilies_50")

for sf_id in sorted(os.listdir(DATA_DIR)):
    sf_dir = os.path.join(DATA_DIR, sf_id)
    if not os.path.isdir(sf_dir):
        continue
    j_path = os.path.join(sf_dir, "sequences.json")
    if not os.path.exists(j_path):
        continue
    with open(j_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    print(f"=== {sf_id}: {d.get('name')} ===")
    for acc, info in d.get("functional_sequences", {}).items():
        print(f"  [FUNC]    {acc}: {info['header'][:80]}")
    for acc, info in d.get("paralog_sequences", {}).items():
        print(f"  [PARALOG] {acc}: {info['header'][:80]}")
