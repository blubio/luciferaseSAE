"""Fetch every external structure this project depends on into <repo>/data/structures.

Round-10: the project read ten structure files from a disposable agent workspace
(.gemini/antigravity/brain/<uuid>/), which made several analyses unreproducible once that
directory became unreadable. This script makes the repository self-contained.

Usage: python scripts/fetch_structures.py
Needs network. Prints a report of what is present and what is still missing.
"""
import os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "structures")

PDB = {                      # RCSB entries
    "pdb_00004g36.cif": "4G36",   # firefly luciferase + DLSA (reference pocket)
    "pdb_00006aaa.cif": "6AAA",   # Amydetes firefly luciferase
    "pdb_00007kyd.cif": "7KYD",   # Drosophila CG6178 ACS
    "pdb_00001u19.cif": "1U19",   # bovine rhodopsin + retinal
}
AFDB = {                     # AlphaFold DB models, by UniProt accession
    "AF-Q9U4U7.cif": "Q9U4U7", "AF-Q9U4U8.cif": "Q9U4U8", "AF-Q7M4K3.cif": "Q7M4K3",
    "AF-A0A8K0DA02.cif": "A0A8K0DA02", "AF-A0A8K0C7P5.cif": "A0A8K0C7P5",
    "AF-A0A0C6DRW5.cif": "A0A0C6DRW5",
}


def get(url, dest, timeout=90):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        b = r.read()
    open(dest, "wb").write(b)
    return len(b)


def main():
    os.makedirs(DEST, exist_ok=True)
    have, got, missing = [], [], []
    for fn, pid in PDB.items():
        p = os.path.join(DEST, fn)
        if os.path.exists(p) and os.path.getsize(p) > 50_000:
            have.append(fn); continue
        try:
            n = get(f"https://files.rcsb.org/download/{pid}.cif", p)
            got.append(f"{fn} ({n/1e6:.2f} MB)")
        except Exception as e:
            missing.append(f"{fn} [RCSB {pid}]: {type(e).__name__}")
    for fn, acc in AFDB.items():
        p = os.path.join(DEST, fn)
        if os.path.exists(p) and os.path.getsize(p) > 50_000:
            have.append(fn); continue
        last = None
        for v in (6, 5, 4, 3, 2, 1):      # AFDB model version varies by entry
            try:
                n = get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.cif", p)
                got.append(f"{fn} (v{v}, {n/1e6:.2f} MB)")
                last = None
                break
            except Exception as e:
                last = f"{type(e).__name__}"
        if last:
            missing.append(f"{fn} [AFDB {acc}]: {last} (tried model_v6..v1)")

    print(f"already present : {len(have)}  {have}")
    print(f"downloaded      : {len(got)}  {got}")
    print(f"STILL MISSING   : {len(missing)}")
    for m in missing:
        print("   ", m)
    if missing:
        print("\nIf this machine has no network, download them manually into:")
        print("   ", DEST)
        for fn, pid in PDB.items():
            if any(fn in m for m in missing):
                print(f"    https://files.rcsb.org/download/{pid}.cif   -> {fn}")
        for fn, acc in AFDB.items():
            if any(fn in m for m in missing):
                print(f"    https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.cif   -> {fn}")
                print(f"      (if 404, try v5/v4/v3; or search {acc} at https://alphafold.ebi.ac.uk/)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
