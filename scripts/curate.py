"""Build a labelled beetle luciferase / ACS-paralog dataset from raw UniProt TSVs.

Label rules (see README "Label hygiene"):
  positive   - named luciferase / luciferin 4-monooxygenase AND experimental support
               (Swiss-Prot reviewed, or evidence at transcript/protein level)
               AND from a luminous beetle family.
  negative   - AMP-binding family member from a non-luminous beetle family.
  ambiguous  - known weak "luciferase-like" activity in a non-luminous beetle; never trained on.
  unlabeled  - anything from a luminous genome without experimental support
               (homology-only names are NOT trusted). Used only for genome retrieval tests.

Output: data/curated/dataset.tsv and data/curated/sequences.fasta
"""

import csv
import glob
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "curated")

LUMINOUS_FAMILIES = {"Lampyridae", "Elateridae", "Phengodidae", "Rhagophthalmidae"}
# Lineage groups for the hold-out split. Fallon et al. 2018 support an origin of
# Elateridae luciferase independent of Lampyridae; the placement of Phengodidae /
# Rhagophthalmidae origins is treated as uncertain and evaluated separately.
SPLIT_GROUP = {
    "Lampyridae": "firefly",
    "Elateridae": "click_beetle",
    "Phengodidae": "railroad_worm_group",
    "Rhagophthalmidae": "railroad_worm_group",
}
EXPERIMENTAL = {"Evidence at transcript level", "Evidence at protein level"}
MIN_LEN, MAX_LEN = 400, 700
# Elateridae is mostly non-luminous; only these genera are luminous in our data.
LUMINOUS_ELATERID_GENERA = {"Pyrophorus", "Ignelater", "Cryptalaus", "Photophorus", "Pyrearinus", "Fulgeochlizus"}
# Non-luminous beetle proteins with reported weak luciferase-like activity.
AMBIGUOUS_GENERA = {"Zophobas"}


def rank(lineage, rank_name):
    for item in lineage.split(", "):
        if item.endswith(f"({rank_name})"):
            return item.rsplit(" (", 1)[0]
    return ""


def load_raw():
    rows = {}
    for fn in sorted(glob.glob(os.path.join(RAW, "*.tsv"))):
        with open(fn, encoding="utf-16") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if not r.get("Entry"):
                    continue
                r["source_file"] = os.path.basename(fn)
                rows.setdefault(r["Entry"], r)
    return list(rows.values())


def label(r):
    name = r["Protein names"].lower()
    family = rank(r["Taxonomic lineage"], "family")
    genus = rank(r["Taxonomic lineage"], "genus")
    length = int(r["Length"])
    reviewed = r["Reviewed"].strip().lower() == "reviewed"
    experimental = reviewed or r["Protein existence"] in EXPERIMENTAL
    luc_name = "luciferase" in name or "luciferin 4-monooxygenase" in name
    luminous = family in LUMINOUS_FAMILIES and (family != "Elateridae" or genus in LUMINOUS_ELATERID_GENERA)

    # Genus-name queries (e.g. taxonomy_name:Aquatica) also match bacteria with
    # species epithets like "aquatica"; keep beetles only.
    if rank(r["Taxonomic lineage"], "order") != "Coleoptera":
        return family, genus, "excluded", "not Coleoptera (query spill-over)"
    if not (MIN_LEN <= length <= MAX_LEN):
        return family, genus, "excluded", f"length {length} outside {MIN_LEN}-{MAX_LEN}"
    if genus in AMBIGUOUS_GENERA:
        return family, genus, "ambiguous", "reported weak luciferase-like activity"
    if luminous and luc_name and experimental:
        return family, genus, "positive", "luciferase name + experimental evidence"
    if luminous:
        why = "homology-only luciferase name" if luc_name else "paralog in luminous genome"
        return family, genus, "unlabeled", why
    if family in LUMINOUS_FAMILIES:
        # Non-luminous click beetles (e.g. Agriotes wireworms): ancestry-matched negatives.
        return family, genus, "negative", "AMP-binding family, non-luminous genus of a luminous family"
    return family, genus, "negative", "AMP-binding family, non-luminous lineage"


def main():
    rows = load_raw()
    os.makedirs(OUT, exist_ok=True)
    seen_seq = {}
    out = []
    for r in sorted(rows, key=lambda x: (x["Reviewed"] != "reviewed", x["Entry"])):
        family, genus, lab, why = label(r)
        superfamily = rank(r["Taxonomic lineage"], "superfamily")
        if lab == "negative":
            group = "nonlum_elateroid" if superfamily == "Elateroidea" else "nonlum_other"
        else:
            group = SPLIT_GROUP.get(family, "other")
        seq = r["Sequence"].strip()
        dup_of = seen_seq.get(seq, "")
        if not dup_of:
            seen_seq[seq] = r["Entry"]
        out.append({
            "accession": r["Entry"],
            "label": lab,
            "reason": why,
            "split_group": group,
            "family": family,
            "genus": genus,
            "organism": r["Organism"],
            "protein_name": r["Protein names"],
            "reviewed": r["Reviewed"],
            "existence": r["Protein existence"],
            "length": r["Length"],
            "exact_duplicate_of": dup_of,
            "source_file": r["source_file"],
            "sequence": seq,
        })

    with open(os.path.join(OUT, "dataset.tsv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(OUT, "sequences.fasta"), "w", encoding="utf-8") as fh:
        for o in out:
            if o["label"] != "excluded" and not o["exact_duplicate_of"]:
                fh.write(f">{o['accession']} {o['label']} {o['genus']}\n{o['sequence']}\n")

    kept = [o for o in out if not o["exact_duplicate_of"]]
    print(f"{len(out)} entries, {len(kept)} unique sequences\n")
    print("label x split_group (unique sequences):")
    for (lab, grp), n in sorted(Counter((o["label"], o["split_group"]) for o in kept).items()):
        print(f"  {lab:10s} {grp:22s} {n}")
    print("\npositives by genus:")
    for g, n in Counter(o["genus"] for o in kept if o["label"] == "positive").most_common():
        print(f"  {g:20s} {n}")
    print("\nnegatives by genus:")
    for g, n in Counter(o["genus"] for o in kept if o["label"] == "negative").most_common():
        print(f"  {g:20s} {n}")
    print("\nnegatives that UniProt names 'luciferase' (homology annotations in non-glowing beetles):")
    for o in kept:
        if o["label"] == "negative" and ("luciferase" in o["protein_name"].lower() or "luciferin" in o["protein_name"].lower()):
            print(f"  {o['accession']:12s} {o['organism'][:35]:35s} {o['protein_name'][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
