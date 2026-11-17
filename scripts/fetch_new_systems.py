# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polite-http",
# ]
# ///
"""Fetch all UniProt sequences, RCSB PDB structures, and AlphaFold models for:
  System 2: Deep-Sea & Cavefish Visual Opsins (RH1 & Cone Opsins) + PDB 1U19 (Rhodopsin + RET)
  System 3: Antarctic Notothenioid vs Arctic Cod Antifreeze Glycoproteins (AFGPs) & Polar Fish AFPs
            + Progenitor Homologs (Trypsinogens, C-type Lectins, Sialic Acid Synthases) + PDB 2PY2, 1HG7, 1WFA
"""

import csv
import gzip
import json
import os
import sys
import urllib.request

SKILL_UNIPROT = r"C:\Users\pagli\.gemini\config\plugins\science\skills\uniprot_database\scripts"
sys.path.insert(0, SKILL_UNIPROT)
import uniprot_tools  # noqa: E402
from polite_http import http_client  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR = r"C:\Users\pagli\.gemini\antigravity\brain\c5b593ce-c44a-4462-9ed6-c68fd6a10dd0"
STRUCT_DIR = os.path.join(ART_DIR, "structures")
os.makedirs(STRUCT_DIR, exist_ok=True)

OPSIN_DIR = os.path.join(ROOT, "data", "raw_opsins")
AFGP_DIR = os.path.join(ROOT, "data", "raw_afgp")
os.makedirs(OPSIN_DIR, exist_ok=True)
os.makedirs(AFGP_DIR, exist_ok=True)

FIELDS = [
    "accession", "id", "gene_names", "protein_name", "organism_name", "organism_id",
    "lineage", "length", "protein_existence", "reviewed", "xref_pdb", "cc_function", "sequence",
]


def fetch_uniprot_tsv(query: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        print(f"[UniProt] Already exists: {out_path}")
        return
    cnt = uniprot_tools.get_count(query, "uniprotkb")
    print(f"[UniProt] Query: {query}\n          Count: {cnt} -> {out_path}")
    if int(cnt) == 0:
        return
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for chunk in uniprot_tools.stream_results(query, "uniprotkb", output_format="tsv", fields=FIELDS):
            fh.write(chunk)
            if not chunk.endswith("\n"):
                fh.write("\n")


def fetch_pdb_cif(pdb_id: str):
    pdb_lower = pdb_id.lower()
    cif_out = os.path.join(STRUCT_DIR, f"pdb_0000{pdb_lower}.cif")
    if os.path.exists(cif_out) and os.path.getsize(cif_out) > 1000:
        print(f"[PDB] Already exists: {cif_out}")
        return
    client = http_client.HttpClient("https://files.rcsb.org", qps=2.0)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    print(f"[PDB] Downloading {url} -> {cif_out}")
    resp = client.fetch(url)
    raw = resp.data
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    with open(cif_out, "wb") as fh:
        fh.write(raw)


def fetch_alphafold_cif(acc: str):
    cif_out = os.path.join(STRUCT_DIR, f"AF-{acc}.cif")
    if os.path.exists(cif_out) and os.path.getsize(cif_out) > 1000:
        print(f"[AlphaFold] Already exists: {cif_out}")
        return
    try:
        req = urllib.request.Request(
            f"https://alphafold.ebi.ac.uk/api/prediction/{acc}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        if not meta:
            return
        cif_url = meta[0].get("cifUrl")
        if not cif_url:
            return
        print(f"[AlphaFold] Downloading {cif_url} -> {cif_out}")
        req2 = urllib.request.Request(cif_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=30) as r2:
            with open(cif_out, "wb") as fh:
                fh.write(r2.read())
    except Exception as exc:
        print(f"[AlphaFold] Skipped {acc}: {exc}")


def main():
    # ------------------------------------------------------------------
    # SYSTEM 2: Deep-Sea & Cavefish Opsins + Reference Bovine Rhodopsin
    # ------------------------------------------------------------------
    opsin_queries = [
        ("ref_bovine_rh1.tsv", "accession:P02699"),
        ("teleost_reviewed_opsins.tsv", '(taxonomy_name:Actinopterygii) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho) AND (reviewed:true)'),
        ("deepsea_myctophiformes.tsv", '(taxonomy_name:Myctophiformes) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho)'),
        ("deepsea_stomiiformes.tsv", '(taxonomy_name:Stomiiformes) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho)'),
        ("deepsea_beryciformes_diretmus.tsv", '(taxonomy_name:Diretmus OR taxonomy_name:Anoplogaster OR taxonomy_name:Hoplostethus OR taxonomy_name:Beryx) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho)'),
        ("deepsea_gadiformes_anguilliformes.tsv", '(taxonomy_name:Coryphaenoides OR taxonomy_name:Antimora OR taxonomy_name:Bathygadus OR taxonomy_name:Macrourus OR taxonomy_name:Anguilla OR taxonomy_name:Conger OR taxonomy_name:Synaphobranchus OR taxonomy_name:Scopelarchus) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho)'),
        ("cavefish_astyanax_sinocyclocheilus.tsv", '(taxonomy_name:Astyanax OR taxonomy_name:Sinocyclocheilus OR taxonomy_name:Lucifuga OR taxonomy_name:Amblyopsis OR taxonomy_name:Typhlichthys OR taxonomy_name:Phreatichthys) AND (protein_name:rhodopsin OR protein_name:opsin OR gene:rh1 OR gene:rho OR gene:opn1lw OR gene:opn1sw OR gene:rh2)'),
        ("shallow_diurnal_controls.tsv", '(taxonomy_name:"Danio rerio" OR taxonomy_name:"Oryzias latipes" OR taxonomy_name:"Gasterosteus aculeatus" OR taxonomy_name:"Oreochromis niloticus" OR taxonomy_name:"Carassius auratus" OR taxonomy_name:"Takifugu rubripes" OR taxonomy_name:"Poecilia reticulata") AND (protein_name:rhodopsin OR gene:rh1 OR gene:rho)'),
    ]

    for fname, q in opsin_queries:
        fetch_uniprot_tsv(q, os.path.join(OPSIN_DIR, fname))

    # ------------------------------------------------------------------
    # SYSTEM 3: Polar Fish Antifreeze Glycoproteins (AFGPs), AFPs & Progenitors
    # ------------------------------------------------------------------
    afgp_queries = [
        ("fish_all_antifreeze.tsv", '(taxonomy_name:Actinopterygii) AND (protein_name:antifreeze OR gene:afgp OR gene:afp)'),
        ("notothenioid_trypsinogen_progenitors.tsv", '(taxonomy_name:Notothenioidei) AND (protein_name:trypsinogen OR protein_name:trypsin OR protein_name:"serine protease" OR protein_name:lectin OR protein_name:"sialic acid synthase")'),
        ("gadiformes_trypsinogen_lectin_controls.tsv", '(taxonomy_name:Boreogadus OR taxonomy_name:Gadus OR taxonomy_name:Microgadus OR taxonomy_name:Eleginus) AND (protein_name:trypsinogen OR protein_name:trypsin OR protein_name:"C-type lectin" OR protein_name:"sialic acid synthase")'),
        ("fish_ctype_lectin_controls.tsv", '(taxonomy_name:Clupea OR taxonomy_name:Osmerus OR taxonomy_name:Hemitripterus OR taxonomy_name:Myoxocephalus OR taxonomy_name:Pseudopleuronectes OR taxonomy_name:"Danio rerio" OR taxonomy_name:"Salmo salar") AND (protein_name:"C-type lectin" OR protein_name:trypsinogen) AND (reviewed:true OR existence:1 OR existence:2)'),
    ]

    for fname, q in afgp_queries:
        fetch_uniprot_tsv(q, os.path.join(AFGP_DIR, fname))

    # ------------------------------------------------------------------
    # 3D STRUCTURES FROM RCSB PDB & ALPHAFOLD DB
    #   1U19: Bovine Rhodopsin bound to 11-cis-retinal (RET) at 2.2 A
    #   2PY2: Polar Fish Type II C-type Lectin Antifreeze Protein at 1.34 A
    #   1HG7: Polar Fish Type III Antifreeze Protein at 1.15 A
    #   1WFA: Winter Flounder Type I Alpha-Helical Antifreeze Protein
    # ------------------------------------------------------------------
    for pdb_id in ["1U19", "2PY2", "1HG7", "1WFA"]:
        fetch_pdb_cif(pdb_id)

    for af_acc in ["P35359", "P02699", "O42294", "Q90243", "P20488"]:
        fetch_alphafold_cif(af_acc)

    print("\nAll downloads complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
