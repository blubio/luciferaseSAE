"""Download real curated Swiss-Prot sequences for the 50 Superfamilies.

Fetches canonical reviewed Swiss-Prot sequences from the UniProt REST API
for both the neofunctionalized enzyme and its ancestral/paralog counterpart
across all 50 superfamilies.
"""

import os
import sys
import json
import urllib.request
import time

ROOT = r"c:\Users\pagli\Desktop\translator"
DATA_DIR = os.path.join(ROOT, "data", "superfamilies_50")
os.makedirs(DATA_DIR, exist_ok=True)

# Curated 50 Superfamilies with canonical Swiss-Prot Accessions
# Format: (Family_ID, Name, EC, Functional_Accessions, Paralog_Accessions)
SUPERFAMILY_REGISTRY = [
    # EC 1: Oxidoreductases
    ("SF01_ANL", "ANL Luciferase vs Acyl-CoA Synthetase", "1.13.12 / 6.2.1", ["P08659", "Q27758"], ["P0C027", "Q9V777"]),
    ("SF02_SDR", "Short-Chain Dehydrogenases (SDR)", "1.1.1", ["P00334", "P00336"], ["P16562", "P22414"]),
    ("SF03_CYP", "Cytochrome P450 Monooxygenases", "1.14.14", ["P00184", "P08684"], ["P05177", "P10635"]),
    ("SF04_ADH", "Alcohol Dehydrogenases", "1.1.1.1", ["P00325", "P00327"], ["P00330", "P00331"]),
    ("SF05_LDH_MDH", "Lactate vs Malate Dehydrogenase", "1.1.1.27 / 37", ["P00338", "P00344"], ["P00346", "P04406"]),
    ("SF06_FMO", "Flavin-Containing Monooxygenases", "1.14.13", ["P31513", "Q01740"], ["P36365", "P46200"]),
    ("SF07_SOD", "Superoxide Dismutase (Cu/Zn vs Mn)", "1.15.1.1", ["P00441", "P00442"], ["P04179", "P09671"]),
    ("SF08_PRX", "Peroxiredoxins (1-Cys vs 2-Cys)", "1.11.1.15", ["P32119", "P30041"], ["P30044", "P30048"]),
    ("SF09_GAPDH", "Glyceraldehyde-3-Phosphate DH", "1.2.1.12", ["P04406", "P00354"], ["P00355", "P00356"]),
    ("SF10_MAO", "Monoamine Oxidase (MAO-A vs MAO-B)", "1.4.3.4", ["P21397", "P21398"], ["P27338", "P14920"]),

    # EC 2: Transferases
    ("SF11_KINASE", "Protein Kinases (Tyr vs Ser/Thr)", "2.7.10 / 11", ["P00533", "P00519"], ["P17612", "P06241"]),
    ("SF12_AMINOTRANS", "Aminotransferases (Asp vs Ala)", "2.6.1", ["P17174", "P00505"], ["P24298", "P25409"]),
    ("SF13_GST", "Glutathione S-Transferases (Alpha vs Mu)", "2.5.1.18", ["P08263", "P09211"], ["P21266", "P04905"]),
    ("SF14_HEXOKINASE", "Hexokinase vs Glucokinase", "2.7.1.1", ["P19367", "P52789"], ["P35557", "P52792"]),
    ("SF15_METHYLTRANS", "SAM-Dependent Methyltransferases", "2.1.1", ["P14900", "P21796"], ["P23526", "P31382"]),
    ("SF16_UGT", "UDP-Glucuronosyltransferases", "2.4.1", ["P06133", "P19224"], ["P22309", "P35503"]),
    ("SF17_SULT", "Sulfotransferases (Cytosolic)", "2.8.2", ["P50225", "P50226"], ["P52839", "P50238"]),
    ("SF18_GNAT", "Gcn5-Related N-Acetyltransferases", "2.3.1", ["Q92830", "Q92831"], ["Q9H7E9", "Q9UBN7"]),
    ("SF19_PFK", "Phosphofructokinases", "2.7.1.11 / 90", ["P08237", "P17858"], ["P52784", "Q9WUB3"]),
    ("SF20_DNAPOL", "DNA Polymerase Family A", "2.7.7.7", ["P00582", "P19821"], ["P09884", "P20278"]),

    # EC 3: Hydrolases
    ("SF21_SER_PROT", "Serine Proteases (Trypsin vs Chymotrypsin)", "3.4.21", ["P00760", "P00761"], ["P00766", "P00772"]),
    ("SF22_BLACTAM", "Beta-Lactamases (Class A vs C)", "3.5.2.6", ["P00808", "P00810"], ["P05364", "P24735"]),
    ("SF23_LIPASE", "Esterases vs Lipases", "3.1.1.1 / 3", ["P07858", "P16301"], ["P04637", "P16233"]),
    ("SF24_DEHALO", "Haloalkane Dehalogenases", "3.8.1.5", ["P22643", "P59336"], ["Q9F818", "Q9XB87"]),
    ("SF25_ALKP", "Alkaline Phosphatases", "3.1.3.1", ["P05186", "P05187"], ["P09923", "P09467"]),
    ("SF26_AMYLASE", "Glycoside Hydrolase Family 13", "3.2.1.1", ["P04745", "P04746"], ["P00687", "P10665"]),
    ("SF27_CA", "Carbonic Anhydrases (CA1 vs CA2)", "4.2.1.1", ["P00915", "P00918"], ["P07451", "P22748"]),
    ("SF28_PDE", "Phosphodiesterases (PDE4 vs PDE5)", "3.1.4.17 / 35", ["P27815", "P54750"], ["O00408", "O76074"]),
    ("SF29_EPH", "Epoxide Hydrolases", "3.3.2.9 / 10", ["P07099", "P34913"], ["P34914", "Q9D379"]),
    ("SF30_SMALL_GTP", "Ras vs Rho/Rab GTPases", "3.6.5.2", ["P01112", "P01116"], ["P15154", "P61586"]),

    # EC 4: Lyases
    ("SF31_ALDOLASE", "Fructose-Bisphosphate Aldolase", "4.1.2.13", ["P04075", "P05062"], ["P09972", "P14540"]),
    ("SF32_AADC", "Aromatic L-Amino Acid Decarboxylase", "4.1.1.28", ["P20711", "P17770"], ["P27723", "P28329"]),
    ("SF33_ENOLASE", "Enolases (Alpha vs Gamma)", "4.2.1.11", ["P06733", "P09104"], ["P13929", "P15429"]),
    ("SF34_TERPENE", "Terpene Synthases", "4.2.3", ["Q38710", "Q43714"], ["Q948R7", "Q9M4J8"]),
    ("SF35_CITRATE", "Citrate Synthases", "2.3.3.1", ["P00889", "P00890"], ["P52651", "P52652"]),

    # EC 5: Isomerases
    ("SF36_TIM", "Triosephosphate Isomerase", "5.3.1.1", ["P60174", "P00940"], ["P00939", "P00942"]),
    ("SF37_PPIASE", "Peptidyl-Prolyl Isomerases (FKBP vs CyP)", "5.2.1.8", ["P62942", "P26885"], ["P05092", "P23284"]),
    ("SF38_TOPO", "DNA Topoisomerases (Type I vs II)", "5.6.2.1 / 2", ["P11387", "P11388"], ["P06786", "P15348"]),
    ("SF39_RACEMASE", "Amino Acid Racemases", "5.1.1.4 / 10", ["P29465", "P46706"], ["Q9I2G4", "Q9WYC1"]),
    ("SF40_CHORISMATE", "Chorismate Mutases", "5.4.99.5", ["P07023", "P07024"], ["P27603", "P35817"]),

    # EC 6: Ligases
    ("SF41_AARS", "Aminoacyl-tRNA Synthetases (Class I/II)", "6.1.1", ["P00959", "P00962"], ["P04803", "P07814"]),
    ("SF42_E3_LIGASE", "Ubiquitin-Protein Ligases", "6.3.2", ["P49755", "Q05086"], ["Q9Y4K3", "Q96PU5"]),
    ("SF43_DNALIG", "DNA Ligases (ATP vs NAD+)", "6.5.1.1 / 2", ["P18858", "P49916"], ["P15042", "P26282"]),
    ("SF44_BIOTINLIG", "Biotin Protein Ligases", "6.4.1.2 / 1", ["P00547", "P11498"], ["P05165", "P11499"]),
    ("SF45_GS", "Glutamine Synthetase", "6.3.1.2", ["P15104", "P04773"], ["P00964", "P05374"]),

    # Specialized Receptors, Sensory & Physiological Systems
    ("SF46_OPSIN", "Visual Rhodopsins (RH1 vs M/LWS)", "GPCR A", ["P02699", "P08255"], ["P04000", "P04001"]),
    ("SF47_NUCLEAR_REC", "Nuclear Hormone Receptors (ER vs PR)", "TF", ["P03372", "P06401"], ["P10275", "P10276"]),
    ("SF48_GLOBIN", "Hemoglobin vs Myoglobin", "Globin", ["P69905", "P68871"], ["P68082", "P02144"]),
    ("SF49_VGIC", "Voltage-Gated Sodium vs Potassium", "Channel", ["P35498", "P35499"], ["P15382", "P22459"]),
    ("SF50_AFP", "Antifreeze Proteins (Type I vs Type III)", "Adhesin", ["P09031", "P09032"], ["P19614", "P12411"])
]


def fetch_uniprot_fasta(accession):
    """Fetch FASTA sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Translator-Research-Pipeline)"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            lines = content.splitlines()
            header = lines[0] if lines else ""
            seq = "".join(l.strip() for l in lines[1:])
            return header, seq
    except Exception as e:
        print(f"    ⚠️ Warning: Failed to fetch {accession} ({e})")
        return None, None


def main():
    print("=" * 78)
    print("FETCHING CANONICAL SWISS-PROT SEQUENCES FOR 50 SUPERFAMILIES")
    print("=" * 78)
    
    total_families = len(SUPERFAMILY_REGISTRY)
    downloaded = 0
    catalog = {}

    for i, (fam_id, name, ec, func_accs, paralog_accs) in enumerate(SUPERFAMILY_REGISTRY, 1):
        print(f"[{i:02d}/{total_families}] Processing {fam_id}: {name} ({ec})...")
        fam_dir = os.path.join(DATA_DIR, fam_id)
        os.makedirs(fam_dir, exist_ok=True)
        
        fam_entry = {
            "id": fam_id,
            "name": name,
            "ec": ec,
            "functional_sequences": {},
            "paralog_sequences": {}
        }
        
        # 1. Fetch functional sequences
        for acc in func_accs:
            hdr, seq = fetch_uniprot_fasta(acc)
            if seq:
                fam_entry["functional_sequences"][acc] = {"header": hdr, "sequence": seq}
                downloaded += 1
                time.sleep(0.15) # Polite request throttling
                
        # 2. Fetch paralog sequences
        for acc in paralog_accs:
            hdr, seq = fetch_uniprot_fasta(acc)
            if seq:
                fam_entry["paralog_sequences"][acc] = {"header": hdr, "sequence": seq}
                downloaded += 1
                time.sleep(0.15)
                
        # Save local JSON record
        fam_json = os.path.join(fam_dir, "sequences.json")
        with open(fam_json, "w", encoding="utf-8") as fh:
            json.dump(fam_entry, fh, indent=2)
            
        catalog[fam_id] = {
            "name": name,
            "n_func": len(fam_entry["functional_sequences"]),
            "n_paralog": len(fam_entry["paralog_sequences"])
        }

    catalog_path = os.path.join(DATA_DIR, "superfamily_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2)
        
    print("\n" + "=" * 78)
    print(f"🎉 COMPLETED: Downloaded {downloaded} curated Swiss-Prot sequences across {total_families} superfamilies!")
    print(f"Catalog saved to: {catalog_path}")
    print("=" * 78)

if __name__ == "__main__":
    main()
