"""Automated audit sweep for the translator repository.

Re-runnable implementation of the checks that found the round-4 and round-5 defects.
Usage: python scripts/audit_sweep.py      Writes: AUDIT_SWEEP.md at the repo root.

Checks
  A  unreachable code after an unconditional return/raise   (found the fake SAE steering)
  B  small .pt "caches" that may hold literals, not arrays  (found step1_records_cache)
  C  numbers in the manuscript that appear in no data file  (found the unbacked headline)
  D  figures with no script that writes them                (found fig11 provenance drift)
  E  tools named in Methods that appear in no code          (found CD-HIT, IQ-TREE, GROMACS)
  F  result tables with no generating script                (found the 3 corrected tables)
"""
import ast, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = "__ABANDONED_DO_NOT_USE__"
TOOLS = ["CD-HIT", "IQ-TREE", "FastML", "GROMACS", "gmx_MMPBSA", "ACPYPE", "GAFF2",
         "HMMER", "MAFFT", "AlphaFold", "OpenMM"]


def rd(p):
    return open(p, encoding="utf-8", errors="ignore").read()


def nb_src(p):
    try:
        cells = json.loads(rd(p)).get("cells", [])
    except Exception:
        return ""
    out = []
    for c in cells:
        if c.get("cell_type") == "code":
            out.append("\n".join(l for l in "".join(c.get("source", [])).split("\n")
                                 if not l.lstrip().startswith(("!", "%"))))
    return "\n".join(out)


def live_code():
    for p in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        if SKIP not in p:
            yield p, rd(p)
    for p in glob.glob(os.path.join(ROOT, "**", "*.ipynb"), recursive=True):
        if SKIP not in p:
            yield p, nb_src(p)


def check_a():
    out = []
    for p, src in live_code():
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for i, st in enumerate(fn.body[:-1]):
                if isinstance(st, (ast.Return, ast.Raise)):
                    tail = fn.body[i + 1:]
                    n = (tail[-1].end_lineno or tail[-1].lineno) - tail[0].lineno + 1
                    out.append(f"{os.path.relpath(p, ROOT)}::{fn.name} line {st.lineno} -> {n} unreachable lines")
                    break
    return out


def check_b(max_kb=200):
    out = []
    for p in glob.glob(os.path.join(ROOT, "results", "**", "*.pt"), recursive=True):
        kb = os.path.getsize(p) / 1e3
        if kb < max_kb:
            out.append(f"{os.path.relpath(p, ROOT)} is only {kb:.1f} kB - too small for embeddings; verify it holds computed values")
    return out


def check_c(ms_path):
    """Manuscript numbers with no source. Matches NUMERICALLY with rounding tolerance:
    a manuscript value v is backed if any number in the corpus rounds to v at v's own
    precision (so 0.5338 in a results file backs 0.534 in the text). Round-7 fix: the
    previous version compared strings and reported rounded values as orphans."""
    if not os.path.exists(ms_path):
        return ["manuscript not found"]
    import bisect
    corpus = []
    for p in glob.glob(os.path.join(ROOT, "**", "*"), recursive=True):
        if SKIP in p or not os.path.isfile(p):
            continue
        if os.path.splitext(p)[1].lower() in (".tsv", ".csv", ".py", ".json", ".txt", ".log"):
            corpus.append(rd(p))
    for p in glob.glob(os.path.join(ROOT, "**", "*.ipynb"), recursive=True):
        if SKIP not in p:
            corpus.append(nb_src(p))
    vals = sorted({float(x) for x in re.findall(r"(?<![\w.])\d+\.\d+", "\n".join(corpus))})
    ms = rd(ms_path)
    out = []
    for tok in sorted(set(re.findall(r"(?<![\w.])\d{1,4}\.\d{2,4}(?![\w])", ms))):
        v = float(tok)
        tol = 0.5 * 10 ** (-len(tok.split(".")[1]))      # half-ulp at the quoted precision
        i = bisect.bisect_left(vals, v - tol)
        if not (i < len(vals) and vals[i] <= v + tol):
            out.append(tok)
    return out


def check_d():
    code = "\n".join(s for _, s in live_code())
    out = []
    for p in glob.glob(os.path.join(ROOT, "results", "figures", "*.png")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in code and stem.split("_")[0] not in code:
            out.append(f"{os.path.relpath(p, ROOT)} - no live script references it")
    return out


def check_e(ms_path):
    if not os.path.exists(ms_path):
        return []
    ms, code = rd(ms_path), "\n".join(s for _, s in live_code())
    return [t for t in TOOLS if t.lower() in ms.lower() and t.lower() not in code.lower()]


def check_f():
    code = "\n".join(s for _, s in live_code())
    out = []
    for p in glob.glob(os.path.join(ROOT, "results", "*.tsv")) + glob.glob(os.path.join(ROOT, "results", "*.csv")):
        if os.path.basename(p).rsplit(".", 1)[0] not in code:
            out.append(f"{os.path.relpath(p, ROOT)} - no live script writes it")
    return out


def check_g():
    """Leakage and data integrity: genus disjointness, identity leakage, duplicates,
    honoured exclusions. Added round 6."""
    import collections
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import numpy as np
    import evaluate as EV
    out = []
    rows = EV.load_rows()
    pos_ids = open(os.path.join(ROOT, "results", "similarity", "pos_ids.txt")).read().split("\n")
    pid = np.load(os.path.join(ROOT, "results", "similarity", "pid_vs_pos.npy"))
    pos_col = {a: k for k, a in enumerate(pos_ids)}
    lab = np.array([r["label"] for r in rows])
    grp = np.array([r["split_group"] for r in rows])
    gen = np.array([r["genus"] for r in rows])
    seqs = [r["sequence"] for r in rows]

    tp = np.where((lab == "positive") & (grp == "firefly"))[0]
    tn = np.where((lab == "negative") & (grp == "nonlum_other") & ~np.isin(gen, list(EV.TEST_NEG_GENERA)))[0]
    te_p = np.where((lab == "positive") & np.isin(grp, ["click_beetle", "railroad_worm_group"]))[0]
    te_n = np.where((lab == "negative") & ((grp == "nonlum_elateroid")
                    | ((grp == "nonlum_other") & np.isin(gen, list(EV.TEST_NEG_GENERA)))))[0]

    ov = set(gen[np.r_[tp, tn]]) & set(gen[np.r_[te_p, te_n]])
    if ov:
        out.append(f"train/test GENUS OVERLAP: {sorted(ov)}")
    if set(np.r_[tp, tn].tolist()) & set(np.r_[te_p, te_n].tolist()):
        out.append("train/test INDEX OVERLAP")

    nn = pid[:, [pos_col[rows[i]["accession"]] for i in tp]].max(1)
    hi = int((nn[te_p] >= 70).sum())
    if hi:
        out.append(f"{hi} test positives >=70% identical to a training luciferase (max {nn[te_p].max():.1f}%)")

    seen = collections.defaultdict(list)
    for i, s_ in enumerate(seqs):
        seen[s_].append(i)
    dup = [v for v in seen.values() if len(v) > 1]
    if dup:
        out.append(f"{len(dup)} duplicate sequence groups inside the evaluation set")

    import csv as _csv
    with open(os.path.join(ROOT, "data", "curated", "dataset.tsv"), encoding="utf-8") as fh:
        raw = list(_csv.DictReader(fh, delimiter="\t"))
    kept = {r["accession"] for r in rows}
    bad = [r["accession"] for r in raw
           if (r["label"] == "excluded" or r["exact_duplicate_of"]) and r["accession"] in kept]
    if bad:
        out.append(f"{len(bad)} excluded/duplicate rows leaked into evaluation: {bad[:5]}")
    return out


def main():
    ms_path = os.path.join(ROOT, "manuscript", "manuscript_draft.md")
    blocks = [
        ("A. Unreachable code after return/raise", check_a()),
        ("B. Suspiciously small .pt caches", check_b()),
        ("C. Manuscript numbers with no source file", check_c(ms_path)),
        ("D. Figures with no generating script", check_d()),
        ("E. Tools named in Methods but absent from code", check_e(ms_path)),
        ("F. Result tables with no generating script", check_f()),
        ("G. Leakage and data integrity", check_g()),
    ]
    lines = ["# Automated audit sweep", ""]
    total = 0
    for title, hits in blocks:
        lines += [f"## {title}", ""]
        if hits:
            total += len(hits)
            lines += [f"- {h}" for h in hits]
        else:
            lines.append("- clean")
        lines.append("")
    lines.insert(1, f"\n**{total} findings.**\n")
    open(os.path.join(ROOT, "AUDIT_SWEEP.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
