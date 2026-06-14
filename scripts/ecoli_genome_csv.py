#!/usr/bin/env python3
"""Emit examples/genome/ecoli_k12_genes.csv from v2ecoli's genes.tsv.

parsimony CSV format:
    # genome_length_bp=4641652
    old_locus_tag,locus_tag,start,end,strand,biotype
    <ecocyc_id>,<symbol>,<start>,<end>,<+|->,<biotype>
"""
import csv, os, sys

GENOME_LENGTH_BP = 4_641_652  # E. coli K-12 MG1655 (RefSeq NC_000913.3)

def find_genes_tsv() -> str:
    # genes.tsv ships with the reconstruction package installed in v2ecoli's venv.
    import importlib.util
    spec = importlib.util.find_spec("reconstruction.ecoli.flat")
    if spec and spec.submodule_search_locations:
        p = os.path.join(list(spec.submodule_search_locations)[0], "genes.tsv")
        if os.path.exists(p):
            return p
    raise SystemExit("genes.tsv not found; run inside the v2ecoli venv")

def main() -> None:
    src = find_genes_tsv()
    out = os.path.join(os.path.dirname(__file__), "..", "examples", "genome", "ecoli_k12_genes.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    with open(src, newline="") as f:
        # genes.tsv is tab-separated with a JSON-ish header; read tolerantly.
        reader = csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t")
        for r in reader:
            gid = (r.get("id") or "").strip().strip('"')
            try:
                start = int(str(r.get("left_end_pos", "")).strip().strip('"'))
                end = int(str(r.get("right_end_pos", "")).strip().strip('"'))
            except (TypeError, ValueError):
                continue
            direction = str(r.get("direction", "+")).strip().strip('"')
            strand = "+" if direction in ("+", "forward", "1") else "-"
            symbol = (r.get("symbol") or "").strip().strip('"')
            if gid and start and end:
                rows.append((gid, symbol, min(start, end), max(start, end), strand, "protein_coding"))
    rows.sort(key=lambda x: x[2])
    with open(out, "w", newline="") as f:
        f.write(f"# genome_length_bp={GENOME_LENGTH_BP}\n")
        w = csv.writer(f)
        w.writerow(["old_locus_tag", "locus_tag", "start", "end", "strand", "biotype"])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} genes")

if __name__ == "__main__":
    main()
