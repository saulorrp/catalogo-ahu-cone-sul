"""Quality control of ahu_brasil_lite.jsonl: writes qc/qc_flags.jsonl (doc_uid -> flags) and qc/RELATORIO_QC.md.
Flags: sem_cota, sem_descricao, sem_data, desc_curta, desc_longa (likely merged entries), cota_repetida,
fora_de_sequencia (D far from neighbours in the same source), diverge_resgate (ACTD vs gov.br text differ)."""
import json, re, sys, collections, difflib
from pathlib import Path

R = Path(sys.argv[1]); (R / "qc").mkdir(exist_ok=True)
rs = [json.loads(l) for l in open(R / "ahu_brasil_lite.jsonl", encoding="utf-8")]
flags = collections.defaultdict(list)
rep = collections.Counter(r["chave_cota"] for r in rs if r.get("chave_cota"))

for r in rs:
    u, d = r["doc_uid"], r.get("description") or ""
    if not r.get("chave_cota"): flags[u].append("sem_cota")
    if not d.strip(): flags[u].append("sem_descricao")
    elif len(d) < 25: flags[u].append("desc_curta")
    elif len(d) > 3000: flags[u].append("desc_longa")
    if not re.search(r"\b1[4-8]\d\d\b|s\.\s*d", r.get("data_local") or "", re.I): flags[u].append("sem_data")
    if r.get("chave_cota") and rep[r["chave_cota"]] > 1: flags[u].append("cota_repetida")

# sequence: within each source file, in reading order, D should move smoothly
por_arq = collections.defaultdict(list)
for r in rs:
    p = (r.get("chave_cota") or "").split("|")
    if len(p) == 3 and p[2].isdigit(): por_arq[r["arquivo_fonte"]].append((int(p[2]), r["doc_uid"]))
for arq, seq in por_arq.items():
    for i, (dn, u) in enumerate(seq):
        viz = [x for x, _ in seq[max(0, i - 3):i] + seq[i + 1:i + 4]]
        if len(viz) >= 4:
            med = sorted(viz)[len(viz) // 2]
            if abs(dn - med) > max(200, med * 0.2): flags[u].append("fora_de_sequencia")

# ACTD vs Projeto Resgate text, where both exist (needs the gov.br record: re-read verbetes/)
gov = {}
for f in (R / "verbetes").glob("*.jsonl"):
    if f.name.startswith(("tematico-", "contratos-")): continue
    for l in open(f, encoding="utf-8"):
        g = json.loads(l)
        if g.get("chave_cota") and g.get("description"): gov.setdefault(g["chave_cota"], g["description"])
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
sims = []
for r in rs:
    if "actd" in r.get("fontes", []) and "resgate" in r.get("fontes", []) and r.get("description") and r["chave_cota"] in gov:
        a, b = norm(r["description"]), norm(gov[r["chave_cota"]])
        s = difflib.SequenceMatcher(None, a[:600], b[:600]).ratio()
        sims.append(s)
        if s < 0.6: flags[r["doc_uid"]].append("diverge_resgate")

with open(R / "qc" / "qc_flags.jsonl", "w", encoding="utf-8") as f:
    for u, fl in flags.items(): f.write(json.dumps({"doc_uid": u, "qc": fl}, ensure_ascii=False) + "\n")

cont = collections.Counter(x for fl in flags.values() for x in fl)
por_serie = collections.defaultdict(collections.Counter)
uid2 = {r["doc_uid"]: r for r in rs}
for u, fl in flags.items():
    s = (uid2[u].get("chave_cota") or "sem-cota").split("|")[0]
    for x in fl: por_serie[s][x] += 1
sims.sort()
md = [f"# Controle de qualidade do banco AHU-Brasil\n", f"Verbetes: {len(rs)}. Com alguma marca: {len(flags)} ({len(flags)/len(rs):.1%}).\n",
      "| Marca | Verbetes |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cont.most_common()]
if sims:
    md += ["", f"Texto ACTD × Projeto Resgate (mesma cota, {len(sims)} pares): similaridade mediana {sims[len(sims)//2]:.3f}; "
           f"{sum(s >= 0.9 for s in sims)/len(sims):.1%} ≥ 0,9; {sum(s < 0.6 for s in sims)} abaixo de 0,6."]
md += ["", "## Por série (só séries com marcas)", "", "| Série | " + " | ".join(k for k, _ in cont.most_common()) + " |",
       "|---|" + "---|" * len(cont)]
for s in sorted(por_serie): md.append(f"| {s} | " + " | ".join(str(por_serie[s][k]) for k, _ in cont.most_common()) + " |")
(R / "qc" / "RELATORIO_QC.md").write_text("\n".join(md) + "\n", "utf-8")
print("\n".join(md[:14]))
