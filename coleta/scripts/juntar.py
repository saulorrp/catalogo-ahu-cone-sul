"""Merge all sources into ahu_brasil_lite.jsonl, one record per document entry, and link to ahu-sul by cota.

Sources, by priority for the same chave_cota:
  verbetes_actd/*.jsonl   ACTD/IICT text catalogues (AHU's own, complete, with old cota)  -> primary text
  verbetes/*.jsonl        Projeto Resgate gov.br pages and RJ PDFs                         -> adds caminho_resgate
  verbetes/tematico-*, verbetes/contratos-*   thematic selections                        -> only tag 'temas',
                                                                                            or added if the cota is new
Records with the same chave in one source stay separate (#2, #3 suffix, matched by order across sources).
doc_uid = sha1(chave[#n]) so it stays stable across rebuilds.

Usage: python juntar.py /mnt/project-files/resgate AHU_SUL_documentos.parquet
"""
import json, re, sys, collections, hashlib, difflib
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from parse_texto import norm_cota

R = Path(sys.argv[1])
sul = pd.read_parquet(sys.argv[2])
sul["k"] = sul.new_code.map(lambda c: norm_cota(c) if isinstance(c, str) else None)
por_k = sul.dropna(subset=["k"]).groupby("k").doc_uid.apply(list).to_dict()

def ler(arquivos):
    for f in arquivos:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); r["arquivo_fonte"] = f.name
            for c, v in r.items():  # stray control characters from the PDF text layer (Reino)
                if isinstance(v, str): r[c] = re.sub(r"[\x00-\x08\x0b-\x1f]", "", v).strip()
            k = norm_cota(r.get("new_code"))  # recompute with the current rules (e.g. 'D. 20.367')
            if k: r["chave_cota"] = k
            separa_data(r)
            yield r

TEM_ANO = re.compile(r"\b1[4-8]\d\d\b|s\.\s*d", re.I)
DATA_NA_DESC = re.compile(r"^(\[?\s*(?:ant\.|post\.|ca\.)?\s*1[4-8]\d\d\b[^\n]{0,80}?)\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{3,}\b)")
def separa_data(r):
    """Blocks without an entry number (RJ Castro e Almeida, Reino, some LF) leave the date at the start of the description."""
    if TEM_ANO.search(r.get("data_local") or ""): return
    m = DATA_NA_DESC.match(r.get("description") or "")
    if m:
        r["data_local"] = m.group(1).strip(" ,")
        r["description"] = r["description"][m.end():].strip()

selecao = lambda nome: nome.startswith(("tematico-", "contratos-"))
actd = sorted((R / "verbetes_actd").glob("*.jsonl"))
gov = sorted(f for f in (R / "verbetes").glob("*.jsonl") if not selecao(f.name))
sel = sorted(f for f in (R / "verbetes").glob("*.jsonl") if selecao(f.name))

docs, sem_chave = {}, []          # (chave, n) -> record
def chaveado(registros):
    vistos = collections.Counter()
    for r in registros:
        k = r.get("chave_cota")
        if not k: sem_chave.append(r); continue
        vistos[k] += 1
        yield (k, vistos[k]), r

for kn, r in chaveado(ler(actd)):
    r["fontes"] = ["actd"]; docs[kn] = r
for kn, r in chaveado(ler(gov)):
    d = docs.get(kn)
    if d is None:
        r["fontes"] = ["resgate"]; docs[kn] = r; continue
    d["fontes"].append("resgate")
    d["caminho_resgate"] = d.get("caminho_resgate") or r.get("caminho_resgate")
    d["arquivo_resgate"] = r["arquivo_fonte"]
    for campo in ("old_code", "description", "data_local", "anexos"):
        if not d.get(campo) and r.get(campo): d[campo] = r[campo]
    # the Resgate edition is sometimes a revised summary: keep it when it differs
    a, b = (re.sub(r"\W+", " ", (x or "").lower()) for x in (d.get("description"), r.get("description")))
    if b and a != b and difflib.SequenceMatcher(None, a[:600], b[:600]).ratio() < 0.9:
        d["description_resgate"] = r["description"]
novos_sel = 0
for r in ler(sel):
    k = r.get("chave_cota"); tema = r.get("tema") or r["arquivo_fonte"].rsplit(".", 1)[0]
    d = docs.get((k, 1)) if k else None
    if d is not None:
        d.setdefault("temas", [])
        if tema not in d["temas"]: d["temas"].append(tema)
    elif k:
        r["fontes"] = [r.get("fonte_tipo") or "selecao"]; r["temas"] = [tema]; docs[(k, 1)] = r; novos_sel += 1

todos = []
for (k, n), r in docs.items():
    r["doc_uid"] = "ahu-brasil:" + hashlib.sha1((k + (f"#{n}" if n > 1 else "")).encode()).hexdigest()[:10]
    r["ahu_sul_doc_uid"] = por_k.get(k, [None])[0] if n == 1 else None
    todos.append(r)
vistos = collections.Counter()
for r in sem_chave:
    if selecao(r["arquivo_fonte"]): continue
    base = f"{Path(r['arquivo_fonte']).stem}|{r.get('numero')}"
    vistos[base] += 1
    r["doc_uid"] = "ahu-brasil:" + hashlib.sha1((base + (f"#{vistos[base]}" if vistos[base] > 1 else "")).encode()).hexdigest()[:10]
    r["ahu_sul_doc_uid"] = None; r.setdefault("fontes", ["actd" if r.get("url_actd") else "resgate"])
    todos.append(r)

with open(R / "ahu_brasil_lite.jsonl", "w", encoding="utf-8") as f:
    for r in todos: f.write(json.dumps(r, ensure_ascii=False) + "\n")

ligados = {r["ahu_sul_doc_uid"] for r in todos if r["ahu_sul_doc_uid"]}
sul["serie"] = sul.k.str.split("|").str[0]
resumo = {"verbetes": len(todos), "com_cota": sum(1 for r in todos if r.get("chave_cota")),
          "por_fontes": collections.Counter("+".join(r["fontes"]) for r in todos),
          "novos_de_selecoes": novos_sel, "com_temas": sum(1 for r in todos if r.get("temas")),
          "por_serie": collections.Counter((r.get("chave_cota") or "sem-cota").split("|")[0] for r in todos),
          "ahu_sul_total": len(sul), "ahu_sul_ligados": len(ligados),
          "ahu_sul_por_serie": {s: {"total": int(len(g)), "ligados": int(g.doc_uid.isin(ligados).sum()), "pasta": g.folder.mode().iat[0]}
                                for s, g in sul.groupby(sul.serie.fillna("sem-cota"))}}
(R / "relatorio_juncao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), "utf-8")
print(json.dumps({k: v for k, v in resumo.items() if k != "por_serie"}, ensure_ascii=False, indent=1))
nao = sul[~sul.doc_uid.isin(ligados)][["doc_uid", "folder", "new_code", "description"]]
nao.to_csv(R / "ahu_sul_sem_par.csv", index=False)
