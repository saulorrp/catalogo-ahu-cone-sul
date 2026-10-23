"""Merge all verbetes/*.jsonl into ahu_brasil_lite.jsonl and link to ahu-sul by normalized cota.

Usage: python juntar.py /mnt/project-files/resgate AHU_SUL_documentos.parquet
"""
import json, re, sys, collections, hashlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from parse_texto import norm_cota

R = Path(sys.argv[1])
sul = pd.read_parquet(sys.argv[2])
sul["k"] = sul.new_code.map(lambda c: norm_cota(c) if isinstance(c, str) else None)
por_k = sul.dropna(subset=["k"]).groupby("k").doc_uid.apply(list).to_dict()

todos, vistos = [], collections.Counter()
for f in sorted((R / "verbetes").glob("*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        r["arquivo_fonte"] = f.name
        base = r.get("chave_cota") or f"{f.stem}|{r.get('numero')}"
        vistos[base] += 1
        sufixo = f"#{vistos[base]}" if vistos[base] > 1 else ""
        r["doc_uid"] = "ahu-brasil:" + hashlib.sha1((base + sufixo).encode()).hexdigest()[:10]
        r["ahu_sul_doc_uid"] = por_k.get(r.get("chave_cota"), [None])[0]
        todos.append(r)

with open(R / "ahu_brasil_lite.jsonl", "w", encoding="utf-8") as f:
    for r in todos: f.write(json.dumps(r, ensure_ascii=False) + "\n")

ligados = {r["ahu_sul_doc_uid"] for r in todos if r["ahu_sul_doc_uid"]}
sul["serie"] = sul.k.str.split("|").str[0]
resumo = {"verbetes": len(todos), "com_cota": sum(1 for r in todos if r.get("chave_cota")),
          "por_arquivo": collections.Counter(r["arquivo_fonte"] for r in todos),
          "ahu_sul_total": len(sul), "ahu_sul_ligados": len(ligados),
          "ahu_sul_por_serie": {s: {"total": int(len(g)), "ligados": int(g.doc_uid.isin(ligados).sum()), "pasta": g.folder.mode().iat[0]}
                                for s, g in sul.groupby(sul.serie.fillna("sem-cota"))}}
(R / "relatorio_juncao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), "utf-8")
print(json.dumps({k: v for k, v in resumo.items() if k != "por_arquivo"}, ensure_ascii=False, indent=1))
nao = sul[~sul.doc_uid.isin(ligados)][["doc_uid", "folder", "new_code", "description"]]
nao.to_csv(R / "ahu_sul_sem_par.csv", index=False)
