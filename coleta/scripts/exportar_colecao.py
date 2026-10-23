"""Export ahu_brasil_lite.jsonl as an app collection (same core columns as ahu-sul) to colecao_ahu_brasil/documentos.parquet.
Cota formats come from the app's converter (ahu/conversor.py in catalogo-ahu-cone-sul):
reference_code, the normalised Resgate cota, and the old cota of the equal-numbering series (017-01, 023-01).
Usage: python exportar_colecao.py /mnt/project-files/resgate APP_DIR"""
import json, re, sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/mnt/project-files/new classifier project/app v3")
from ahu.conversor import Cota, para_referencia, para_resgate, para_antiga

def cota(k):
    """chave_cota 'serie|cx|D' (serie '00501' = 005-01) as a converter Cota; None for códices and lettered boxes."""
    p = (k or "").split("|")
    if len(p) != 3 or not all(x.isdigit() for x in p): return None
    return Cota("resgate", int(p[1]), int(p[2]), int(p[0][:3]), int(p[0][3:]) if len(p[0]) > 3 else None)

QC = {}
if (R / "qc" / "qc_flags.jsonl").exists():
    for l in open(R / "qc" / "qc_flags.jsonl", encoding="utf-8"):
        q = json.loads(l); QC[q["doc_uid"]] = "; ".join(q["qc"])

# folder = captaincy/series name from the ACTD list, keyed by the series in chave_cota
NOMES = {}
for l in open(R / "actd" / "series_actd.tsv", encoding="utf-8"):
    cod, titulo = l.split("\t")[:2]
    m = re.fullmatch(r"CUc(\d{3})(?:s(\d\d))?", cod)
    NOMES[m.group(1) + (m.group(2) or "")] = titulo.split(" - actd")[0].replace("Colónia", "Colônia").replace("Montevideu", "Montevidéu").replace("Serviço De", "Serviço de")
NOMES.update({"022": "Sergipe d’el-Rei", "076": "Reino"})
def pasta(r):
    s = (r.get("chave_cota") or "").split("|")[0]
    if s.startswith("cod"): return "Códices"
    return NOMES.get(s) or re.sub(r"\s*[-–].*$", "", r.get("capitania") or "") or None

ACESSO = "acesso em 22/10/2026"
def fonte(r):
    """Citable origin of the record's text (the source whose description/cota were used)."""
    if r.get("url_actd"):
        return (f"Arquivo Histórico Ultramarino, catálogo da série {r.get('capitania')}, PDF publicado no ACTD/IICT, "
                f"{r['url_actd']}" + (f", p. {r['pagina']}" if r.get("pagina") else "") + f" ({ACESSO})")
    if (r.get("formato_fonte") or "").startswith("pdf"):
        return f"Biblioteca Nacional, Projeto Resgate, catálogo em PDF (gov.br), {r.get('caminho_resgate')} ({ACESSO})"
    return f"Biblioteca Nacional, Projeto Resgate, {r.get('titulo_pagina') or 'catálogo'} (gov.br), {r.get('caminho_resgate')} ({ACESSO})"

linhas = []
for l in open(R / "ahu_brasil_lite.jsonl", encoding="utf-8"):
    r = json.loads(l)
    c = cota(r.get("chave_cota"))
    ano = re.search(r"\b(1[4-8]\d\d)\b", r.get("data_local") or "")
    linhas.append({
        "doc_uid": r["doc_uid"], "colecao": "ahu-brasil", "folder": pasta(r),
        "description": r.get("description"), "old_code": r.get("old_code") or (c and para_antiga(c)), "new_code": r.get("new_code"),
        "reference_code": c and para_referencia(c), "cota_resgate": c and para_resgate(c),
        "document_id_and_date": " ".join(x for x in (f"{r['numero']}." if r.get("numero") else None, r.get("data_local")) if x) or None,
        "ano_isolado": int(ano.group(1)) if ano else None, "extracted_typology": r.get("tipologia_lite"),
        "sender_name": None, "recipient_name": None,
        "chave_cota": r.get("chave_cota"), "anexos": r.get("anexos"),
        "observacoes": " | ".join(r.get("observacoes") or []) or None,
        "caminho_resgate": r.get("caminho_resgate"), "url_actd": r.get("url_actd"), "pagina_actd": r.get("pagina") if r.get("url_actd") else None,
        "temas": "; ".join(r.get("temas") or []) or None, "fontes": "+".join(r.get("fontes") or []),
        "ahu_sul_doc_uid": r.get("ahu_sul_doc_uid"),
        "fonte_texto": fonte(r),
        "description_resgate": r.get("description_resgate"), "qc": QC.get(r["doc_uid"]),
    })
d = pd.DataFrame(linhas)
d["ano_isolado"] = d["ano_isolado"].astype("Int64")
d.to_parquet(R / "colecao_ahu_brasil" / "documentos.parquet", index=False)
print(len(d), "documentos;", d.reference_code.notna().sum(), "com reference_code;", d.ano_isolado.notna().sum(), "com ano")
print(d.folder.value_counts().head(45).to_string())
