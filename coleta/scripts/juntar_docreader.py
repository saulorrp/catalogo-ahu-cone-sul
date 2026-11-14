"""Junta ao índice `corpus/resgate_docreader.tsv.gz` os quadros achados pela varredura do DocReader.

A varredura (`coleta/scripts/varrer_docreader.py`, rodada com IP brasileiro) lê o rótulo de cada
página do visualizador ("Cx. 69\\Doc. 6420") e grava, por coleção, a página em que cada documento
começa: `coleta/docreader/resgate_pagfis_<bib>.tsv.gz` (bib, caixa, doc, pagfis). Aqui cada linha
vira uma referência CRAV (`PT/AHU/CU/017/0069/06420`). Quando um documento aparece em mais de um
trecho, vale a primeira página. As referências que o índice do gov.br já tem ficam como estão.

Casos especiais:
  - `089_UL_AV`: o HTML da BN traz o cabeçalho errado `AHU_ACL_CU_089`, mas as imagens pertencem à
    série `035` (Ultramar/Geral), enquanto `089_CU` é de fato a Secretaria do Conselho Ultramarino
    (`089`).
  - `011_MG`: os rótulos do DocReader em Minas Gerais numeram `Doc.` reiniciando dentro de cada
    caixa (`cx. N, doc. M`, como no `old_code`), enquanto `reference_code` usa a numeração contínua
    do catálogo novo. O mapeamento `(caixa, doc) -> reference_code` é resolvido pelo `old_code` do
    corpus em `corpus/colecoes/*/documentos.parquet`.
  - `022_SE`: os rótulos do DocReader (`022_SE`) usam 6 caixas e sequência contínua (`1..495`) que
    descola das 9 caixas locais do catálogo (`Cx. 1..9`, com `3-A`/`3-B`). O mapeamento combina
    `document_id_and_date`, `old_code` e os pares `(caixa, doc_local, sufixo)` extraídos da página
    de Sergipe no gov.br (`coleta/docreader/sergipe_govbr_pagfis.tsv.gz`).

Uso: python coleta/scripts/juntar_docreader.py
"""
from __future__ import annotations

import csv
import gzip
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from build import cotas  # noqa: E402

INDICE = RAIZ / "corpus" / "resgate_docreader.tsv.gz"
VARREDURA = RAIZ / "coleta" / "docreader"
COLECOES = RAIZ / "corpus" / "colecoes"

# Série do Conselho Ultramarino (CRAV) de cada rolo do DocReader.
SERIE = {
    "003_BG_AV": "003",
    "004_AL": "004",
    "005_BA_AV": "005",
    "005_BA_CA": "005-001",
    "005_BA_LF": "005-002",
    "006_CE": "006",
    "007_ES": "007",
    "008_GO": "008",
    "009_MA": "009",
    "010_MT": "010",
    "011_MG": "011",
    "012_CSRP": "012",
    "013_PA": "013",
    "014_PB": "014",
    "015_PE": "015",
    "016_PI": "016",
    "017_RJ_AV": "017",
    "017-1_RJ": "017-001",
    "018_RN": "018",
    "019_RS": "019",
    "020_AM_RN": "020",
    "021_SC": "021",
    "022_SE": "022",
    "023_SP_AV": "023",
    "023-1_SP_MG": "023-001",
    "030_SP_AV": "030",
    "050_CS_AV": "050",
    "084_OU_AV": "084",
    "089_CU": "089",
    "089_UL_AV": "035",
}

RE_DOC_INICIAL = re.compile(r"\s*(\d+)")
RE_OLD_MG = re.compile(r"cx\.\s*(\d+)\s*,\s*doc\.\s*(\d+)", re.IGNORECASE)
RE_OLD_SE = re.compile(r"cx\s*\.\s*(\d+)\s*,?\s*doc\s*\.?\s*(\d+)\s*(?:-\s*([A-Z]))?", re.I)
RE_DID_NUM = re.compile(r"^\s*(\d+)\.")
RE_NEW_SUF = re.compile(r"D\.\s*\d+\s*([A-Z])")


def carregar_mapa_mg() -> dict[tuple[int, int], str]:
    """Constrói `(caixa, doc_por_caixa) -> PT/AHU/CU/011/...` via `old_code` do corpus."""
    candidatos: dict[tuple[int, int], list[str]] = defaultdict(list)
    for arq in sorted(COLECOES.glob("*/documentos.parquet")):
        tabela = pq.read_table(arq, columns=["reference_code", "old_code", "new_code"])
        for r, o, n in zip(tabela["reference_code"], tabela["old_code"], tabela["new_code"]):
            ref = cotas.referencia(cotas.limpar_new_code(n.as_py()), r.as_py())
            if not ref or not ref.startswith("PT/AHU/CU/011/"):
                continue
            m = RE_OLD_MG.search(o.as_py() or "")
            if m:
                candidatos[(int(m.group(1)), int(m.group(2)))].append(ref)

    mapa: dict[tuple[int, int], str] = {}
    for (cx, dc), refs in candidatos.items():
        if len(refs) == 1:
            mapa[(cx, dc)] = refs[0]
        else:
            mesma_cx = [ref for ref in refs if int(ref.split("/")[4]) == cx]
            mapa[(cx, dc)] = mesma_cx[0] if mesma_cx else refs[0]
    return mapa


def carregar_quadros_se() -> dict[str, int]:
    """Resolve `PT/AHU/CU/022/... -> pagfis` combinando `022_SE.tsv.gz` e links do gov.br."""
    tsv_se = VARREDURA / "resgate_pagfis_022_SE.tsv.gz"
    gov_se = VARREDURA / "sergipe_govbr_pagfis.tsv.gz"
    seq_para_pag: dict[int, int] = {}
    if tsv_se.exists():
        with gzip.open(tsv_se, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                m_doc = RE_DOC_INICIAL.match(r["doc"])
                if m_doc and r["pagfis"].isdigit():
                    dc, pg = int(m_doc.group(1)), int(r["pagfis"])
                    if dc not in seq_para_pag or pg < seq_para_pag[dc]:
                        seq_para_pag[dc] = pg

    cx_doc_para_pag: dict[tuple[int, int, str], int] = {}
    if gov_se.exists():
        with gzip.open(gov_se, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                k = (int(r["caixa"]), int(r["doc_local"]), r["sufixo"])
                pg = int(r["pagfis"])
                if k not in cx_doc_para_pag or pg < cx_doc_para_pag[k]:
                    cx_doc_para_pag[k] = pg

    mapa_ref: dict[str, int] = {}
    for arq in sorted(COLECOES.glob("*/documentos.parquet")):
        cols = ["reference_code", "old_code", "new_code", "document_id_and_date"]
        tabela = pq.read_table(arq, columns=cols)
        for r, o, n, d in zip(
            tabela["reference_code"],
            tabela["old_code"],
            tabela["new_code"],
            tabela["document_id_and_date"],
        ):
            novo = cotas.limpar_new_code(n.as_py())
            ref = cotas.referencia(novo, r.as_py())
            if not ref or not ref.startswith("PT/AHU/CU/022/"):
                continue
            pg = None
            m_did = RE_DID_NUM.match(d.as_py() or "")
            if m_did and int(m_did.group(1)) in seq_para_pag:
                pg = seq_para_pag[int(m_did.group(1))]
            elif o.as_py() and RE_OLD_SE.search(o.as_py()):
                m_old = RE_OLD_SE.search(o.as_py())
                pg = cx_doc_para_pag.get(
                    (int(m_old.group(1)), int(m_old.group(2)), (m_old.group(3) or "").upper())
                )
            else:
                partes = ref.split("/")
                m_suf = RE_NEW_SUF.search(novo or "")
                suf = m_suf.group(1) if m_suf else ""
                pg = cx_doc_para_pag.get((int(partes[4]), int(partes[5]), suf))
            if pg is not None and (ref not in mapa_ref or pg < mapa_ref[ref]):
                mapa_ref[ref] = pg
    return mapa_ref


def main():
    with gzip.open(INDICE, "rt", encoding="utf-8") as f:
        linhas = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    cabecalho, corpo = linhas[0], linhas[1:]
    existentes = {l[0] for l in corpo}
    mapa_mg = carregar_mapa_mg()
    novos: dict[str, tuple[str, int]] = {}

    for ref, pagfis in carregar_quadros_se().items():
        if ref not in existentes and (ref not in novos or pagfis < novos[ref][1]):
            novos[ref] = ("022_SE", pagfis)

    for arquivo in sorted(VARREDURA.glob("resgate_pagfis_*.tsv.gz")):
        with gzip.open(arquivo, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                bib = r["bib"]
                if bib == "022_SE":
                    continue
                serie = SERIE.get(bib)
                m_doc = RE_DOC_INICIAL.match(r["doc"])
                if not serie or not r["caixa"].isdigit() or not m_doc:
                    continue
                cx = int(r["caixa"])
                doc = int(m_doc.group(1))
                if bib == "011_MG":
                    ref = mapa_mg.get((cx, doc))
                    if not ref:
                        continue
                else:
                    ref = f"PT/AHU/CU/{serie}/{cx:04d}/{doc:05d}"
                if ref in existentes:
                    continue
                pagfis = int(r["pagfis"])
                if ref not in novos or pagfis < novos[ref][1]:
                    novos[ref] = (bib, pagfis)

    corpo += [[ref, bib, str(p)] for ref, (bib, p) in novos.items()]
    corpo.sort()
    texto = io.StringIO()
    for l in [cabecalho] + corpo:
        texto.write("\t".join(l) + "\n")
    with gzip.GzipFile(INDICE, "wb", mtime=0) as f:
        f.write(texto.getvalue().encode("utf-8"))
    print(f"{len(novos)} referências novas; índice com {len(corpo)}", file=sys.stderr)


if __name__ == "__main__":
    main()
