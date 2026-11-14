"""Junta ao índice `corpus/resgate_docreader.tsv.gz` os quadros achados pela varredura do DocReader.

A varredura (`coleta/scripts/varrer_docreader.py`, rodada com IP brasileiro) lê o rótulo de cada
página do visualizador ("Cx. 69\\Doc. 6420") e grava, por coleção, a página em que cada documento
começa: `coleta/docreader/resgate_pagfis_<bib>.tsv.gz` (bib, caixa, doc, pagfis). Aqui cada linha
vira uma referência CRAV (`PT/AHU/CU/017/0069/06420`). Quando um documento aparece em mais de um
trecho, vale a primeira página. As referências que o índice do gov.br já tem ficam como estão.

Uso: python coleta/scripts/juntar_docreader.py
"""
import csv
import gzip
import io
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
INDICE = RAIZ / "corpus" / "resgate_docreader.tsv.gz"
VARREDURA = RAIZ / "coleta" / "docreader"
# Série do Conselho Ultramarino de cada rolo do DocReader (lista oficial em docmulti.aspx?bib=resgate).
SERIE = {"003_BG_AV": "003", "004_AL": "004", "005_BA_AV": "005", "005_BA_CA": "005-001",
         "005_BA_LF": "005-002", "006_CE": "006", "007_ES": "007", "008_GO": "008", "009_MA": "009",
         "010_MT": "010", "011_MG": "011", "012_CSRP": "012", "013_PA": "013", "014_PB": "014",
         "015_PE": "015", "016_PI": "016", "017_RJ_AV": "017", "018_RN": "018", "019_RS": "019",
         "020_AM_RN": "020", "021_SC": "021", "022_SE": "022", "023_SP_AV": "023"}


def main():
    with gzip.open(INDICE, "rt", encoding="utf-8") as f:
        linhas = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    cabecalho, corpo = linhas[0], linhas[1:]
    existentes = {l[0] for l in corpo}
    novos = {}
    for arquivo in sorted(VARREDURA.glob("resgate_pagfis_*.tsv.gz")):
        with gzip.open(arquivo, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                serie = SERIE.get(r["bib"])
                doc = "".join(c for c in r["doc"].split("(")[0] if c.isdigit())
                if not serie or not r["caixa"].isdigit() or not doc:
                    continue
                ref = f"PT/AHU/CU/{serie}/{int(r['caixa']):04d}/{int(doc):05d}"
                if ref in existentes:
                    continue
                pagfis = int(r["pagfis"])
                if ref not in novos or pagfis < novos[ref][1]:
                    novos[ref] = (r["bib"], pagfis)
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
