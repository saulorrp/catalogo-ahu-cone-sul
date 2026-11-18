"""Dá o link do Resgate às duplicatas do catálogo que ficaram sem ele (auditoria de 17/11/2026).

O catálogo tem pares de linhas com a mesma cota antiga e o mesmo verbete (número, data e local),
mas cotas novas diferentes: 010/0006/00434 e 010/0007/00434 (Mato Grosso), 018/0008/00450 e
016/0008/00450 (Piauí). O gov.br ligou só uma linha de cada par, e a BN rotula o documento com a
cota da outra (Cx. 7; série 016). As duas linhas descrevem o mesmo documento, então a que está
sem link recebe o mesmo link no índice.

Só entram pares em que as duas cotas têm o mesmo número de documento e as séries são 010 ou
016/018. Os pares da série 017 têm números diferentes e ficam como estão.

Uso: python coleta/scripts/ligar_duplicatas.py [--dry-run]; depois aplicar_indice_resgate.py.
"""
from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
INDICE = RAIZ / "corpus" / "resgate_docreader.tsv.gz"
DOCS_PARQUET = RAIZ / "app_dados" / "documentos.parquet"
SERIES = ({"010"}, {"016", "018"})


def pares(docs: pd.DataFrame, indice: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    novos: dict[str, tuple[str, str]] = {}
    docs = docs.dropna(subset=["reference_code", "old_code", "document_id_and_date"])
    for _, grupo in docs.groupby(["old_code", "document_id_and_date"]):
        refs = sorted(set(grupo["reference_code"]))
        if len(refs) != 2:
            continue
        partes = [r.split("/") for r in refs]
        if partes[0][5] != partes[1][5] or {p[3] for p in partes} not in SERIES:
            continue
        com = [r for r in refs if r in indice]
        sem = [r for r in refs if r not in indice]
        if len(com) == 1 and len(sem) == 1:
            novos[sem[0]] = indice[com[0]]
    return novos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    with gzip.open(INDICE, "rt", encoding="utf-8") as f:
        cabecalho = next(f).rstrip("\n")
        indice = {}
        for linha in f:
            ref, bib, pagfis = linha.rstrip("\n").split("\t")
            indice[ref] = (bib, pagfis)
    docs = pd.read_parquet(DOCS_PARQUET, columns=["reference_code", "old_code", "document_id_and_date"])
    novos = pares(docs, indice)
    for ref, (bib, pagfis) in sorted(novos.items()):
        print("novo", ref, bib, pagfis)
    print(len(novos), "links novos")
    if args.dry_run or not novos:
        return 0
    indice.update(novos)
    buf = io.StringIO()
    buf.write(cabecalho + "\n")
    for ref in sorted(indice):
        buf.write(f"{ref}\t{indice[ref][0]}\t{indice[ref][1]}\n")
    with gzip.GzipFile(INDICE, "wb", mtime=0) as f:
        f.write(buf.getvalue().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
