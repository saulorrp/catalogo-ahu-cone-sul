"""Leva o índice `corpus/resgate_docreader.tsv.gz` à coluna `resgate_docreader` do app.

Faz o que `build/construir_app.py` faz com essa coluna, sem refazer o resto do `app_dados` (que pede
os embeddings). Cada documento do `app_dados/documentos.parquet` recebe `<bib>#<pagfis>` do índice,
ou fica sem link se o índice não o tem. Depois o `manifest.json` é atualizado (contagem e SHA-256).
Criado na correção do rolo 012_CSRP (auditoria de 17/11/2026).

Uso: python coleta/scripts/aplicar_indice_resgate.py [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
INDICE = RAIZ / "corpus" / "resgate_docreader.tsv.gz"
DOCS_PARQUET = RAIZ / "app_dados" / "documentos.parquet"
MANIFEST_JSON = RAIZ / "app_dados" / "manifest.json"


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with gzip.open(INDICE, "rt", encoding="utf-8") as f:
        next(f)
        indice = {}
        for linha in f:
            ref, bib, pagfis = linha.rstrip("\n").split("\t")
            indice[ref] = f"{bib}#{pagfis}"

    tabela = pq.read_table(DOCS_PARQUET)
    refs = tabela["reference_code"].to_pylist()
    antes = tabela["resgate_docreader"].to_pylist()
    depois = [indice.get(r) if isinstance(r, str) else None for r in refs]

    mudancas = Counter()
    for ref, a, d in zip(refs, antes, depois):
        if a != d:
            tipo = "novo" if a is None else ("retirado" if d is None else "trocado")
            mudancas[(ref.split("/")[3] if isinstance(ref, str) else "?", tipo)] += 1
    relatorio = {
        "com_link_antes": sum(v is not None for v in antes),
        "com_link_depois": sum(v is not None for v in depois),
        "mudancas_por_serie": {f"{s} {t}": n for (s, t), n in sorted(mudancas.items())},
    }
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if args.dry_run or not mudancas:
        return 0

    idx = tabela.schema.get_field_index("resgate_docreader")
    tabela = tabela.set_column(idx, "resgate_docreader", pa.array(depois, type=pa.string()))
    pq.write_table(tabela, DOCS_PARQUET)

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest["gerado_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.setdefault("acertos", {})["resgate_com_quadro"] = relatorio["com_link_depois"]
    manifest["artefatos"]["documentos.parquet"] = sha256_arquivo(DOCS_PARQUET)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
