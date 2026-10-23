"""importar_colecao.py — importa uma coleção já exportada em parquet (ex.: ahu-brasil).

Entrada : documentos.parquet de uma coleção, com `doc_uid` já atribuído (a coleção ahu-brasil
          sai de `resgate/scripts/exportar_colecao.py`, que mantém os uids estáveis entre
          reimportações).
Saídas  : corpus/colecoes/<colecao>/documentos.parquet   (campos da base + os da coleção)
          corpus/colecoes/<colecao>/registro_ids.json    (de onde vieram os uids)
          corpus/colecoes/<colecao>/camada.json          (proveniência)
          corpus/manifest.json

Os documentos que já estão em outra coleção (coluna `ahu_sul_doc_uid` preenchida) ficam de fora:
a versão do ahu-sul, com as camadas de pessoas e sociolinguística, é a que vai para o app.

Uso: python build/importar_colecao.py --entrada CAMINHO [--colecao ahu-brasil] [--corpus corpus]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ahu import conversor  # noqa: E402
from build.importar_base import (  # noqa: E402
    CAMPOS_BASE,
    ErroImportacao,
    atualizar_manifesto,
    commit_atual,
    sha256_arquivo,
)

VERSAO_SCRIPT = "1.0.0"
# Campos próprios da coleção que o app pode mostrar: o link do documento no site e a citação.
CAMPOS_EXTRAS = ["caminho_resgate", "url_actd", "pagina_actd", "fonte_texto", "qc"]
COLUNA_JA_EM_OUTRA = "ahu_sul_doc_uid"
GRAFIAS = {"Espirito Santo": "Espírito Santo"}


def pasta_da_serie(new_code, pasta):
    """A pasta como o ahu-sul a escreve: o nome da série na cifra, sem a coleção depois da
    vírgula (`Bahia-CA` e `Bahia-LF` viram `Bahia`, como `Rio de Janeiro-CA` vira `Rio de
    Janeiro` no ahu-sul). Sem série reconhecida (Reino, Códices), a pasta fica como veio."""
    cota = conversor.reconhecer(new_code)
    serie = conversor.POR_CODIGO.get(cota.codigo_serie) if cota is not None else None
    if serie is not None:
        return serie.nome.split(",")[0]
    # Sem série: tira a marca da coleção e acerta a grafia, para não abrir uma pasta à parte.
    pasta = re.sub(r"(-CA|-LF|-MG| \(.*\)| \|.*)$", "", str(pasta)).strip()
    return GRAFIAS.get(pasta, pasta)


def montar(entrada: pd.DataFrame, colecao: str) -> pd.DataFrame:
    faltam = [c for c in ["doc_uid", *CAMPOS_BASE] if c not in entrada.columns]
    if faltam:
        raise ErroImportacao(f"A entrada não tem as colunas {', '.join(faltam)}.")
    docs = entrada
    if COLUNA_JA_EM_OUTRA in docs.columns:
        docs = docs[docs[COLUNA_JA_EM_OUTRA].isna()]
    if not docs["doc_uid"].is_unique:
        raise ErroImportacao("doc_uid repetido na entrada.")
    if not docs["doc_uid"].astype(str).str.startswith(f"{colecao}:").all():
        raise ErroImportacao(f"Todo doc_uid deve começar com '{colecao}:'.")
    colunas = ["doc_uid", *CAMPOS_BASE, *[c for c in CAMPOS_EXTRAS if c in docs.columns]]
    docs = docs[colunas].copy()
    docs.insert(1, "colecao", colecao)
    docs["folder"] = [pasta_da_serie(n, p) for n, p in zip(docs["new_code"], docs["folder"])]
    docs["ano_isolado"] = docs["ano_isolado"].astype("Int64")
    return docs.reset_index(drop=True)


def importar(entrada: Path, corpus: Path, colecao: str) -> pd.DataFrame:
    bruto = pd.read_parquet(entrada)
    docs = montar(bruto, colecao)
    pasta = corpus / "colecoes" / colecao
    pasta.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(pasta / "documentos.parquet", index=False)
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(pasta / "registro_ids.json", "w", encoding="utf-8") as f:
        json.dump({"origem": "doc_uid atribuído na exportação da coleção (estável entre "
                             "reimportações)", "documentos": len(docs),
                   "fora_por_estarem_em_outra_colecao": int(len(bruto) - len(docs))},
                  f, ensure_ascii=False, indent=2)
    camada = {
        "nome": f"base_{colecao}", "tipo": "base", "versao": "1", "colecao": colecao,
        "processo": {"script": "build/importar_colecao.py", "versao_script": VERSAO_SCRIPT,
                     "commit": commit_atual()},
        "entrada": {"arquivo": entrada.name, "sha256": sha256_arquivo(entrada)},
        "data": agora, "documentos": len(docs),
        "esquema": {c: str(t) for c, t in docs.dtypes.items()},
        "origem": "Arquivo Histórico Ultramarino (AHU): catálogos publicados no ACTD/IICT e no "
                  "Projeto Resgate 'Barão do Rio Branco' (Biblioteca Nacional). Resumos e cotas "
                  "pertencem a essas instituições e são creditados como tal.",
    }
    with open(pasta / "camada.json", "w", encoding="utf-8") as f:
        json.dump(camada, f, ensure_ascii=False, indent=2)
    atualizar_manifesto(corpus, colecao, agora)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--colecao", default="ahu-brasil")
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    a = ap.parse_args()
    try:
        docs = importar(Path(a.entrada), Path(a.corpus), a.colecao)
    except ErroImportacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"{a.colecao}: {len(docs)} documentos importados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
