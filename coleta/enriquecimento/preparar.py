"""Monta a fila do enriquecimento: lotes de verbetes em entrada/lotes.jsonl.gz.

Todo o banco entra. Os primeiros lotes são uma amostra da ahu-sul, cujo remetente e destinatário
já foram anotados: servem para medir a qualidade (avaliar.py) logo no começo. Depois vem o resto da
ahu-sul e a ahu-brasil, na ordem do parquet.

Uso: python coleta/enriquecimento/preparar.py [--app-dados app_dados] [--lote 40] [--avaliacao 600]
"""
import argparse, gzip, json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dados", default="app_dados")
    ap.add_argument("--lote", type=int, default=40)
    ap.add_argument("--avaliacao", type=int, default=600, help="documentos da ahu-sul no começo")
    ap.add_argument("--semente", type=int, default=7)
    a = ap.parse_args()

    docs = pd.read_parquet(Path(a.app_dados) / "documentos.parquet",
                           columns=["doc_uid", "colecao", "description"])
    docs = docs[docs["description"].fillna("").str.strip().str.len() > 0]
    todos_sul = docs[docs["colecao"] == "ahu-sul"]
    sul = todos_sul.sample(a.avaliacao, random_state=a.semente)
    resto = pd.concat([todos_sul.drop(sul.index), docs[docs["colecao"] == "ahu-brasil"]])

    saida = AQUI / "entrada" / "lotes.jsonl.gz"
    saida.parent.mkdir(exist_ok=True)
    n = 0
    with gzip.open(saida, "wt", encoding="utf-8") as f:
        for nome, parte in (("avaliacao", sul), ("resto", resto)):
            linhas = parte[["doc_uid", "description"]].values.tolist()
            for i in range(0, len(linhas), a.lote):
                docs_lote = [{"doc_uid": u, "texto": " ".join(str(t).split())}
                             for u, t in linhas[i:i + a.lote]]
                f.write(json.dumps({"lote": f"{nome}_{i // a.lote:05d}", "docs": docs_lote},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} lotes ({len(sul)} documentos de avaliação, {len(resto)} no resto) em {saida}")


if __name__ == "__main__":
    main()
