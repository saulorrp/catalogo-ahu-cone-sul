"""listar_duplicatas.py — escreve corpus/duplicatas.ignore com as cópias de verbetes repetidos.

Repetido = mesma série, caixa, documento (e fim do intervalo) e mesmo resumo, sem acento, caixa e
pontuação, depois dos acertos de build/cotas.py. Fica a primeira ocorrência na ordem da base (o
ahu-sul vem antes do ahu-brasil); as outras vão para a lista. Nada é apagado: a construção do app
(build/construir_app.py) marca as cópias na coluna `duplicata_de`, a busca não as devolve, e elas
seguem nos dados publicados, por transparência. Para a busca voltar a mostrar um verbete, apague a
linha dele.

Uso: python scripts/listar_duplicatas.py [--corpus corpus]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from build import cotas  # noqa: E402
from build.construir_app import ler_base, ler_manifesto  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=RAIZ / "corpus")
    args = p.parse_args(argv)
    base = ler_base(args.corpus, ler_manifesto(args.corpus))
    indice = cotas.IndiceDigitArq.ler(args.corpus / "digitarq_referencias.tsv.gz")
    base, _ = cotas.ajustar(base, indice)
    repete = cotas.duplicatas(base)
    linhas = [
        "# Verbetes repetidos que a busca ignora; seguem nos dados (scripts/listar_duplicatas.py).",
        "# Formato: doc_uid  # repete <doc_uid mantido> | <cota>. Apague a linha para trazer de volta.",
    ]
    for i in repete[repete.notna()].index:
        linhas.append(f"{base.at[i, 'doc_uid']}  # repete {repete[i]} | {base.at[i, 'new_code']}")
    destino = args.corpus / "duplicatas.ignore"
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"{int(repete.notna().sum())} verbetes em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
