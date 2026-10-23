"""Mede o remetente e o destinatário do Gemini contra os da ahu-sul, já anotados.

Compara, nos lotes de avaliação, o nome extraído com `sender_name` e `recipient_name` do
documentos.parquet, pela `chave_nome` do app (a mesma que une variantes de grafia). Conta como
acerto a igualdade das chaves ou uma conter a outra (ex.: "Gomes Freire" × "Gomes Freire de
Andrade"). Quando o nome veio vazio, compara o cargo.

Uso: python coleta/enriquecimento/avaliar.py PASTA_SAIDA [--app-dados app_dados]
"""
import argparse, collections, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ahu.texto import chave_nome  # noqa: E402


SEM_ANOTACAO = {"", "unknown", "desconhecido", "n/a", "none"}


def confere(extraido, anotado):
    """True, False, ou None quando não há anotação para comparar. "Unknown" na anotação e
    nenhum nome no Gemini contam como acordo: os dois dizem que o resumo não traz o campo."""
    if not isinstance(anotado, str):
        return None
    if anotado.strip().lower() in SEM_ANOTACAO:
        return not extraido
    if not extraido:
        return False
    a = chave_nome(anotado)
    for campo in ("nome", "cargo"):
        e = chave_nome(extraido.get(campo) or "")
        if e and a and (e == a or e in a or a in e):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("saida")
    ap.add_argument("--app-dados", default="app_dados")
    a = ap.parse_args()
    docs = pd.read_parquet(Path(a.app_dados) / "documentos.parquet",
                           columns=["doc_uid", "sender_name", "recipient_name"]).set_index("doc_uid")
    contas = collections.defaultdict(collections.Counter)
    exemplos = []
    for arq in sorted(Path(a.saida).glob("avaliacao_*.json")):
        lote = json.loads(arq.read_text(encoding="utf-8"))
        modelo = lote["modelo"]
        for r in lote["docs"]:
            if r["doc_uid"] not in docs.index:
                continue
            linha = docs.loc[r["doc_uid"]]
            for campo, anotado in (("rem", linha["sender_name"]), ("dest", linha["recipient_name"])):
                ok = confere(r[campo], anotado)
                if ok is None:
                    continue
                contas[(modelo, campo)]["acerto" if ok else "erro"] += 1
                if not ok and len(exemplos) < 30:
                    exemplos.append((campo, anotado, r[campo]))
    if not contas:
        sys.exit("Nenhum lote de avaliação em " + a.saida)
    print("| Modelo | Campo | Acertos | Total | % |\n|---|---|---|---|---|")
    for (modelo, campo), c in sorted(contas.items()):
        n = c["acerto"] + c["erro"]
        print(f"| {modelo} | {campo} | {c['acerto']} | {n} | {100 * c['acerto'] / n:.1f} |")
    print("\nDivergências (campo · anotado · Gemini):")
    for campo, anotado, extraido in exemplos:
        print(f"- {campo} · {anotado} · {extraido}")


if __name__ == "__main__":
    main()
