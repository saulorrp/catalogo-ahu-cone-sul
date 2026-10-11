"""avaliar_busca.py — WP 4.1: Cálculo de métricas nDCG@10, Recall@20 e MRR (spec §5.6, A5).

Calcula a qualidade dos motores de busca (v2 de referência vs v3) por consulta e por categoria:
- nDCG@10: ganho cumulativo descontado normalizado no top 10 (relevâncias 0, 1, 2)
- Recall@20: cobertura de documentos relevantes (escore >= 1) no top 20
- MRR: reciprocal rank do primeiro documento relevante
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def dcg_at_k(relevancias: Sequence[float | int], k: int = 10) -> float:
    """Calcula DCG@k usando a formula canonica 2^rel - 1 / log2(i + 1)."""
    dcg = 0.0
    for i, rel in enumerate(relevancias[:k], start=1):
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    relevancias_obtidas: Sequence[float | int],
    todas_relevancias: Sequence[float | int],
    k: int = 10,
) -> float:
    """Calcula nDCG@k comparando o ranking obtido com a ordenacao ideal."""
    dcg = dcg_at_k(relevancias_obtidas, k=k)
    relevancias_ordenadas = sorted(todas_relevancias, reverse=True)
    idcg = dcg_at_k(relevancias_ordenadas, k=k)

    if idcg <= 0.0:
        return 1.0 if dcg <= 0.0 else 0.0
    return dcg / idcg


def recall_at_k(
    recuperados: Sequence[str],
    relevantes_alvo: set[str] | Sequence[str],
    k: int = 20,
) -> float:
    """Proporção de documentos relevantes recuperados no top k."""
    alvos = set(relevantes_alvo)
    if not alvos:
        return 1.0 if not recuperados else 0.0
    top_k = set(recuperados[:k])
    encontrados = len(top_k & alvos)
    return encontrados / len(alvos)


def mrr(
    recuperados: Sequence[str],
    relevantes_alvo: set[str] | Sequence[str],
) -> float:
    """Mean Reciprocal Rank do primeiro documento relevante no ranking."""
    alvos = set(relevantes_alvo)
    if not alvos:
        return 0.0
    for i, doc_id in enumerate(recuperados, start=1):
        if doc_id in alvos:
            return 1.0 / i
    return 0.0


def calcular_metricas_consulta(
    ranking: Sequence[str],
    julgamentos_consulta: Mapping[str, int],
) -> dict[str, float]:
    """Calcula nDCG@10, Recall@20 e MRR para uma única consulta avaliada."""
    relevancias_obtidas = [julgamentos_consulta.get(doc, 0) for doc in ranking]
    todas_relevancias = list(julgamentos_consulta.values())

    relevantes_positivos = {doc for doc, nota in julgamentos_consulta.items() if nota >= 1}

    return {
        "ndcg_10": ndcg_at_k(relevancias_obtidas, todas_relevancias, k=10),
        "recall_20": recall_at_k(ranking, relevantes_positivos, k=20),
        "mrr": mrr(ranking, relevantes_positivos),
    }


def agregar_metricas(
    resultados_consultas: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Agrega métricas por tipo de consulta e globalmente."""
    if not resultados_consultas:
        return {"global": {"ndcg_10": 0.0, "recall_20": 0.0, "mrr": 0.0, "n": 0}, "por_tipo": {}}

    por_tipo: dict[str, list[dict[str, float]]] = {}
    todos: list[dict[str, float]] = []

    for item in resultados_consultas:
        tipo = item.get("tipo", "indefinido")
        metr = {
            "ndcg_10": float(item["ndcg_10"]),
            "recall_20": float(item["recall_20"]),
            "mrr": float(item["mrr"]),
        }
        por_tipo.setdefault(tipo, []).append(metr)
        todos.append(metr)

    def _media(lista: list[dict[str, float]]) -> dict[str, float]:
        n = len(lista)
        if n == 0:
            return {"ndcg_10": 0.0, "recall_20": 0.0, "mrr": 0.0, "n": 0}
        return {
            "ndcg_10": sum(x["ndcg_10"] for x in lista) / n,
            "recall_20": sum(x["recall_20"] for x in lista) / n,
            "mrr": sum(x["mrr"] for x in lista) / n,
            "n": n,
        }

    return {
        "global": _media(todos),
        "por_tipo": {t: _media(vals) for t, vals in sorted(por_tipo.items())},
    }


def formatar_tabela_markdown(
    comparacao: Mapping[str, Mapping[str, Any]],
) -> str:
    """Gera tabela comparativa em Markdown conforme spec §5.6 / A5."""
    linhas = [
        "| Configuração | Categoria | nDCG@10 | Recall@20 | MRR | N |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config_nome, stats in comparacao.items():
        glob = stats["global"]
        linhas.append(
            f"| **{config_nome}** | **Global** | **{glob['ndcg_10']:.4f}** | "
            f"**{glob['recall_20']:.4f}** | **{glob['mrr']:.4f}** | {glob['n']} |"
        )
        for tipo, t_stats in stats.get("por_tipo", {}).items():
            linhas.append(
                f"| {config_nome} | {tipo.capitalize()} | {t_stats['ndcg_10']:.4f} | "
                f"{t_stats['recall_20']:.4f} | {t_stats['mrr']:.4f} | {t_stats['n']} |"
            )
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação da busca (WP 4.1)")
    parser.add_argument("--julgamentos", type=Path, default=Path("avaliacao/julgamentos.json"))
    parser.add_argument("--saida-tabela", type=Path, default=None)
    args = parser.parse_args()

    if not args.julgamentos.is_file():
        print(f"Arquivo de julgamentos {args.julgamentos} não encontrado.")
        return

    with open(args.julgamentos, "r", encoding="utf-8") as f:
        dados = json.load(f)

    print("Julgamentos carregados com sucesso.")
    # Exibe resumo das consultas disponíveis
    print(f"Total de consultas julgadas: {len(dados)}")


if __name__ == "__main__":
    main()
