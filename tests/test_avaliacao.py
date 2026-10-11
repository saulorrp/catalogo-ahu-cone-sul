"""test_avaliacao.py — Testes do WP 4.1: Conjunto de teste, pool e métricas (spec §5.6, A5)."""
import math
from pathlib import Path

from avaliacao.avaliar_busca import (
    agregar_metricas,
    calcular_metricas_consulta,
    dcg_at_k,
    formatar_tabela_markdown,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from avaliacao.julgar import (
    carregar_consultas,
    carregar_julgamentos_existentes,
    salvar_julgamentos,
)

RAIZ = Path(__file__).resolve().parent.parent


def test_consultas_ouro_estrutura_e_contagens():
    consultas = carregar_consultas(RAIZ / "avaliacao" / "consultas_ouro.json")
    assert len(consultas) == 30, f"Esperado 30 consultas, obtido {len(consultas)}"

    tipos = [c["tipo"] for c in consultas]
    assert tipos.count("tema") == 12
    assert tipos.count("nome") == 6
    assert tipos.count("lugar") == 6
    assert tipos.count("cota") == 6

    ids = [c["id"] for c in consultas]
    assert len(set(ids)) == 30
    for c in consultas:
        assert c["consulta"].strip()


def test_metricas_dcg_e_ndcg():
    # Perfeito
    rels_perfeitas = [2, 2, 1, 1, 0]
    idcg = (
        (2**2 - 1) / math.log2(2)
        + (2**2 - 1) / math.log2(3)
        + (2**1 - 1) / math.log2(4)
        + (2**1 - 1) / math.log2(5)
    )
    assert math.isclose(dcg_at_k(rels_perfeitas, k=10), idcg, rel_tol=1e-5)
    assert math.isclose(ndcg_at_k(rels_perfeitas, rels_perfeitas, k=10), 1.0)

    # Invertido
    rels_invertidas = [0, 1, 1, 2, 2]
    ndcg_inv = ndcg_at_k(rels_invertidas, rels_perfeitas, k=10)
    assert 0.0 < ndcg_inv < 1.0

    # Vazio / tudo zero
    assert ndcg_at_k([], [], k=10) == 1.0
    assert ndcg_at_k([0, 0], [0, 0], k=10) == 1.0


def test_metricas_recall_e_mrr():
    recuperados = ["docA", "docB", "docC", "docD"]
    alvos = {"docB", "docD", "docZ"}

    # docB e docD estão nos recuperados (2 de 3)
    rec = recall_at_k(recuperados, alvos, k=20)
    assert math.isclose(rec, 2 / 3)

    # Primeiro relevante é docB na posição 2 -> MRR = 1/2 = 0.5
    assert math.isclose(mrr(recuperados, alvos), 0.5)

    # Nenhum recuperado
    assert mrr(["docX", "docY"], alvos) == 0.0


def test_calcular_metricas_consulta_e_agregacao():
    ranking = ["doc1", "doc2", "doc3", "doc4"]
    julgamentos = {
        "doc1": 2,
        "doc2": 0,
        "doc3": 1,
        "doc4": 0,
        "doc5": 1,  # relevante mas não recuperado no top 4
    }

    metr = calcular_metricas_consulta(ranking, julgamentos)
    assert "ndcg_10" in metr
    assert "recall_20" in metr
    assert "mrr" in metr
    assert metr["mrr"] == 1.0  # doc1 é relevante e está em 1º
    assert math.isclose(metr["recall_20"], 2 / 3)  # doc1 e doc3 recuperados de 3 relevantes

    consultas_stats = [
        {"tipo": "tema", **metr},
        {"tipo": "cota", "ndcg_10": 1.0, "recall_20": 1.0, "mrr": 1.0},
    ]
    agregado = agregar_metricas(consultas_stats)
    assert "global" in agregado
    assert "por_tipo" in agregado
    assert agregado["global"]["n"] == 2
    assert "tema" in agregado["por_tipo"]
    assert "cota" in agregado["por_tipo"]

    tab = formatar_tabela_markdown({"v3": agregado})
    assert "| Configuração | Categoria | nDCG@10 |" in tab
    assert "| **v3** | **Global** |" in tab


def test_persistencia_de_julgamentos(tmp_path):
    arquivo_teste = tmp_path / "julgamentos_teste.json"
    dados = {
        "Q01": {
            "consulta": "deserção",
            "tipo": "tema",
            "julgamentos": {"docA": {"nota": 2, "delta_seconds": 3.5}},
        }
    }
    salvar_julgamentos(dados, arquivo_teste)
    carregados = carregar_julgamentos_existentes(arquivo_teste)
    assert carregados == dados
