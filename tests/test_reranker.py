"""Testes do cross-encoder / reranker (spec §1, ahu/reranker.py e busca.reordenar)."""
from __future__ import annotations

import numpy as np
import pytest

from ahu import busca, reranker
from ahu.dados import carregar_sem_cache
from tests.conftest import APP_DADOS, tem_artefatos

# ---------------------------------------------------------------------------
# Reranker falso para testes unitários (sem rede nem pesos reais)
# ---------------------------------------------------------------------------

def criar_reranker_falso(palavra_destaque: str = "porto"):
    """Cria um reranker falso que dá nota maior a textos contendo `palavra_destaque`.

    Também conta a quantidade de invocações recebidas em `chamadas`.
    """
    def _reranker(consulta: str, textos: list[str]) -> np.ndarray:
        _reranker.chamadas += 1
        alvo = palavra_destaque.lower()
        return np.array(
            [10.0 if alvo in t.lower() else 1.0 for t in textos],
            dtype=np.float32,
        )

    _reranker.chamadas = 0
    return _reranker


# ---------------------------------------------------------------------------
# (1) Ativação por variável de ambiente
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "valor,esperado",
    [
        ("1", True),
        ("sim", True),
        ("true", True),
        ("SIM", True),
        ("True", True),
        (" 1 ", True),
        (" true ", True),
        ("", False),
        ("   ", False),
        ("0", False),
        ("nao", False),
        ("false", False),
        ("outro", False),
    ],
)
def test_ligado_le_variavel_ambiente(monkeypatch, valor: str, esperado: bool):
    """ligado() lê AHU_RERANKER: '1', 'sim' e 'true' ligam; vazios ou outros desligam."""
    monkeypatch.setenv("AHU_RERANKER", valor)
    assert reranker.ligado() is esperado


def test_ligado_quando_variavel_ausente(monkeypatch):
    """Sem a variável AHU_RERANKER definida no ambiente, ligado() retorna False."""
    monkeypatch.delenv("AHU_RERANKER", raising=False)
    assert reranker.ligado() is False


# ---------------------------------------------------------------------------
# (2) Montagem do texto do documento
# ---------------------------------------------------------------------------

def test_texto_do_documento_com_descricao_e_data():
    """texto_do_documento junta description e 'Data: ...'."""
    doc = {
        "description": "Carta do governador sobre tropas e munições.",
        "document_id_and_date": "1750, Lisboa",
    }
    esperado = "Carta do governador sobre tropas e munições. Data: 1750, Lisboa"
    assert reranker.texto_do_documento(doc) == esperado


def test_texto_do_documento_tolera_campos_vazios_ou_none():
    """texto_do_documento tolera ausência de data, campos None ou strings vazias."""
    # Apenas descrição (data None ou vazia)
    assert (
        reranker.texto_do_documento({"description": "Resumo simples.", "document_id_and_date": None})
        == "Resumo simples."
    )
    assert (
        reranker.texto_do_documento({"description": "Resumo simples.", "document_id_and_date": ""})
        == "Resumo simples."
    )
    # Apenas data (descrição ausente, None ou vazia)
    assert (
        reranker.texto_do_documento({"description": None, "document_id_and_date": "1782"})
        == "Data: 1782"
    )
    assert (
        reranker.texto_do_documento({"description": "", "document_id_and_date": "1782"})
        == "Data: 1782"
    )
    # Ambos vazios ou None
    assert reranker.texto_do_documento({"description": None, "document_id_and_date": None}) == ""
    assert reranker.texto_do_documento({"description": "", "document_id_and_date": ""}) == ""
    assert reranker.texto_do_documento({}) == ""


# ---------------------------------------------------------------------------
# (3) Reordenação: limite, preservação da cauda, conjunto e posições
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (tem_artefatos() if callable(tem_artefatos) else tem_artefatos),
    reason="app_dados/ ainda não construído",
)
def test_reordenar_limite_cauda_preserva_conjunto_e_grava_posicoes():
    """reordenar reordena só os primeiros 'limite', mantém a cauda e grava posicoes['reranker']."""
    dados = carregar_sem_cache(APP_DADOS)
    limite = 3
    total_docs = min(8, len(dados.documentos))
    assert total_docs > limite, "Corpus precisa de documentos suficientes para o teste"

    uids = getattr(dados.lexico, "doc_uids", [f"doc_{i}" for i in range(total_docs)])
    ordem_original = [
        busca.Resultado(doc_uid=str(uids[i]), linha=i)
        for i in range(total_docs)
    ]

    # Escolhemos uma palavra do terceiro documento (índice 2) que não exista no primeiro (índice 0)
    textos_topo = [
        reranker.texto_do_documento(dados.documentos.iloc[r.linha])
        for r in ordem_original[:limite]
    ]
    palavras_2 = set(textos_topo[2].lower().split()) - set(textos_topo[0].lower().split())
    palavra = next((p for p in palavras_2 if len(p) > 3), "porto")

    falso = criar_reranker_falso(palavra)
    candidatos = [busca.Resultado(doc_uid=r.doc_uid, linha=r.linha) for r in ordem_original]
    reordenados = busca.reordenar(dados, "consulta de teste", candidatos, falso, limite=limite)

    # 1. Reranker foi chamado uma vez para o lote do topo
    assert falso.chamadas == 1

    # 2. Preserva o conjunto exato de documentos (tamanho, linhas e doc_uids)
    assert len(reordenados) == len(ordem_original)
    assert {r.linha for r in reordenados} == {r.linha for r in ordem_original}
    assert {r.doc_uid for r in reordenados} == {r.doc_uid for r in ordem_original}

    # 3. O topo contém exatamente os mesmos documentos do topo original
    assert {r.linha for r in reordenados[:limite]} == {r.linha for r in ordem_original[:limite]}

    # 4. A cauda mantém rigorosamente a ordem original
    assert [r.linha for r in reordenados[limite:]] == [r.linha for r in ordem_original[limite:]]

    # 5. Grava posicoes['reranker'] (1..limite) apenas nos itens do topo
    for p, r in enumerate(reordenados[:limite], start=1):
        assert r.posicoes.get("reranker") == p

    # 6. Nenhum item da cauda recebe marcação de reranker
    for r in reordenados[limite:]:
        assert "reranker" not in r.posicoes


# ---------------------------------------------------------------------------
# (4) Busca no modo 'exata' não chama o reranker
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (tem_artefatos() if callable(tem_artefatos) else tem_artefatos),
    reason="app_dados/ ainda não construído",
)
def test_buscar_modo_exata_nao_chama_reranker():
    """buscar com reranker no modo 'exata' não chama o reranker."""
    dados = carregar_sem_cache(APP_DADOS)
    falso = criar_reranker_falso("soldados")

    resposta = busca.buscar(dados, "deserção de soldados", modo="exata", reranker=falso)
    assert resposta.total > 0
    assert falso.chamadas == 0
    assert all("reranker" not in r.posicoes for r in resposta.resultados)


# ---------------------------------------------------------------------------
# (5) Busca no modo 'automatico' com encoder=None chama reranker e mantém total
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (tem_artefatos() if callable(tem_artefatos) else tem_artefatos),
    reason="app_dados/ ainda não construído",
)
def test_buscar_modo_automatico_com_reranker_chama_uma_vez_e_preserva_total():
    """buscar em 'automatico' com encoder=None chama o reranker uma vez e preserva o total."""
    dados = carregar_sem_cache(APP_DADOS)
    falso = criar_reranker_falso("presídio")

    sem_reranker = busca.buscar(
        dados, "deserção de soldados", modo="automatico", encoder=None, reranker=None
    )
    com_reranker = busca.buscar(
        dados, "deserção de soldados", modo="automatico", encoder=None, reranker=falso
    )

    assert falso.chamadas == 1
    assert com_reranker.total == sem_reranker.total > 0
    assert len(com_reranker.resultados) == len(sem_reranker.resultados)
    assert any("reranker" in r.posicoes for r in com_reranker.resultados)


# ---------------------------------------------------------------------------
# (6) Consulta com cota isolada não chama o reranker
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (tem_artefatos() if callable(tem_artefatos) else tem_artefatos),
    reason="app_dados/ ainda não construído",
)
def test_consulta_apenas_cota_nao_chama_reranker():
    """Consulta que é estritamente uma cota é resolvida via tabela e não chama o reranker."""
    dados = carregar_sem_cache(APP_DADOS)
    cota = "PT/AHU/CU/021/0006/00390"
    falso = criar_reranker_falso("porto")

    resposta = busca.buscar(
        dados, cota, modo="automatico", encoder=None, reranker=falso
    )
    assert resposta.cota == "cu|21|6|390"
    assert resposta.total >= 1
    assert falso.chamadas == 0
    assert all("reranker" not in r.posicoes for r in resposta.resultados)
