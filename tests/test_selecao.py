"""Testes do WP 3.1: a seleção que atravessa buscas (spec §8.1).

O que a §8.1 pede e estes testes cobram: a seleção guarda só `doc_uid`, não passa de 500, é
baixada e carregada como JSON, e um arquivo de outro corpus não vira uma seleção menor em
silêncio.
"""
import json

import pytest

from ahu import selecao as mod
from ahu.consulta import Consulta, executar
from ahu.dados import carregar_sem_cache
from ahu.filtros import Filtros
from ahu.selecao import ErroDeSelecao, Selecao
from tests.conftest import APP_DADOS, tem_artefatos


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


# ---------------------------------------------------------------------------
# Marcar e desmarcar
# ---------------------------------------------------------------------------

def test_marcar_e_desmarcar():
    s = Selecao()
    assert len(s) == 0 and "a" not in s
    assert s.alternar("a", True) == 0
    assert s.alternar("b", True) == 0
    assert len(s) == 2 and "a" in s and "b" in s
    s.alternar("a", False)
    assert len(s) == 1 and "a" not in s


def test_nao_duplica_e_mantem_a_ordem_de_chegada():
    s = Selecao()
    s.adicionar(["c", "a", "b"])
    s.adicionar(["a", "c"])
    assert s.uids == ["c", "a", "b"]


def test_limpar_esvazia():
    s = Selecao()
    s.adicionar(["a", "b"])
    s.limpar()
    assert len(s) == 0


def test_teto_de_500_nao_e_passado_e_o_excedente_e_relatado():
    s = Selecao()
    sobraram = s.adicionar([f"u{i}" for i in range(mod.LIMITE + 25)])
    assert len(s) == mod.LIMITE and sobraram == 25 and s.cheia
    assert s.adicionar("mais_um") == 1          # já cheia: o novo não entra
    assert len(s) == mod.LIMITE
    s.remover("u0")
    assert not s.cheia and s.adicionar("mais_um") == 0


def test_remover_o_que_nao_esta_nao_quebra():
    s = Selecao()
    s.adicionar("a")
    s.remover(["b", "c"])
    assert s.uids == ["a"]


# ---------------------------------------------------------------------------
# A seleção atravessa buscas (§8.1)
# ---------------------------------------------------------------------------

def test_a_selecao_atravessa_buscas(corpus_busca):
    """Marcar numa busca, buscar outra coisa, e a seleção continua inteira."""
    primeira = executar(corpus_busca, Consulta(texto="deserção", modo="exata"))
    s = Selecao()
    s.adicionar([r.doc_uid for r in primeira.resultados[:2]])
    guardados = list(s.uids)

    segunda = executar(corpus_busca, Consulta(texto="gado", modo="exata"))
    assert {r.doc_uid for r in segunda.resultados} != set(guardados)
    assert s.uids == guardados           # a busca nova não mexeu na seleção

    # E um filtro que não devolve nenhum dos dois também não mexe.
    executar(corpus_busca, Consulta(filtros=Filtros(tipologias=["MAPA"])))
    assert s.uids == guardados


# ---------------------------------------------------------------------------
# Ver e ordenar
# ---------------------------------------------------------------------------

def test_ordem_desconhecida_erra_alto(corpus_busca):
    s = Selecao()
    with pytest.raises(ValueError, match="Ordem desconhecida"):
        s.ordenar(corpus_busca, "aleatoria")


def test_ordenar_nao_muda_a_selecao(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:5]
    s = Selecao()
    s.adicionar(uids)
    for ordem in mod.ORDENS:
        assert sorted(s.ordenar(corpus_busca, ordem)) == sorted(uids)
    assert s.uids == uids                # a ordem de chegada é preservada


def test_ordenar_por_ano_poe_sem_data_no_fim(corpus_busca):
    import pandas as pd

    uids = corpus_busca.documentos["doc_uid"].tolist()[:5]
    s = Selecao()
    s.adicionar(uids)
    anos = corpus_busca.documentos["ano_isolado"].copy()
    try:
        corpus_busca.documentos.loc[corpus_busca.documentos.index[2], "ano_isolado"] = None
        sem_data = corpus_busca.documentos.iloc[2]["doc_uid"]
        for ordem in ("ano_asc", "ano_desc"):
            assert s.ordenar(corpus_busca, ordem)[-1] == sem_data, ordem
        crescente = [corpus_busca.documentos.iloc[corpus_busca.indice_de(u)]["ano_isolado"]
                     for u in s.ordenar(corpus_busca, "ano_asc")[:-1]]
        assert crescente == sorted(crescente)
        assert pd.isna(corpus_busca.documentos.iloc[2]["ano_isolado"])
    finally:
        corpus_busca.documentos["ano_isolado"] = anos


def test_tabela_mostra_cotas_e_nenhum_identificador(corpus_busca):
    """D10: a tabela da seleção traz cota, ano e tipologia — nunca o `doc_uid`."""
    uids = corpus_busca.documentos["doc_uid"].tolist()[:3]
    s = Selecao()
    s.adicionar(uids)
    tabela = s.tabela(corpus_busca)
    assert len(tabela) == 3
    assert tabela["Nº"].tolist() == [1, 2, 3]
    assert "Código de referência" in tabela.columns
    assert not any("uid" in str(c).lower() for c in tabela.columns)
    texto = "\n".join(tabela.astype(str).to_numpy().ravel().tolist())
    assert "scribe_mediation_likely" not in texto
    assert not any(u in texto for u in uids)


# ---------------------------------------------------------------------------
# Baixar e carregar (R4)
# ---------------------------------------------------------------------------

def test_ida_e_volta_pelo_json(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:4]
    s = Selecao()
    s.adicionar(uids)
    bruto = s.para_json()
    conteudo = json.loads(bruto)
    assert conteudo["schema"] == mod.ESQUEMA and conteudo["documentos"] == 4
    assert conteudo["gerado_em"]

    volta, relatorio = Selecao.de_json(bruto, corpus_busca)
    assert volta.uids == uids
    assert relatorio == {"carregados": 4, "desconhecidos": [], "excedentes": 0,
                         "gerado_em": conteudo["gerado_em"]}


def test_json_aceita_proveniencia_opcional():
    s = Selecao()
    s.adicionar("a")
    conteudo = json.loads(s.para_json({"doi": "10.1590/x"}))
    assert conteudo["proveniencia"] == {"doi": "10.1590/x"}


def test_uid_que_nao_existe_mais_e_descartado_e_relatado(corpus_busca):
    bom = corpus_busca.documentos["doc_uid"].iloc[0]
    bruto = json.dumps({"schema": mod.ESQUEMA, "doc_uids": [bom, "ahu-sul:naoexiste"]})
    volta, relatorio = Selecao.de_json(bruto, corpus_busca)
    assert volta.uids == [bom]
    assert relatorio["desconhecidos"] == ["ahu-sul:naoexiste"]
    assert relatorio["carregados"] == 1


def test_json_sem_corpus_nao_filtra():
    bruto = json.dumps({"schema": mod.ESQUEMA, "doc_uids": ["x", "y"]})
    volta, relatorio = Selecao.de_json(bruto)
    assert volta.uids == ["x", "y"] and relatorio["desconhecidos"] == []


def test_json_acima_do_teto_relata_o_excedente():
    bruto = json.dumps({"schema": mod.ESQUEMA,
                        "doc_uids": [f"u{i}" for i in range(mod.LIMITE + 3)]})
    volta, relatorio = Selecao.de_json(bruto)
    assert len(volta) == mod.LIMITE and relatorio["excedentes"] == 3


@pytest.mark.parametrize("bruto, erro", [
    (b"nao sou json", "não é um JSON válido"),
    (json.dumps([1, 2]), "deveria ser um objeto JSON"),
    (json.dumps({"schema": "selecao-ahu-sul/99", "doc_uids": []}), "Esquema de seleção"),
    (json.dumps({"schema": "selecao-ahu-sul/1"}), "lista de identificadores"),
    (json.dumps({"schema": "selecao-ahu-sul/1", "doc_uids": [1, 2]}),
     "lista de identificadores"),
])
def test_arquivo_invalido_erra_com_mensagem_clara(bruto, erro):
    with pytest.raises(ErroDeSelecao, match=erro):
        Selecao.de_json(bruto)


# ---------------------------------------------------------------------------
# Nos dados de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_selecao_grande_no_corpus_real(corpus_real):
    """Uma seleção no teto, nos dados de verdade, ordena e tabela sem reclamar."""
    s = Selecao()
    assert s.adicionar(corpus_real.documentos["doc_uid"].tolist()) > 0   # 7.051 > 500
    assert len(s) == mod.LIMITE
    for ordem in mod.ORDENS:
        assert len(s.ordenar(corpus_real, ordem)) == mod.LIMITE
    assert len(s.tabela(corpus_real, "cota")) == mod.LIMITE
    volta, relatorio = Selecao.de_json(s.para_json(), corpus_real)
    assert volta.uids == s.uids and not relatorio["desconhecidos"]


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_tem_a_caixa_de_selecionar_em_cada_resultado():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DADOS.parent / "app.py"), default_timeout=180)
    at.query_params["v"] = "2"
    at.query_params["q"] = "AHU_ACL_CU_017, Cx. 5, D. 487"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
    assert len(caixas) == 1, [c.key for c in at.checkbox]

    # Marcar a caixa põe o documento na seleção, e ela sobrevive a uma busca nova.
    caixas[0].check().run()
    assert not at.exception, [e.value for e in at.exception]
    selecao = at.session_state["selecao"]
    assert len(selecao) == 1
    marcado = selecao.uids[0]

    at.query_params["q"] = "gado"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["selecao"].uids == [marcado]
