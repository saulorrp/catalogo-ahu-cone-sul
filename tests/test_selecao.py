"""Testes da seleção (spec §8.1) e da sua velocidade (plano v3.1, R2).

O critério que o R2 pede por nome está em `test_marcar_nao_chama_o_motor_de_busca`: marcar uma
caixa **não** chama o motor. Antes do R2 cada clique custava duas buscas completas — a reexecução
do widget mais um `st.rerun()` explícito.

O resto cobre o que a §8.1 pede (atravessar buscas, teto de 500) e o espelho na URL, que é como a
seleção sobrevive a um recarregamento agora que o JSON de baixar e carregar saiu (D7).
"""
import pytest

from ahu import selecao as mod
from ahu.consulta import Consulta, executar
from ahu.dados import carregar_sem_cache
from ahu.filtros import Filtros
from ahu.selecao import Selecao
from tests.conftest import APP_DADOS, tem_artefatos

RAIZ = APP_DADOS.parent


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


def test_a_selecao_atravessa_buscas(corpus_busca):
    """§8.1: marcar numa busca, buscar outra coisa, e a seleção continua inteira."""
    primeira = executar(corpus_busca, Consulta(texto="deserção", modo="exata"))
    s = Selecao()
    s.adicionar([r.doc_uid for r in primeira.resultados[:2]])
    guardados = list(s.uids)

    segunda = executar(corpus_busca, Consulta(texto="gado", modo="exata"))
    assert {r.doc_uid for r in segunda.resultados} != set(guardados)
    assert s.uids == guardados

    executar(corpus_busca, Consulta(filtros=Filtros(tipologias=["MAPA"])))
    assert s.uids == guardados


# ---------------------------------------------------------------------------
# Ver: uma linha por documento, sem identificador (R2, D10)
# ---------------------------------------------------------------------------

def test_linhas_trazem_ano_cota_e_resumo(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:3]
    s = Selecao()
    s.adicionar(uids)
    linhas = s.linhas(corpus_busca)
    assert [linha["n"] for linha in linhas] == [1, 2, 3]
    assert [linha["uid"] for linha in linhas] == uids       # só para o ✕ saber a linha
    for linha in linhas:
        assert linha["resumo"]
        assert "UNFORMATTED_CODE" not in linha["cota"]
        assert not linha["cota"].endswith(".")


def test_a_ordem_das_linhas_e_a_de_chegada(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:4]
    s = Selecao()
    s.adicionar(list(reversed(uids)))
    assert [linha["uid"] for linha in s.linhas(corpus_busca)] == list(reversed(uids))


def test_tabela_nao_tem_identificador(corpus_busca):
    """D10: a tabela traz cota, ano e referência — nunca o `doc_uid`."""
    uids = corpus_busca.documentos["doc_uid"].tolist()[:3]
    s = Selecao()
    s.adicionar(uids)
    tabela = s.tabela(corpus_busca)
    assert list(tabela.columns) == ["Nº", "Ano", "Cota", "Código de referência"]
    texto = "\n".join(tabela.astype(str).to_numpy().ravel().tolist())
    assert not any(u in texto for u in uids)
    assert "scribe_mediation_likely" not in texto


def test_a_selecao_nao_ordena_mais_nem_vira_json():
    """R2: o painel é mínimo. D7: o JSON saiu."""
    assert not hasattr(Selecao, "ordenar")
    assert not hasattr(Selecao, "para_json") and not hasattr(Selecao, "de_json")
    assert not hasattr(mod, "ORDENS") and not hasattr(mod, "ESQUEMA")


# ---------------------------------------------------------------------------
# Espelho na URL (R2)
# ---------------------------------------------------------------------------

def test_o_prefixo_nao_colide_no_corpus(corpus_busca):
    tamanho = mod.tamanho_do_prefixo(corpus_busca)
    assert tamanho >= mod.MINIMO_PREFIXO
    prefixos = {mod._parte_variavel(u)[:tamanho]
                for u in corpus_busca.documentos["doc_uid"]}
    assert len(prefixos) == len(corpus_busca.documentos)


def test_ida_e_volta_pela_url(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:5]
    s = Selecao()
    s.adicionar(uids)
    texto = s.para_url(corpus_busca)
    assert texto and mod.SEPARADOR_URL in texto
    # Nenhum `doc_uid` inteiro vai para a URL: só o prefixo.
    assert not any(u in texto for u in uids)

    volta, perdidos = Selecao.de_url(texto, corpus_busca)
    assert volta.uids == uids and perdidos == 0


def test_selecao_vazia_da_url_vazia(corpus_busca):
    assert Selecao().para_url(corpus_busca) == ""
    volta, perdidos = Selecao.de_url("", corpus_busca)
    assert len(volta) == 0 and perdidos == 0
    volta, perdidos = Selecao.de_url(None, corpus_busca)
    assert len(volta) == 0 and perdidos == 0


def test_prefixo_desconhecido_e_contado_e_nao_quebra(corpus_busca):
    bom = corpus_busca.documentos["doc_uid"].iloc[0]
    tamanho = mod.tamanho_do_prefixo(corpus_busca)
    texto = mod.SEPARADOR_URL.join([mod._parte_variavel(bom)[:tamanho], "zzzzzzzzzz"[:tamanho]])
    volta, perdidos = Selecao.de_url(texto, corpus_busca)
    assert volta.uids == [bom] and perdidos == 1


def test_url_com_espacos_e_vazios_e_tolerada(corpus_busca):
    bom = corpus_busca.documentos["doc_uid"].iloc[0]
    tamanho = mod.tamanho_do_prefixo(corpus_busca)
    prefixo = mod._parte_variavel(bom)[:tamanho]
    volta, perdidos = Selecao.de_url(f" {prefixo} ,,", corpus_busca)
    assert volta.uids == [bom] and perdidos == 0


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_o_prefixo_no_corpus_real_nao_colide(corpus_real):
    tamanho = mod.tamanho_do_prefixo(corpus_real)
    prefixos = {mod._parte_variavel(u)[:tamanho] for u in corpus_real.documentos["doc_uid"]}
    assert len(prefixos) == len(corpus_real.documentos) == 7051
    assert tamanho == 7                      # medido: 7 caracteres bastam para 7.051 documentos


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_selecao_grande_desliga_o_espelho_em_vez_de_cortar(corpus_real):
    """R2: acima do teto da URL o espelho é desligado — nunca truncado em silêncio."""
    maximo = mod.maximo_na_url(corpus_real)
    assert 100 < maximo < mod.LIMITE          # vale a pena avisar, e não é o teto da seleção

    cabe = Selecao()
    cabe.adicionar(corpus_real.documentos["doc_uid"].tolist()[:maximo])
    texto = cabe.para_url(corpus_real)
    assert texto is not None and len(texto) <= mod.LIMITE_URL
    volta, _ = Selecao.de_url(texto, corpus_real)
    assert volta.uids == cabe.uids

    nao_cabe = Selecao()
    nao_cabe.adicionar(corpus_real.documentos["doc_uid"].tolist()[:maximo + 1])
    assert nao_cabe.para_url(corpus_real) is None


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_selecao_no_teto_da_selecao_ainda_monta_as_linhas(corpus_real):
    s = Selecao()
    assert s.adicionar(corpus_real.documentos["doc_uid"].tolist()) > 0   # 7.051 > 500
    assert len(s) == mod.LIMITE
    assert len(s.linhas(corpus_real)) == mod.LIMITE
    assert len(s.tabela(corpus_real)) == mod.LIMITE


# ---------------------------------------------------------------------------
# R2: velocidade — marcar não chama o motor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_marcar_nao_chama_o_motor_de_busca(monkeypatch):
    """O critério do R2: alternar a seleção **não** passa pelo motor de busca.

    Antes do R2 eram duas buscas por clique. O que garante o zero é a memoização da resposta na
    sessão (`resposta_da_consulta`), e por isso o teste vale mesmo onde o harness do Streamlit
    reexecuta a página inteira — como de fato acontece: o `AppTest` não respeita o escopo de
    `st.fragment`, e é por isso que o tempo de parede dele não serve para medir um clique.
    """
    from streamlit.testing.v1 import AppTest

    from ahu import busca

    chamadas = []
    original = busca.buscar

    def contado(*args, **kwargs):
        chamadas.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(busca, "buscar", contado)

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    at.query_params["v"] = "2"
    at.query_params["q"] = "deserção de soldados"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(chamadas) == 1, "a carga inicial faz uma busca"

    caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
    assert len(caixas) == 25

    for esperado in (1, 2, 3):
        chamadas.clear()
        caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
        caixas[esperado - 1].check().run()
        assert not at.exception, [e.value for e in at.exception]
        assert chamadas == [], f"marcar a caixa {esperado} chamou o motor {len(chamadas)}x"
        assert len(at.session_state["selecao"]) == esperado

    # E desmarcar também não.
    chamadas.clear()
    caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
    caixas[0].uncheck().run()
    assert chamadas == []
    assert len(at.session_state["selecao"]) == 2


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_paginar_nao_refaz_a_busca(monkeypatch):
    """A memoização ignora a página: paginar reexecuta a página, mas não o motor."""
    from streamlit.testing.v1 import AppTest

    from ahu import busca

    chamadas = []
    original = busca.buscar
    monkeypatch.setattr(busca, "buscar",
                        lambda *a, **k: (chamadas.append(1), original(*a, **k))[1])

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    at.query_params["v"] = "2"
    at.query_params["q"] = "deserção de soldados"
    at.query_params["m"] = "exata"
    at.run()
    assert len(chamadas) == 1

    paginas = [n for n in at.number_input if n.label == "Página"]
    assert paginas, "a busca de teste tem de ter mais de uma página"
    chamadas.clear()
    paginas[0].set_value(3).run()
    assert not at.exception, [e.value for e in at.exception]
    assert chamadas == []
    assert at.session_state["consulta"].pagina == 3

    # Mudar a consulta, isso sim, refaz a busca. A caixa está num `st.form`: é o "Buscar" que
    # envia, e é por isso que o teste aperta o botão em vez de só escrever.
    chamadas.clear()
    caixa = next(t for t in at.text_input if t.label == "Busca")
    caixa.set_value("gado")
    next(b for b in at.button if b.label == "Buscar").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["consulta"].texto == "gado"
    assert len(chamadas) == 1


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a_selecao_volta_de_um_recarregamento_pela_url():
    """R2: um recarregamento, um favorito ou um link copiado trazem a seleção de volta."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    at.query_params["v"] = "2"
    at.query_params["q"] = "deserção de soldados"
    at.query_params["m"] = "exata"
    at.run()
    caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
    caixas[0].check().run()
    caixas = [c for c in at.checkbox if c.key and c.key.startswith("sel_")]
    caixas[1].check().run()
    guardada = list(at.session_state["selecao"].uids)
    assert len(guardada) >= 1
    espelho = at.query_params[mod.CHAVE_URL]
    assert espelho

    # Uma sessão nova, com a mesma URL: a seleção volta.
    outra = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    outra.query_params["v"] = "2"
    outra.query_params["q"] = "deserção de soldados"
    outra.query_params["m"] = "exata"
    outra.query_params[mod.CHAVE_URL] = espelho
    outra.run()
    assert not outra.exception, [e.value for e in outra.exception]
    assert outra.session_state["selecao"].uids == guardada
