"""Testes do WP 3.4: o link da busca (spec §8.3, A6).

O A6 — "abrir um link de busca copiado reproduz a mesma lista, na mesma ordem" — é verificado com
**cinco buscas variadas**, de ponta a ponta: a `Consulta` vira URL, a URL volta a `Consulta`, e a
busca executada com a consulta que voltou devolve a mesma lista, na mesma ordem, na mesma página.
"""
from urllib.parse import parse_qs, urlsplit

import pytest

from ahu.consulta import Consulta, executar, para_query, url_da_busca
from ahu.dados import carregar_sem_cache
from ahu.filtros import Filtros
from tests.conftest import APP_DADOS, tem_artefatos


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


# As cinco buscas variadas do A6: texto simples, frase entre aspas, cota, lente com filtros do
# DeepSeek, e uma consulta vazia só com filtros — cada uma exercitando um pedaço diferente da URL.
CINCO_BUSCAS = (
    Consulta(texto="deserção de soldados", modo="automatico"),
    Consulta(texto='"obras do porto"', modo="exata", rigor="alto", ordenacao="ano_asc"),
    Consulta(texto="AHU_ACL_CU_017, Cx. 5, D. 487", modo="exata"),
    Consulta(texto="gado", modo="conceitual", rigor="baixo", lente="sintaxe_diacronica",
             filtros=Filtros(srsp_de=7, categorias=["Commoner"]), semente=42, pagina=2),
    Consulta(filtros=Filtros(tipologias=["CARTA"], ano_de=1770, ano_ate=1780, sem_data=False),
             ordenacao="ano_desc"),
)


def _de_url(url: str) -> Consulta:
    """Lê a consulta de um link, como o app faz com `st.query_params`."""
    return Consulta.de_url({k: v[0] for k, v in parse_qs(urlsplit(url).query).items()})


# ---------------------------------------------------------------------------
# A URL
# ---------------------------------------------------------------------------

def test_link_relativo_quando_nao_se_sabe_o_endereco():
    c = Consulta(texto="gado")
    assert url_da_busca(c) == para_query(c) == "?v=1&q=gado"


def test_link_absoluto_substitui_a_query_antiga():
    c = Consulta(texto="gado")
    url = url_da_busca(c, "https://catalogo-ahu-sul.streamlit.app/?v=1&q=outra&p=9")
    assert url == "https://catalogo-ahu-sul.streamlit.app/?v=1&q=gado"
    assert "outra" not in url and "p=9" not in url


def test_link_absoluto_em_localhost_preserva_o_caminho():
    url = url_da_busca(Consulta(texto="gado"), "http://localhost:8501/")
    assert url.startswith("http://localhost:8501/?v=1")


def test_consulta_padrao_leva_so_a_versao():
    """§8.3: um link de busca não carrega estado morto."""
    assert url_da_busca(Consulta()) == "?v=1"


@pytest.mark.parametrize("consulta", CINCO_BUSCAS, ids=lambda c: (c.texto or "só filtros")[:24])
def test_a6_a_consulta_volta_igual_do_link(consulta):
    """A ida e volta, campo por campo, nas cinco buscas."""
    volta = _de_url(url_da_busca(consulta, "https://exemplo.app/"))
    assert volta == consulta


@pytest.mark.parametrize("consulta", CINCO_BUSCAS, ids=lambda c: (c.texto or "só filtros")[:24])
def test_a6_o_link_reproduz_a_mesma_lista_na_mesma_ordem(corpus_busca, encoder, consulta):
    """A6 de ponta a ponta no corpus de mentira: mesma lista, mesma ordem, mesma página."""
    antes = executar(corpus_busca, consulta, encoder=encoder)
    depois = executar(corpus_busca, _de_url(url_da_busca(consulta)), encoder=encoder)

    assert depois.total == antes.total
    assert [r.doc_uid for r in depois.resultados] == [r.doc_uid for r in antes.resultados]
    assert [r.posicao for r in depois.resultados] == [r.posicao for r in antes.resultados]
    pagina = consulta.pagina
    assert ([r.doc_uid for r in depois.pagina(pagina, 25)]
            == [r.doc_uid for r in antes.pagina(pagina, 25)])


def test_a_semente_viaja_no_link():
    """§8.3: "abrir o link reproduz a mesma busca, a mesma página e a mesma semente"."""
    volta = _de_url(url_da_busca(Consulta(texto="x", semente=12345, pagina=4)))
    assert volta.semente == 12345 and volta.pagina == 4


def test_link_com_parametro_estranho_e_recusado():
    from ahu.consulta import ErroDeLink

    with pytest.raises(ErroDeLink, match="parâmetros desconhecidos"):
        _de_url("https://exemplo.app/?v=1&q=gado&bisbilhoteiro=1")


# ---------------------------------------------------------------------------
# Nos dados de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
@pytest.mark.parametrize("consulta", CINCO_BUSCAS, ids=lambda c: (c.texto or "só filtros")[:24])
def test_a6_nos_dados_reais_sem_carregar_o_modelo(corpus_real, consulta):
    """O A6 no corpus de verdade. Só o motor léxico, para não baixar o e5 no gate."""
    so_lexica = Consulta(**{**{f.name: getattr(consulta, f.name)
                               for f in consulta.__dataclass_fields__.values()},
                            "modo": "exata"})
    antes = executar(corpus_real, so_lexica)
    depois = executar(corpus_real, _de_url(url_da_busca(so_lexica, "https://exemplo.app/")))
    assert depois.total == antes.total
    assert [r.doc_uid for r in depois.resultados] == [r.doc_uid for r in antes.resultados]


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_mostra_o_link_e_ele_reproduz_a_busca():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DADOS.parent / "app.py"), default_timeout=180)
    at.query_params["v"] = "1"
    at.query_params["q"] = "AHU_ACL_CU_017, Cx. 5, D. 487"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    # O link aparece numa caixa de código (que é onde o Streamlit dá o botão de copiar).
    links = [c.value for c in at.code if c.value.startswith("?") or "?v=1" in c.value]
    assert links, [c.value for c in at.code]
    volta = _de_url(links[0])
    assert volta.texto == "AHU_ACL_CU_017, Cx. 5, D. 487" and volta.modo == "exata"
