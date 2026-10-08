"""Testes do WP 2.1: rota de cotas, BM25 por campos, semântico e RRF (spec §5.2–§5.5, A3)."""
import pandas as pd
import pytest

from ahu import busca
from ahu.dados import carregar_sem_cache
from tests.conftest import APP_DADOS, tem_artefatos

# As três formas da mesma cota (A3) e a cota antiga, com região no lugar do fundo (§5.3).
FORMAS_DA_MESMA_COTA = [
    "PT/AHU/CU/021/0006/00390",
    "AHU_ACL_CU_021, Cx. 6, D. 390.",
    "ahu acl cu 021 cx 6 d 390",
    "AHU_ACL_CU_021,Cx.6,D.390",
]


# ---------------------------------------------------------------------------
# Rota de cotas (§5.3) — sem precisar de corpus
# ---------------------------------------------------------------------------

def test_as_tres_formas_da_cota_dao_a_mesma_chave():
    chaves = {busca.chave_cota(f) for f in FORMAS_DA_MESMA_COTA}
    assert chaves == {"cu|21|6|390"}


def test_cota_antiga_com_regiao():
    assert busca.chave_cota("AHU-Santa Catarina, cx. 8, doc. 14") == "reg|santa catarina|8|14"
    assert busca.chave_cota("AHU-Rio de Janeiro, cx. 287, doc. 85.") == "reg|rio de janeiro|287|85"


def test_a3_numero_solto_nao_e_cota():
    """A consulta `17` não pode devolver documentos por cota (A3)."""
    for texto in ("17", "1750", "cx. 6", "d. 390", "390"):
        assert busca.chave_cota(texto) is None, texto


def test_texto_comum_nao_e_cota():
    for texto in ("deserção de soldados", "Lavradio", "", "ahu", "AHU-Bahia"):
        assert busca.chave_cota(texto) is None, texto


def test_cota_dentro_de_frase_e_reconhecida_e_o_resto_sobra():
    chave, resto = busca._trecho_da_cota("ver AHU_ACL_CU_021, Cx. 6, D. 390 sobre desertores")
    assert chave == "cu|21|6|390"
    assert resto == "ver sobre desertores"


# ---------------------------------------------------------------------------
# Corpus de mentira, com codificador falso
# ---------------------------------------------------------------------------

def descricoes(dados, resposta, quantas=5):
    return [str(dados.documentos.iloc[r.linha]["description"]) for r in resposta.resultados[:quantas]]


# ---------------------------------------------------------------------------
# Motor léxico e normalização (§5.1, §5.2)
# ---------------------------------------------------------------------------

def test_mello_encontra_melo(corpus_busca):
    """A grafia antiga é neutralizada: 'Mello' e 'Melo' caem na mesma forma (§5.1)."""
    por_mello = busca.buscar(corpus_busca, "Mello", modo="exata")
    por_melo = busca.buscar(corpus_busca, "Melo", modo="exata")
    assert por_mello.total == por_melo.total > 0
    assert {r.doc_uid for r in por_mello.resultados} == {r.doc_uid for r in por_melo.resultados}
    assert any("Manuel de Melo" in d for d in descricoes(corpus_busca, por_mello))
    assert any("Manoel de Mello" in d for d in descricoes(corpus_busca, por_mello))


def test_stopword_sozinha_nao_devolve_o_corpus(corpus_busca):
    assert busca.buscar(corpus_busca, "de", modo="exata").total == 0


def test_campo_de_pessoas_acha_pelo_titulo(corpus_busca):
    """O campo 'pessoas' do léxico traz nome principal, variantes e títulos (§5.2)."""
    r = busca.buscar(corpus_busca, "Lavradio", modo="exata")
    assert r.total > 0
    assert "Lavradio" in descricoes(corpus_busca, r)[0]


def test_evidencia_traz_os_termos_encontrados(corpus_busca):
    r = busca.buscar(corpus_busca, "deserção de soldados", modo="exata")
    assert r.termos_consulta == ["desercao", "soldados"]
    assert r.resultados[0].termos  # o primeiro resultado mostra o que casou
    assert set(r.resultados[0].termos) <= set(r.termos_consulta)
    assert r.resultados[0].motores == ["lexico"]


# ---------------------------------------------------------------------------
# Frase entre aspas (§5.5)
# ---------------------------------------------------------------------------

def test_frase_entre_aspas_exige_a_sequencia(corpus_busca):
    com_aspas = busca.buscar(corpus_busca, '"deserção de soldados"', modo="exata")
    sem_aspas = busca.buscar(corpus_busca, "deserção de soldados", modo="exata")
    assert 0 < com_aspas.total < sem_aspas.total
    for d in descricoes(corpus_busca, com_aspas, com_aspas.total):
        assert "deserção de soldados" in d


def test_frase_que_nao_existe_devolve_nada(corpus_busca):
    assert busca.buscar(corpus_busca, '"soldados de deserção"', modo="exata").total == 0


# ---------------------------------------------------------------------------
# Rota de cotas sobre o corpus (§5.3, A3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("forma", FORMAS_DA_MESMA_COTA)
def test_a3_as_formas_da_cota_devolvem_o_mesmo_documento_em_primeiro(corpus_busca, forma):
    r = busca.buscar(corpus_busca, forma, modo="exata")
    assert r.cota == "cu|21|6|390"
    assert r.total == 1
    assert r.resultados[0].posicoes == {"cota": 1}
    assert "obras do porto" in descricoes(corpus_busca, r)[0]


def test_a3_consulta_17_nao_usa_a_rota_de_cotas(corpus_busca):
    r = busca.buscar(corpus_busca, "17", modo="exata")
    assert r.cota is None
    assert all("cota" not in x.posicoes for x in r.resultados)


def test_cota_na_frase_vem_primeiro_e_o_resto_segue(corpus_busca):
    r = busca.buscar(corpus_busca, "AHU_ACL_CU_021, Cx. 6, D. 390 deserção", modo="exata")
    assert r.cota == "cu|21|6|390"
    assert r.resultados[0].posicoes["cota"] == 1
    assert "obras do porto" in descricoes(corpus_busca, r)[0]
    # O resto da frase continua valendo: os documentos de deserção vêm depois.
    assert r.total > 1
    assert any("deserção" in d for d in descricoes(corpus_busca, r, r.total)[1:])


def test_tabela_de_cotas_cobre_as_tres_colunas(corpus_busca):
    tabela = busca.tabela_de_cotas(corpus_busca)
    assert "cu|21|6|390" in tabela
    assert all(isinstance(v, list) and v for v in tabela.values())


# ---------------------------------------------------------------------------
# Modos, rigor e fusão (§5.4, §5.5)
# ---------------------------------------------------------------------------

def test_modo_exata_nao_usa_o_semantico(corpus_busca, encoder):
    r = busca.buscar(corpus_busca, "deserção de soldados", modo="exata", encoder=encoder)
    assert all(x.posicoes.keys() <= {"lexico", "cota"} for x in r.resultados)


def test_modo_conceitual_nao_usa_o_lexico(corpus_busca, encoder):
    r = busca.buscar(corpus_busca, "deserção de soldados", modo="conceitual", encoder=encoder,
                     cosseno_minimo=0.0)
    assert r.total > 0
    assert all("lexico" not in x.posicoes for x in r.resultados)


def test_automatico_funde_os_dois_motores(corpus_busca, encoder):
    r = busca.buscar(corpus_busca, "deserção de soldados", modo="automatico", encoder=encoder,
                     cosseno_minimo=0.0)
    assert any(x.posicoes.keys() >= {"lexico", "semantico"} for x in r.resultados)
    assert r.resultados[0].cosseno is not None


def test_rigor_alto_e_mais_restritivo_que_baixo(corpus_busca, encoder):
    alto = busca.buscar(corpus_busca, "soldados", modo="conceitual", rigor="alto", encoder=encoder,
                        cosseno_minimo=0.0)
    baixo = busca.buscar(corpus_busca, "soldados", modo="conceitual", rigor="baixo", encoder=encoder,
                         cosseno_minimo=0.0)
    assert 0 < alto.total <= baixo.total


def test_pesos_do_rrf_mudam_com_o_tamanho_da_consulta():
    curta = busca.pesos_do_rrf(["desercao", "soldados"])
    longa = busca.pesos_do_rrf(["desercao", "soldados", "no", "presidio", "do", "sul"])
    assert curta["lexico"] > curta["semantico"]
    assert longa["semantico"] > longa["lexico"]


def test_nota_e_por_posicao_nao_por_relevancia(corpus_busca, encoder):
    """§5.4/C4: a nota sai da posição em cada motor, não de um cosseno normalizado."""
    r = busca.buscar(corpus_busca, "deserção de soldados", encoder=encoder,
                     cosseno_minimo=0.0)
    primeiro = r.resultados[0]
    esperado = sum(r.pesos[m] / (busca.K_RRF + p) for m, p in primeiro.posicoes.items())
    assert primeiro.nota_rrf == pytest.approx(esperado)
    assert [x.posicao for x in r.resultados] == list(range(1, r.total + 1))


def test_consulta_sem_sentido_nao_passa_do_topo_de_cada_motor(corpus_busca, encoder):
    r = busca.buscar(corpus_busca, "xpto qwerty zzz", encoder=encoder, limite_motor=3, cosseno_minimo=0.0)
    assert r.total <= 6  # no máximo o topo de cada um dos dois motores
    r2 = busca.buscar(corpus_busca, "soldados", encoder=encoder, limite_motor=2, cosseno_minimo=0.0)
    assert r2.total <= 4


def test_paginacao_de_25_com_total_real(corpus_busca):
    r = busca.buscar(corpus_busca, "carta", modo="exata")
    assert len(r.pagina(1, 25)) == min(25, r.total)
    assert len(r.pagina(1, 3)) == 3 and len(r.pagina(2, 3)) <= 3
    assert r.pagina(1, 3)[0].posicao == 1 and r.pagina(2, 3)[0].posicao == 4


def test_modo_e_rigor_invalidos_avisam(corpus_busca):
    with pytest.raises(ValueError, match="Modo desconhecido"):
        busca.buscar(corpus_busca, "x", modo="fuzzy")
    with pytest.raises(ValueError, match="Rigor desconhecido"):
        busca.buscar(corpus_busca, "x", rigor="altissimo")


# ---------------------------------------------------------------------------
# A3 sobre os artefatos de verdade (pulado se app_dados/ não existir)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a3_nos_dados_reais():
    d = carregar_sem_cache(APP_DADOS)
    uids = set()
    for forma in FORMAS_DA_MESMA_COTA:
        r = busca.buscar(d, forma, modo="exata")
        assert r.cota == "cu|21|6|390", forma
        assert r.total >= 1, forma
        uids.add(r.resultados[0].doc_uid)
    assert len(uids) == 1  # as três formas levam ao mesmo documento, em 1º

    r = busca.buscar(d, "17", modo="exata")
    assert r.cota is None
    assert all("cota" not in x.posicoes for x in r.resultados)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_busca_lexica_real_e_rapida_e_faz_sentido():
    import time

    d = carregar_sem_cache(APP_DADOS)
    t = time.time()
    r = busca.buscar(d, "deserção de soldados", modo="exata")
    assert time.time() - t < 1.0  # A3: consulta em menos de 1 s
    assert r.total > 50
    primeiras = [str(d.documentos.iloc[x.linha]["description"]).lower()
                 for x in r.resultados[:10]]
    assert sum(("deser" in p or "soldado" in p) for p in primeiras) >= 8


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a2_lavradio_tem_163_documentos_na_camada_de_pessoas():
    """A2 é do WP 2.4, mas o dado que o sustenta já tem de estar certo."""
    d = carregar_sem_cache(APP_DADOS)
    pessoas = d.pessoas
    lavradio = pessoas[pessoas["titulos"].apply(
        lambda ts: any("lavradio" in str(t).lower() for t in (ts if ts is not None else [])))]
    assert len(lavradio) == 1
    uid = lavradio.iloc[0]["pessoa_uid"]
    assert int((d.doc_pessoa["pessoa_uid"] == uid).sum()) == 163
    assert int(pd.Series(lavradio.iloc[0]["n_documentos"]).iloc[0]) == 163


