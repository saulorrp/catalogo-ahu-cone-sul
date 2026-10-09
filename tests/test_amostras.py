"""Testes do WP 3.5: amostra diacrônica, amostra estratificada e plano EINS (§7.2–§7.4).

Dois critérios de aceite por nome:

* **A7** — a mesma semente gera a mesma amostra estratificada
  (`test_a7_mesma_semente_mesma_amostra`), e uma semente diferente gera outra;
* **A9** — o plano de coleta valida contra o esquema `eins-plano-coleta/1`
  (`test_a9_o_plano_valida_contra_o_esquema`).
"""
import json
from pathlib import Path

import pytest

from ahu import amostras as am
from ahu import pessoas as mp
from ahu.amostras import ErroDeAmostra
from ahu.consulta import Consulta, executar
from ahu.dados import carregar_sem_cache
from ahu.ficha import CAMPOS_PROIBIDOS, NIVEIS_PROIBIDOS
from tests.conftest import APP_DADOS, tem_artefatos

RAIZ = Path(__file__).resolve().parent.parent
ESQUEMA = RAIZ / "schemas" / "eins-plano-coleta-1.json"


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


@pytest.fixture(scope="module")
def lavradio(corpus_real):
    return mp.encontrar(corpus_real, "Lavradio")[0].pessoa_uid


def validar(plano: dict) -> None:
    import jsonschema

    jsonschema.validate(plano, json.loads(ESQUEMA.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# autoria_verificada por entidade (§7.2)
# ---------------------------------------------------------------------------

def test_autoria_verificada_acha_o_nome_na_descricao(corpus_busca):
    """O documento 10 do conftest cita "marquês do Lavradio", que é título da pessoa."""
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    docs = corpus_busca.documentos
    do_lavradio = [u for u, d in zip(docs["doc_uid"], docs["description"])
                   if "Lavradio" in str(d)]
    assert do_lavradio
    for uid in do_lavradio:
        assert am.autoria_verificada(corpus_busca, uid, pessoa.pessoa_uid)


def test_autoria_verificada_e_falsa_quando_o_nome_nao_aparece(corpus_busca):
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    docs = corpus_busca.documentos
    sem = [u for u, d in zip(docs["doc_uid"], docs["description"])
           if "Lavradio" not in str(d) and "Almeida" not in str(d)]
    assert sem
    assert not am.autoria_verificada(corpus_busca, sem[0], pessoa.pessoa_uid)


def test_autoria_verificada_neutraliza_grafia_antiga(corpus_busca):
    """§5.1: quem se chama "Melo" é reconhecido numa descrição que escreve "Mello"."""
    pessoas = corpus_busca.pessoas
    docs = corpus_busca.documentos
    uid_pessoa = pessoas["pessoa_uid"].iloc[0]
    nomes = pessoas["nome_exibicao"].copy()
    descricoes = docs["description"].copy()
    try:
        pessoas.loc[pessoas.index[0], "nome_exibicao"] = "Manuel de Melo"
        docs.loc[docs.index[0], "description"] = "REQUERIMENTO de Manoel de Mello e Castro."
        assert am.autoria_verificada(corpus_busca, docs["doc_uid"].iloc[0], uid_pessoa)
    finally:
        pessoas["nome_exibicao"] = nomes
        docs["description"] = descricoes


def test_forma_curta_nao_casa_por_acidente(corpus_busca):
    """Uma forma de até três letras casaria dentro de qualquer palavra: é ignorada."""
    pessoas = corpus_busca.pessoas
    docs = corpus_busca.documentos
    uid_pessoa = pessoas["pessoa_uid"].iloc[0]
    nomes = pessoas["nome_exibicao"].copy()
    variantes = pessoas["variantes_nome"].copy()
    descricoes = docs["description"].copy()
    try:
        pessoas.loc[pessoas.index[0], "nome_exibicao"] = "Sá"
        pessoas.at[pessoas.index[0], "variantes_nome"] = []
        docs.loc[docs.index[0], "description"] = "CARTA sobre a saca de farinha."
        assert not am.autoria_verificada(corpus_busca, docs["doc_uid"].iloc[0], uid_pessoa)
    finally:
        pessoas["nome_exibicao"] = nomes
        pessoas["variantes_nome"] = variantes
        docs["description"] = descricoes


def test_pessoa_desconhecida_nao_tem_autoria_verificada(corpus_busca):
    uid = corpus_busca.documentos["doc_uid"].iloc[0]
    assert not am.autoria_verificada(corpus_busca, uid, "pes:naoexiste")


# ---------------------------------------------------------------------------
# Amostra diacrônica (§7.2)
# ---------------------------------------------------------------------------

def test_pessoa_desconhecida_erra_alto(corpus_busca):
    with pytest.raises(ErroDeAmostra, match="pessoa_uid desconhecido"):
        am.amostra_diacronica(corpus_busca, "pes:naoexiste")


def test_maximo_por_ano_invalido_erra_alto(corpus_busca):
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    with pytest.raises(ErroDeAmostra, match="1 ou mais"):
        am.amostra_diacronica(corpus_busca, pessoa.pessoa_uid, maximo_por_ano=0)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_amostra_diacronica_e_cronologica_e_em_lotes_de_dez(corpus_real, lavradio):
    amostra = am.amostra_diacronica(corpus_real, lavradio)
    assert amostra.tipo == "diacronica" and amostra.pessoa_uid == lavradio

    anos = [int(corpus_real.documentos.iloc[corpus_real.indice_de(u)]["ano_isolado"])
            for u in amostra.doc_uids]
    assert anos == sorted(anos)                       # ordem cronológica
    assert all(len(lote) == am.TAMANHO_LOTE for lote in amostra.lotes[:-1])
    assert 1 <= len(amostra.lotes[-1]) <= am.TAMANHO_LOTE
    assert sum(len(lote) for lote in amostra.lotes) == len(amostra)

    # Documento sem data fica fora, e o relatório conta.
    r = amostra.relatorio
    assert r["vinculados"] == 163 and r["sem_data"] == 1
    assert len(amostra) == r["sorteados"] == r["vinculados"] - r["sem_data"]


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_espacamento_limita_documentos_por_ano(corpus_real, lavradio):
    """§7.2: "opção de no máximo N documentos por ano"."""
    inteira = am.amostra_diacronica(corpus_real, lavradio)
    for maximo in (1, 2, 5):
        espacada = am.amostra_estratificada  # noqa: F841  (só para o leitor não confundir)
        espacada = am.amostra_diacronica(corpus_real, lavradio, maximo_por_ano=maximo)
        por_ano = {}
        for uid in espacada.doc_uids:
            ano = int(corpus_real.documentos.iloc[corpus_real.indice_de(uid)]["ano_isolado"])
            por_ano[ano] = por_ano.get(ano, 0) + 1
        assert all(n <= maximo for n in por_ano.values()), (maximo, por_ano)
        assert len(espacada) < len(inteira)
        assert set(por_ano) == set(inteira.relatorio["anos"])   # todos os anos sobrevivem
        assert espacada.relatorio["cortados_pelo_maximo_por_ano"] > 0


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_espacamento_e_deterministico(corpus_real, lavradio):
    """A amostra diacrônica não tem semente: duas chamadas dão o mesmo."""
    uma = am.amostra_diacronica(corpus_real, lavradio, maximo_por_ano=2)
    outra = am.amostra_diacronica(corpus_real, lavradio, maximo_por_ano=2)
    assert uma.doc_uids == outra.doc_uids


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_exigir_autoria_verificada_so_tira_documentos(corpus_real, lavradio):
    com = am.amostra_diacronica(corpus_real, lavradio, so_autoria_verificada=True)
    sem = am.amostra_diacronica(corpus_real, lavradio, so_autoria_verificada=False)
    assert set(com.doc_uids) <= set(sem.doc_uids)
    assert com.relatorio["sem_autoria_verificada"] == len(sem) - len(com)


# ---------------------------------------------------------------------------
# Amostra estratificada (§7.3) e o A7
# ---------------------------------------------------------------------------

@pytest.fixture
def conjunto(corpus_busca):
    return corpus_busca.documentos["doc_uid"].tolist()


@pytest.mark.parametrize("estratos, erro", [
    ([], "1 ou 2 estratos"),
    (["decada", "tipologia", "pasta"], "1 ou 2 estratos"),
    (["inventado"], "Estrato desconhecido"),
    (["decada", "decada"], "são o mesmo"),
])
def test_estratos_invalidos_erram_alto(corpus_busca, conjunto, estratos, erro):
    with pytest.raises(ErroDeAmostra, match=erro):
        am.amostra_estratificada(corpus_busca, conjunto, estratos, 2, 42)


def test_n_invalido_e_conjunto_vazio_erram_alto(corpus_busca, conjunto):
    with pytest.raises(ErroDeAmostra, match="1 ou mais"):
        am.amostra_estratificada(corpus_busca, conjunto, ["decada"], 0, 42)
    with pytest.raises(ErroDeAmostra, match="Não há documentos"):
        am.amostra_estratificada(corpus_busca, [], ["decada"], 2, 42)


def test_a7_mesma_semente_mesma_amostra(corpus_busca, conjunto):
    """A7: a mesma semente gera a mesma amostra estratificada."""
    import random

    for estratos in (["tipologia"], ["decada", "tipologia"]):
        uma = am.amostra_estratificada(corpus_busca, conjunto, estratos, 2, 42)
        # Mexer no `random` global entre as duas chamadas não pode mudar nada.
        random.seed(999)
        random.random()
        outra = am.amostra_estratificada(corpus_busca, conjunto, estratos, 2, 42)
        assert uma.doc_uids == outra.doc_uids, estratos
        assert uma.lotes == outra.lotes
        assert uma.relatorio == outra.relatorio


def test_a7_a_ordem_do_conjunto_nao_muda_a_amostra(corpus_busca, conjunto):
    """A amostra não pode depender da ordem em que a busca devolveu os documentos."""
    uma = am.amostra_estratificada(corpus_busca, conjunto, ["tipologia"], 2, 42)
    outra = am.amostra_estratificada(corpus_busca, list(reversed(conjunto)), ["tipologia"], 2, 42)
    assert sorted(uma.doc_uids) == sorted(outra.doc_uids)


def test_semente_diferente_da_amostra_diferente(corpus_busca):
    docs = corpus_busca.documentos
    conjunto = docs["doc_uid"].tolist()
    # Um estrato só, com todos dentro, para o sorteio ter o que variar.
    uma = am.amostra_estratificada(corpus_busca, conjunto, ["pasta"], 3, 1)
    outra = am.amostra_estratificada(corpus_busca, conjunto, ["pasta"], 3, 2)
    assert len(uma) == len(outra)
    assert uma.doc_uids != outra.doc_uids


def test_n_por_estrato_e_respeitado_e_o_relatorio_conta(corpus_busca, conjunto):
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["tipologia"], 1, 42)
    tipologias = [corpus_busca.documentos.iloc[corpus_busca.indice_de(u)]["tipologia"]
                  for u in amostra.doc_uids]
    assert len(tipologias) == len(set(tipologias))          # um por tipologia
    r = amostra.relatorio
    assert r["conjunto"] == len(conjunto)
    assert r["sorteados"] == len(amostra) == r["estratos"]
    assert all(v["sorteados"] <= v["disponiveis"] for v in r["por_estrato"].values())


def test_estrato_insuficiente_entra_com_o_que_ha_e_e_relatado(corpus_busca, conjunto):
    """§7.3: "estratos sem documentos suficientes" têm de aparecer no relatório."""
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["tipologia"], 5, 42)
    r = amostra.relatorio
    assert r["insuficientes"]
    for chave in r["insuficientes"]:
        assert r["por_estrato"][chave]["disponiveis"] < 5
        assert r["por_estrato"][chave]["sorteados"] == r["por_estrato"][chave]["disponiveis"]
    tabela = am.tabela_do_relatorio(amostra)
    assert set(tabela.columns) == {"Estrato", "Disponíveis", "Sorteados", "Suficiente"}
    assert (tabela["Suficiente"] == "não").sum() == len(r["insuficientes"])


def test_dois_estratos_cruzam_os_valores(corpus_busca, conjunto):
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["decada", "tipologia"], 1, 42)
    assert amostra.estratos == ["decada", "tipologia"]
    assert all(" · " in chave for chave in amostra.relatorio["por_estrato"])


def test_cada_lote_pertence_a_um_estrato(corpus_busca, conjunto):
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["tipologia"], 3, 42)
    assert len(amostra.estrato_do_lote) == len(amostra.lotes)
    for lote, estrato in zip(amostra.lotes, amostra.estrato_do_lote):
        for uid in lote:
            assert am.valor_do_estrato(corpus_busca, uid, "tipologia") == estrato


def test_valor_do_estrato_para_cada_tipo(corpus_busca):
    uid = corpus_busca.documentos["doc_uid"].iloc[0]
    for estrato in am.ESTRATOS:
        valor = am.valor_do_estrato(corpus_busca, uid, estrato)
        assert isinstance(valor, str) and valor
    assert am.valor_do_estrato(corpus_busca, uid, "decada").endswith("0")
    with pytest.raises(ErroDeAmostra, match="Estrato desconhecido"):
        am.valor_do_estrato(corpus_busca, uid, "cor_da_tinta")


def test_documento_sem_valor_cai_num_estrato_proprio(corpus_busca):
    docs = corpus_busca.documentos
    uid = docs["doc_uid"].iloc[0]
    original = docs["ano_isolado"].copy()
    try:
        docs.loc[docs.index[0], "ano_isolado"] = None
        assert am.valor_do_estrato(corpus_busca, uid, "decada") == am.SEM_VALOR
    finally:
        docs["ano_isolado"] = original


# ---------------------------------------------------------------------------
# Plano de coleta (§7.4) e o A9
# ---------------------------------------------------------------------------

def test_o_esquema_existe_e_e_json_valido():
    esquema = json.loads(ESQUEMA.read_text(encoding="utf-8"))
    assert esquema["properties"]["schema"]["const"] == am.ESQUEMA_EINS


def test_a9_o_plano_valida_contra_o_esquema(corpus_busca, conjunto):
    """A9: o plano de coleta valida contra o esquema `eins-plano-coleta/1`."""
    diacronica = am.amostra_diacronica(
        corpus_busca, mp.encontrar(corpus_busca, "Lavradio")[0].pessoa_uid,
        so_autoria_verificada=False)
    estratificada = am.amostra_estratificada(corpus_busca, conjunto, ["decada", "tipologia"],
                                             2, 42)
    for amostra in (diacronica, estratificada):
        if not len(amostra):
            continue
        plano = am.plano_de_coleta(corpus_busca, amostra, Consulta(texto="gado", modo="exata"))
        validar(plano)
        assert plano["schema"] == am.ESQUEMA_EINS
        assert plano["origem"]["app"] == "classificador-ahu-sul"
        assert plano["origem"]["corpus"]
        assert plano["origem"]["amostra"]["tipo"] == amostra.tipo
        assert plano["origem"]["semente"] == amostra.semente
        assert sum(len(lote["documentos"]) for lote in plano["lotes"]) == len(amostra)


def test_plano_fora_do_esquema_e_recusado_pelo_validador(corpus_busca, conjunto):
    """Se o validador aceitasse qualquer coisa, o A9 não valeria nada."""
    import jsonschema

    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["decada"], 1, 42)
    plano = am.plano_de_coleta(corpus_busca, amostra)
    plano["schema"] = "eins-plano-coleta/99"
    with pytest.raises(jsonschema.ValidationError):
        validar(plano)

    plano = am.plano_de_coleta(corpus_busca, amostra)
    plano["lotes"][0]["documentos"][0]["inventado"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validar(plano)


def test_d1_o_plano_nao_leva_o_campo_do_pdf(corpus_busca, conjunto):
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["decada"], 2, 42)
    bruto = am.plano_em_json(corpus_busca, amostra).decode("utf-8")
    for campo in CAMPOS_PROIBIDOS:
        assert campo not in bruto, campo
    for nivel in NIVEIS_PROIBIDOS:
        assert nivel not in bruto, nivel


def test_plano_em_json_e_utf8_legivel(corpus_busca, conjunto):
    amostra = am.amostra_estratificada(corpus_busca, conjunto, ["decada"], 1, 42)
    bruto = am.plano_em_json(corpus_busca, amostra)
    conteudo = json.loads(bruto.decode("utf-8"))
    assert conteudo["schema"] == am.ESQUEMA_EINS
    # `ensure_ascii=False`: os acentos saem como acentos, não como \\u00e7.
    assert b"\\u00" not in bruto


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a9_no_corpus_real_com_a_amostra_do_lavradio(corpus_real, lavradio):
    amostra = am.amostra_diacronica(corpus_real, lavradio, maximo_por_ano=2)
    plano = am.plano_de_coleta(corpus_real, amostra, Consulta())
    validar(plano)
    assert plano["origem"]["amostra"]["pessoa_uid"] == lavradio
    assert plano["origem"]["amostra"]["maximo_por_ano"] == 2
    documentos = [d for lote in plano["lotes"] for d in lote["documentos"]]
    assert len(documentos) == len(amostra)
    assert all(d["reference_code"] and d["reference_code"].startswith("PT/AHU/CU/")
               for d in documentos)
    assert all(d["autoria_verificada"] for d in documentos)
    assert all(d["links"]["digitarq"] for d in documentos)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a9_no_corpus_real_com_amostra_estratificada(corpus_real):
    resposta = executar(corpus_real, Consulta(texto="deserção de soldados", modo="exata"))
    uids = [r.doc_uid for r in resposta.resultados]
    amostra = am.amostra_estratificada(corpus_real, uids, ["decada", "pasta"], 2, 42)
    plano = am.plano_de_coleta(corpus_real, amostra,
                               Consulta(texto="deserção de soldados", modo="exata"))
    validar(plano)
    assert plano["origem"]["consulta"]["texto"] == "deserção de soldados"
    assert plano["origem"]["semente"] == 42
    assert plano["relatorio"]["estratos"] == amostra.relatorio["estratos"]


# ---------------------------------------------------------------------------
# A aba Amostras no app de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_tem_a_aba_amostras_funcionando():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=240)
    at.query_params["v"] = "1"
    at.query_params["q"] = "AHU_ACL_CU_017, Cx. 5, D. 487"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    # A aba de amostras não é mais "em breve": há um radio com os dois tipos.
    tipos = [r for r in at.radio if r.label == "Tipo de amostra"]
    assert tipos and list(tipos[0].options) == list(__import__("app").TIPOS_DE_AMOSTRA)

    # E a estratificada corre sobre o resultado da busca, com semente visível.
    tipos[0].set_value(__import__("app").TIPOS_DE_AMOSTRA[1]).run()
    assert not at.exception, [e.value for e in at.exception]
    sementes = [n for n in at.number_input if n.label == "Semente"]
    assert sementes and sementes[0].value == 42
