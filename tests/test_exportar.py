"""Testes do WP 3.2: exportação em CSV e JSON, com proveniência (spec §8.2).

Dois testes são os que a spec cobra por nome: **o CSV reabre com as mesmas linhas** e **a
proveniência traz a `Consulta`, as versões e o DOI**. Em volta deles, a D1: nem o CSV nem o JSON
podem conter `scribe_mediation_likely`.
"""
import json

import pytest

from ahu import exportar as ex
from ahu.consulta import Consulta, executar
from ahu.dados import carregar_sem_cache
from ahu.ficha import CAMPOS_PROIBIDOS
from ahu.filtros import Filtros
from tests.conftest import APP_DADOS, tem_artefatos


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


@pytest.fixture
def uids(corpus_busca):
    return corpus_busca.documentos["doc_uid"].tolist()[:5]


# ---------------------------------------------------------------------------
# As linhas
# ---------------------------------------------------------------------------

def test_colunas_sao_as_do_8_2(corpus_busca, uids):
    tabela = ex.tabela(corpus_busca, uids)
    assert list(tabela.columns) == [n for n, _ in ex.COLUNAS]
    for obrigatoria in ("doc_uid", "reference_code", "new_code", "ano", "pasta", "tipologia",
                        "pessoas", "destinatario", "srsp", "categoria_social", "vetor",
                        "origem_remetente", "origem_nivel", "link_digitarq", "link_resgate"):
        assert obrigatoria in tabela.columns, obrigatoria
    assert len(tabela) == len(uids)


def test_a_ordem_pedida_e_a_ordem_exportada(corpus_busca, uids):
    ao_contrario = list(reversed(uids))
    assert [r["doc_uid"] for r in ex.linhas(corpus_busca, ao_contrario)] == ao_contrario


def test_justificativa_so_no_json(corpus_busca, uids):
    assert "justificativa_ia" not in [n for n, _ in ex.COLUNAS]
    assert "justificativa_ia" in [n for n, _ in ex.COLUNAS_JSON]
    conteudo = json.loads(ex.para_json(corpus_busca, uids))
    assert "justificativa_ia" in conteudo["documentos"][0]


def test_doc_uid_desconhecido_erra_alto(corpus_busca):
    with pytest.raises(KeyError, match="doc_uid desconhecido"):
        ex.linhas(corpus_busca, ["ahu-sul:naoexiste"])


# ---------------------------------------------------------------------------
# CSV: UTF-8 com BOM, `;`, e reabre com as mesmas linhas
# ---------------------------------------------------------------------------

def test_csv_tem_bom_e_separador_de_ponto_e_virgula(corpus_busca, uids):
    bruto = ex.para_csv(corpus_busca, uids)
    assert bruto.startswith(ex.BOM.encode("utf-8"))      # o Excel em português precisa do BOM
    texto = bruto.decode("utf-8-sig")
    cabecalho = next(ln for ln in texto.splitlines() if not ln.startswith("#"))
    assert cabecalho.startswith("doc_uid;")
    assert "\t" not in cabecalho


def test_o_csv_reabre_com_as_mesmas_linhas(corpus_busca, uids):
    """O que a spec §10 cobra: o CSV reabre com as mesmas linhas."""
    esperado = ex.linhas(corpus_busca, uids)
    tabela, _ = ex.ler_csv(ex.para_csv(corpus_busca, uids))

    assert len(tabela) == len(esperado)
    assert list(tabela.columns) == [n for n, _ in ex.COLUNAS]
    for lida, original in zip(tabela.to_dict("records"), esperado):
        for coluna in tabela.columns:
            assert str(lida[coluna]) == str(original[coluna]), (original["doc_uid"], coluna)


def test_resumo_com_ponto_e_virgula_e_quebra_de_linha_sobrevive(corpus_busca):
    """O separador dentro do texto não pode partir a linha: o csv cita o campo."""
    docs = corpus_busca.documentos
    uid = docs["doc_uid"].iloc[0]
    original = docs["description"].copy()
    try:
        docs.loc[docs.index[0], "description"] = 'CARTA; com "aspas", ponto-e-vírgula\ne quebra'
        tabela, _ = ex.ler_csv(ex.para_csv(corpus_busca, [uid]))
        assert len(tabela) == 1
        assert tabela["resumo_ahu"].iloc[0] == docs["description"].iloc[0]
    finally:
        docs["description"] = original


def test_resumo_que_comeca_com_cerquilha_nao_e_lido_como_comentario(corpus_busca):
    """Só as linhas de comentário **do início** são tiradas."""
    docs = corpus_busca.documentos
    uid = docs["doc_uid"].iloc[0]
    original = docs["description"].copy()
    try:
        docs.loc[docs.index[0], "description"] = "# não sou proveniência"
        tabela, prov = ex.ler_csv(ex.para_csv(corpus_busca, [uid]))
        assert len(tabela) == 1
        assert tabela["resumo_ahu"].iloc[0] == "# não sou proveniência"
        assert all(ln.startswith("#") for ln in prov)
    finally:
        docs["description"] = original


# ---------------------------------------------------------------------------
# Proveniência (§8.2)
# ---------------------------------------------------------------------------

def test_proveniencia_traz_consulta_versoes_e_doi(corpus_busca):
    """O que a spec §10 cobra: a proveniência traz a `Consulta`, as versões e o DOI."""
    consulta = Consulta(texto="deserção de soldados", modo="conceitual", rigor="alto",
                        filtros=Filtros(tipologias=["CARTA"]), semente=42, pagina=2)
    prov = ex.proveniencia(corpus_busca, consulta, total=7, origem="resultado")

    # a Consulta completa
    assert prov["consulta"]["texto"] == "deserção de soldados"
    assert prov["consulta"]["modo"] == "conceitual" and prov["consulta"]["rigor"] == "alto"
    assert prov["consulta"]["semente"] == 42 and prov["consulta"]["pagina"] == 2
    assert prov["consulta"]["filtros_ativos"]["tipologias"] == ["CARTA"]
    # a data e a contagem
    assert prov["gerado_em"] and prov["documentos"] == 7 and prov["origem"] == "resultado"
    # a versão do app
    assert prov["app_versao"]["versao"]
    # as camadas ativas, com versão, lidas do manifesto
    camadas = prov["corpus"]["camadas_ativas"]
    assert {"tipologia", "sociolinguistica", "entidades", "biografias", "embeddings"} <= set(camadas)
    assert all(c["camada"] for c in camadas.values())
    assert camadas["embeddings"].get("modelo")
    # o SHA-256 do manifesto e o modelo
    assert len(prov["corpus"]["manifesto_sha256"]) == 64
    assert prov["corpus"]["modelo_embeddings"]["nome"]
    # o DOI e a citação
    assert prov["doi"]["preprint"] == ex.DOI_PREPRINT
    assert "Classificador AHU-Sul" in prov["citacao"]


def test_proveniencia_sem_consulta_nao_quebra(corpus_busca):
    prov = ex.proveniencia(corpus_busca)
    assert prov["consulta"] is None and prov["corpus"]["manifesto_sha256"]


def test_csv_carrega_a_proveniencia_em_comentario(corpus_busca, uids):
    consulta = Consulta(texto="gado", modo="exata")
    bruto = ex.para_csv(corpus_busca, uids, ex.proveniencia(corpus_busca, consulta, len(uids)))
    _, prov = ex.ler_csv(bruto)
    texto = "\n".join(prov)
    assert "classificador-ahu-sul" in texto
    assert "'gado'" in texto and "modo=exata" in texto
    assert ex.DOI_PREPRINT in texto
    assert "manifesto (sha256)" in texto
    assert "embeddings=" in texto
    # Uma linha de comentário nunca pode conter quebra, senão ela viraria duas.
    assert all("\n" not in ln and "\r" not in ln for ln in prov)


def test_json_tem_proveniencia_e_esquema(corpus_busca, uids):
    conteudo = json.loads(ex.para_json(corpus_busca, uids,
                                       ex.proveniencia(corpus_busca, Consulta(), len(uids))))
    assert conteudo["schema"] == "exportacao-ahu-sul/1"
    assert conteudo["proveniencia"]["doi"]["preprint"] == ex.DOI_PREPRINT
    assert len(conteudo["documentos"]) == len(uids)


def test_versao_do_app_sai_do_pyproject():
    versao = ex.versao_do_app()
    assert versao["versao"] == "3.0.0.dev0"
    assert versao["commit"] is None or len(versao["commit"]) >= 7


# ---------------------------------------------------------------------------
# D1: o campo do PDF não sai em exportação nenhuma
# ---------------------------------------------------------------------------

def test_a_lista_de_colunas_nao_tem_campo_proibido():
    assert ex.campos_proibidos_presentes() == []


def test_d1_nem_no_csv_nem_no_json(corpus_busca, uids):
    consulta = Consulta(texto="deserção")
    prov = ex.proveniencia(corpus_busca, consulta, len(uids))
    csv_bruto = ex.para_csv(corpus_busca, uids, prov).decode("utf-8-sig")
    json_bruto = ex.para_json(corpus_busca, uids, prov).decode("utf-8")
    for campo in CAMPOS_PROIBIDOS:
        assert campo not in csv_bruto, campo
        assert campo not in json_bruto, campo


# ---------------------------------------------------------------------------
# Nos dados de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_exportar_o_resultado_de_uma_busca_real(corpus_real):
    resposta = executar(corpus_real, Consulta(texto="deserção de soldados", modo="exata"))
    assert resposta.total > 50
    uids = [r.doc_uid for r in resposta.resultados[:50]]
    prov = ex.proveniencia(corpus_real, Consulta(texto="deserção de soldados", modo="exata"),
                           len(uids), origem="resultado")

    tabela, comentarios = ex.ler_csv(ex.para_csv(corpus_real, uids, prov))
    assert tabela["doc_uid"].tolist() == uids
    assert comentarios and tabela["link_digitarq"].str.startswith("https://").all()
    assert (tabela["resumo_ahu"].str.len() > 0).all()

    conteudo = json.loads(ex.para_json(corpus_real, uids, prov))
    assert [d["doc_uid"] for d in conteudo["documentos"]] == uids


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a1_a_exportacao_nunca_traz_nivel_proibido(corpus_real):
    """A1: nenhuma biografia REJEITADA ou BAIXA aparece em exportação."""
    from ahu.ficha import NIVEIS_PROIBIDOS

    uids = corpus_real.documentos["doc_uid"].tolist()[:400]
    bruto = ex.para_csv(corpus_real, uids).decode("utf-8-sig")
    for nivel in NIVEIS_PROIBIDOS:
        assert nivel not in bruto, nivel
    tabela, _ = ex.ler_csv(ex.para_csv(corpus_real, uids))
    assert set(tabela["origem_nivel"].unique()) <= {"", "ALTA", "MEDIA"}


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_todo_o_corpus_exporta_sem_quebrar(corpus_real):
    """O caso extremo: as 7.051 linhas saem e reabrem com o mesmo número de linhas."""
    uids = corpus_real.documentos["doc_uid"].tolist()
    tabela, _ = ex.ler_csv(ex.para_csv(corpus_real, uids))
    assert len(tabela) == len(uids)
    assert tabela["doc_uid"].tolist() == uids
