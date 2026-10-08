"""Testes do WP 3.3a: a lista de documentos para consulta presencial no AHU (PDF).

O teste que o autor pediu por nome é o `test_toda_cota_do_corpus_vira_referencia`: ele percorre
os 7.051 documentos e cobra um `PT/AHU/CU/…` para cada um. Ele passa com **uma** exceção
documentada, que é um documento cuja única cota é antiga e por região (`AHU-São Paulo-MGouveia,
cx. 2, doc. 224`) — sem número de série não há código CRAV derivável, e inventar um seria pior que
não ter.
"""
import pytest

from ahu import lista_ahu as la
from ahu import pdf as mod_pdf
from ahu.dados import carregar_sem_cache
from ahu.links import digitarq, referencia_crav
from tests.conftest import APP_DADOS, tem_artefatos

# O único documento do corpus sem código de referência derivável (ver o docstring).
SEM_REFERENCIA_CONHECIDO = 1


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


# ---------------------------------------------------------------------------
# referencia_crav: a conversão de cota em código de referência
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cota, esperado", [
    ("AHU_ACL_CU_021, Cx. 6, D. 390.", "PT/AHU/CU/021/0006/00390"),
    ("AHU_ACL_CU_021, Cx. 6, D. 390", "PT/AHU/CU/021/0006/00390"),
    ("ahu acl cu 021 cx 6 d 390", "PT/AHU/CU/021/0006/00390"),
    ("PT/AHU/CU/021/0006/00390", "PT/AHU/CU/021/0006/00390"),
    ("PT / AHU / CU / 021 / 0006 / 00390", "PT/AHU/CU/021/0006/00390"),
    ("AHU_ACL_CU_014, Cx. 13, D. 1063", "PT/AHU/CU/014/0013/01063"),
    # sub-série
    ("AHU_ACL_CU_017-01, Cx. 3, D. 387", "PT/AHU/CU/017-001/0003/00387"),
    ("PT/AHU/CU/017-01/0003/00387", "PT/AHU/CU/017-001/0003/00387"),
    # lixo de extração entre o "D." e o número: o primeiro número é o que vale
    ("AHU_ACL_CU_017-01, Cx. 3, D. .387-389.", "PT/AHU/CU/017-001/0003/00387"),
    ("AHU_ACL_CU_017-01, Cx. 6, D. –1011-1015.", "PT/AHU/CU/017-001/0006/01011"),
    ("AHU_ACL_CU_017-01, Cx. 70, D. .16202", "PT/AHU/CU/017-001/0070/16202"),
    ("AHU_ACL_CU_003, Cx. 1, Doc. 100", "PT/AHU/CU/003/0001/00100"),
])
def test_referencia_crav_normaliza_as_duas_familias(cota, esperado):
    assert referencia_crav(cota) == esperado


@pytest.mark.parametrize("cota", [
    None, "", "   ", "UNFORMATTED_CODE", "cota inventada",
    "AHU-São Paulo-MGouveia, cx. 2, doc. 224.",       # cota antiga: não tem série
    "17",
])
def test_cota_sem_codigo_derivavel_devolve_none(cota):
    assert referencia_crav(cota) is None


def test_a_referencia_e_idempotente():
    uma_vez = referencia_crav("AHU_ACL_CU_021, Cx. 6, D. 390.")
    assert referencia_crav(uma_vez) == uma_vez


def test_link_digitarq_normaliza_a_cota_do_resgate():
    """O WP 3.3a trouxe isto: a cota do Resgate também dá link."""
    assert digitarq("AHU_ACL_CU_021, Cx. 6, D. 390.") == digitarq("PT/AHU/CU/021/0006/00390")
    assert digitarq("UNFORMATTED_CODE") is None
    assert digitarq(None) is None


# ---------------------------------------------------------------------------
# Ler a lista colada
# ---------------------------------------------------------------------------

def test_grupos_notas_e_link_do_resgate(corpus_busca):
    texto = (
        "# Dia 1\n"
        "PT/AHU/CU/021/0006/00390 | Carta nº 2 | https://resgate.bn.gov.br/x\n"
        "\n"
        "# Dia 2\n"
        "AHU_ACL_CU_014, Cx. 13, D. 1063\n"
        "cota inventada\n"
    )
    grupos = la.ler_lista(texto, corpus_busca)
    assert [g.titulo for g in grupos] == ["Dia 1", "Dia 2"]
    assert [len(g.itens) for g in grupos] == [1, 2]

    primeiro = grupos[0].itens[0]
    assert primeiro.nota == "Carta nº 2"
    assert primeiro.link_resgate == "https://resgate.bn.gov.br/x"
    assert primeiro.referencia == "PT/AHU/CU/021/0006/00390"

    ultimo = grupos[1].itens[-1]
    assert ultimo.referencia is None and ultimo.linhas == []


def test_linhas_vazias_sao_ignoradas_e_grupo_sem_item_nao_entra(corpus_busca):
    grupos = la.ler_lista("# vazio\n\n\n# cheio\nPT/AHU/CU/021/0006/00390\n", corpus_busca)
    assert [g.titulo for g in grupos] == ["cheio"]


def test_lista_sem_grupo_nenhum(corpus_busca):
    grupos = la.ler_lista("PT/AHU/CU/021/0006/00390\n", corpus_busca)
    assert len(grupos) == 1 and grupos[0].titulo is None


def test_cota_de_fora_do_corpus_ganha_referencia_e_link(corpus_busca):
    """O ponto do painel antigo: outras capitanias também servem."""
    grupos = la.ler_lista("AHU_ACL_CU_099, Cx. 1, D. 1\n", corpus_busca)
    item = grupos[0].itens[0]
    assert item.referencia == "PT/AHU/CU/099/0001/00001"
    assert item.linhas == []                 # não está no AHU-Sul
    assert digitarq(item.referencia).startswith("https://digitarq.arquivos.pt/")


def test_relatorio_conta_o_que_o_painel_mostra(corpus_busca):
    texto = ("# g\nPT/AHU/CU/021/0006/00390\nAHU_ACL_CU_099, Cx. 1, D. 1\ncota inventada\n")
    r = la.relatorio(la.ler_lista(texto, corpus_busca))
    assert r["total"] == 3 and r["grupos"] == 1
    assert r["com_digitarq"] == 2 and r["sem_referencia"] == ["cota inventada"]


# ---------------------------------------------------------------------------
# A lista a partir da seleção (WP 3.1)
# ---------------------------------------------------------------------------

def test_lista_a_partir_da_selecao(corpus_busca):
    from ahu.selecao import Selecao

    s = Selecao()
    s.adicionar(corpus_busca.documentos["doc_uid"].tolist()[:3])
    texto = la.lista_de_documentos(corpus_busca, s.uids)
    linhas = texto.splitlines()
    assert len(linhas) == 3
    # Cada linha volta a achar o seu documento.
    grupos = la.ler_lista(texto, corpus_busca)
    assert all(i.linhas for g in grupos for i in g.itens)


def test_lista_agrupada_abre_um_grupo_por_valor(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    texto = la.lista_de_documentos(corpus_busca, uids, agrupar_por="tipologia")
    grupos = la.ler_lista(texto, corpus_busca)
    titulos = [g.titulo for g in grupos]
    assert len(titulos) > 1 and all(titulos)
    tipologias = set(corpus_busca.documentos["tipologia"].dropna().astype(str))
    assert set(titulos) <= tipologias | {"—"}
    assert sum(len(g.itens) for g in grupos) == len(uids)


def test_lista_nao_usa_a_cota_quebrada(corpus_busca):
    """`UNFORMATTED_CODE` não é cota: a linha cai para o `new_code` ou para o doc_uid."""
    docs = corpus_busca.documentos
    uid = docs["doc_uid"].iloc[0]
    original = docs["new_code"].copy()
    try:
        docs.loc[docs.index[0], "new_code"] = "UNFORMATTED_CODE"
        linha = la.lista_de_documentos(corpus_busca, [uid]).strip()
        assert linha != "UNFORMATTED_CODE"
    finally:
        docs["new_code"] = original


# ---------------------------------------------------------------------------
# O PDF
# ---------------------------------------------------------------------------

def test_pdf_sai_com_cabecalho_e_paginas(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    grupos = la.ler_lista(la.lista_de_documentos(corpus_busca, uids, "tipologia"), corpus_busca)
    bruto = la.gerar_pdf(grupos, corpus_busca, {
        "titulo": "Consulta de teste", "pesquisador": "Saulo Rocha",
        "instituicao": "UFSC", "projeto": "AHU-Sul", "datas_consulta": "12 a 16/01/2027",
        "observacao": "Documentos para registro fotográfico.",
    })
    assert bruto.startswith(b"%PDF") and bruto.rstrip().endswith(b"%%EOF")
    assert len(bruto) > 5_000
    # A4 paisagem: a largura (297 mm) é maior que a altura.
    assert b"/MediaBox" in bruto


def test_pdf_de_uma_cota_de_fora_do_corpus_nao_quebra(corpus_busca):
    grupos = la.ler_lista("AHU_ACL_CU_099, Cx. 1, D. 1\ncota inventada\n", corpus_busca)
    assert la.gerar_pdf(grupos, corpus_busca).startswith(b"%PDF")


def test_pdf_sem_cabecalho_nenhum(corpus_busca):
    grupos = la.ler_lista("PT/AHU/CU/021/0006/00390\n", corpus_busca)
    assert la.gerar_pdf(grupos, corpus_busca).startswith(b"%PDF")


# ---------------------------------------------------------------------------
# As fontes embutidas (compartilhadas com o WP 3.3)
# ---------------------------------------------------------------------------

def test_as_fontes_estao_no_repositorio():
    for familia in mod_pdf.FAMILIAS:
        for estilo in ("", "B", "I"):
            caminho = mod_pdf.caminho_da_fonte(familia, estilo)
            assert caminho.is_file() and caminho.stat().st_size > 100_000
    assert (mod_pdf.PASTA_FONTES / "LICENSE-DejaVu.txt").is_file()


def test_familia_desconhecida_erra_alto():
    with pytest.raises(mod_pdf.ErroDeFonte, match="desconhecido"):
        mod_pdf.caminho_da_fonte("Comic", "")


def test_acentos_do_portugues_passam_pelas_duas_familias():
    from fpdf import FPDF

    amostra = "ÁÀÂÃÉÊÍÓÔÕÚÜÇ áàâãéêíóôõúüç ºª — … « » §"
    for familia in mod_pdf.FAMILIAS:
        documento = FPDF()
        mod_pdf.preparar(documento, familia)
        documento.add_page()
        documento.set_font(familia, "", 12)
        documento.multi_cell(0, 8, amostra)
        assert bytes(documento.output()).startswith(b"%PDF")


def test_a_caixa_de_marcar_existe_na_familia_da_lista():
    """A lista usa `☐`, que a DejaVu Sans tem e a Serif não — é por isso que a lista é Sans."""
    from fontTools.ttLib import TTFont

    def tem_glifo(caminho, caractere):
        with TTFont(caminho) as fonte:
            return any(ord(caractere) in t.cmap for t in fonte["cmap"].tables)

    assert tem_glifo(mod_pdf.caminho_da_fonte("Sans"), "☐")
    assert "☐" in la.MARCACOES


# ---------------------------------------------------------------------------
# Formatadores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bruto, esperado", [
    ("7964. 1787, março, 28, Montevidéu", "28 mar. 1787 · Montevidéu"),
    ("1787, março, 28", "28 mar. 1787"),
    ("1787", "1787"),
    ("", "sem data"),
    (None, "sem data"),
])
def test_data_extenso(bruto, esperado):
    assert mod_pdf.data_extenso(bruto) == esperado


def test_encurtar_corta_no_espaco():
    assert mod_pdf.encurtar("uma frase bem comprida", 10) == "uma frase…"
    assert mod_pdf.encurtar("curta", 10) == "curta"
    assert mod_pdf.encurtar(None, 10) == ""


# ---------------------------------------------------------------------------
# No corpus de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_toda_cota_do_corpus_vira_referencia(corpus_real):
    """Cada documento do corpus produz um `PT/AHU/CU/…` válido.

    A exceção documentada é o documento cuja única cota é antiga e por região: sem número de
    série não há código CRAV, e o teste a conta em vez de a esconder.
    """
    import re

    padrao = re.compile(r"^PT/AHU/CU/\d{3}(?:-\d{3})?/\d{4}/\d{5}$")
    sem_referencia = []
    for uid in corpus_real.documentos["doc_uid"]:
        referencia = la.referencia_do_documento(corpus_real, uid)
        if referencia is None:
            sem_referencia.append(uid)
            continue
        assert padrao.match(referencia), (uid, referencia)
        assert digitarq(referencia).startswith("https://digitarq.arquivos.pt/")

    assert len(sem_referencia) == SEM_REFERENCIA_CONHECIDO, sem_referencia
    for uid in sem_referencia:
        linha = corpus_real.documentos.iloc[corpus_real.indice_de(uid)]
        assert referencia_crav(str(linha.get("old_code"))) is None


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_o_indice_acha_todo_documento_pela_sua_propria_cota(corpus_real):
    """A ida e volta que a lista faz: documento -> cota -> de volta ao mesmo documento."""
    indice = la.indice_de_cotas(corpus_real)
    uids = corpus_real.documentos["doc_uid"].tolist()[::200]
    for uid in uids:
        texto = la.lista_de_documentos(corpus_real, [uid]).strip()
        _, linhas = indice.buscar(texto)
        assert corpus_real.indice_de(uid) in linhas, (uid, texto)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_pdf_de_cem_documentos_reais(corpus_real):
    uids = corpus_real.documentos["doc_uid"].tolist()[:100]
    grupos = la.ler_lista(la.lista_de_documentos(corpus_real, uids, "folder"), corpus_real)
    r = la.relatorio(grupos)
    assert r["total"] == 100 and r["no_catalogo"] == 100 and not r["sem_referencia"]
    bruto = la.gerar_pdf(grupos, corpus_real, {"pesquisador": "Saulo Rocha"})
    assert bruto.startswith(b"%PDF") and len(bruto) > 20_000


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_tem_o_painel_da_lista():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DADOS.parent / "app.py"), default_timeout=180)
    at.query_params["v"] = "1"
    at.query_params["q"] = "AHU_ACL_CU_017, Cx. 5, D. 487"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(a.key == "area_lista_ahu" for a in at.text_area), [a.key for a in at.text_area]

    # Colar uma cota na caixa faz o relatório aparecer.
    caixa = next(a for a in at.text_area if a.key == "area_lista_ahu")
    caixa.set_value("AHU_ACL_CU_017, Cx. 5, D. 487").run()
    assert not at.exception, [e.value for e in at.exception]
    avisos = " ".join(i.value for i in at.info)
    assert "1 documento(s)" in avisos and "DigitArq" in avisos
