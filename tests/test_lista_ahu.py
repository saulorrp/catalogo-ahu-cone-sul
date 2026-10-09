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
# O PDF (R3): uma folha para levar ao arquivo
# ---------------------------------------------------------------------------

def texto_do_pdf(bruto: bytes) -> str:
    import io as _io

    import pypdf

    return "\n".join(p.extract_text() for p in pypdf.PdfReader(_io.BytesIO(bruto)).pages)


def paginas(bruto: bytes) -> int:
    import io as _io

    import pypdf

    return len(pypdf.PdfReader(_io.BytesIO(bruto)).pages)


def _sem_espacos(texto: str) -> str:
    return "".join(texto.split())


@pytest.fixture
def pdf_do_corpus(corpus_busca):
    grupos = la.ler_lista(la.lista_de_documentos(
        corpus_busca, corpus_busca.documentos["doc_uid"].tolist()), corpus_busca)
    return la.gerar_pdf(grupos, corpus_busca)


def test_o_pdf_abre_e_fecha_bem(pdf_do_corpus):
    assert pdf_do_corpus.startswith(b"%PDF") and pdf_do_corpus.rstrip().endswith(b"%%EOF")


def test_a4_paisagem(pdf_do_corpus):
    """R3: A4 paisagem — 297 × 210 mm, ou seja 841,89 × 595,28 pontos."""
    import io as _io

    import pypdf

    caixa = pypdf.PdfReader(_io.BytesIO(pdf_do_corpus)).pages[0].mediabox
    largura, altura = float(caixa.width), float(caixa.height)
    assert largura > altura
    assert round(largura) == 842 and round(altura) == 595


def test_o_alto_da_pagina_tem_so_o_titulo_e_a_data(pdf_do_corpus):
    from datetime import date

    texto = texto_do_pdf(pdf_do_corpus)
    primeiras = [ln.strip() for ln in texto.splitlines() if ln.strip()][:2]
    assert primeiras[0] == la.TITULO
    assert primeiras[1] == date.today().strftime("%d/%m/%Y")


def test_as_colunas_sao_as_seis_do_plano(pdf_do_corpus):
    texto = _sem_espacos(texto_do_pdf(pdf_do_corpus))
    assert "NºAnoCotaResumoDigitArq" in texto


def test_cada_documento_tem_numero_ano_cota_e_resumo(corpus_busca, pdf_do_corpus):
    texto = _sem_espacos(texto_do_pdf(pdf_do_corpus))
    docs = corpus_busca.documentos
    for numero, (_, linha) in enumerate(docs.iterrows(), start=1):
        assert str(numero) in texto
        assert _sem_espacos(str(linha["description"])[:40]) in texto, numero
        assert _sem_espacos(str(int(linha["ano_isolado"]))) in texto


def test_o_resumo_sai_inteiro_e_nao_cortado(corpus_busca):
    """R3: "Resumo: full, wrapped" — nada de reticências."""
    docs = corpus_busca.documentos
    original = docs["description"].copy()
    try:
        comprido = "palavra " * 90
        docs.loc[docs.index[0], "description"] = comprido.strip()
        uid = docs["doc_uid"].iloc[0]
        grupos = la.ler_lista(la.lista_de_documentos(corpus_busca, [uid]), corpus_busca)
        texto = _sem_espacos(texto_do_pdf(la.gerar_pdf(grupos, corpus_busca)))
        assert "…" not in texto
        assert texto.count("palavra") == 90
    finally:
        docs["description"] = original


def test_a_cota_traz_a_referencia_abaixo(corpus_busca):
    grupos = la.ler_lista("AHU_ACL_CU_021, Cx. 6, D. 390.", corpus_busca)
    texto = _sem_espacos(texto_do_pdf(la.gerar_pdf(grupos, corpus_busca)))
    assert "AHU_ACL_CU_021,Cx.6,D.390" in texto      # a cota do Resgate, sem o ponto final
    assert "PT/AHU/CU/021/0006/00390" in texto       # o código de referência


def test_o_digitarq_e_um_rotulo_clicavel_e_nao_a_url(pdf_do_corpus):
    texto = texto_do_pdf(pdf_do_corpus)
    assert "abrir" in texto
    assert "https://digitarq" not in texto          # a URL crua não vai impressa
    # mas o link está lá, como anotação do PDF
    assert b"digitarq.arquivos.pt" in pdf_do_corpus


def test_o_rodape_e_a_linha_do_plano(pdf_do_corpus):
    texto = _sem_espacos(texto_do_pdf(pdf_do_corpus))
    assert _sem_espacos(la.RODAPE) in texto
    assert "página1/" in texto


@pytest.mark.parametrize("proibido", [
    "scribe_mediation", "doc_uid", "proveniência", "SRSP", "Tipologia", "Remetente",
    "Pesquisador", "Projeto", "Instituição",
])
def test_o_pdf_nao_traz_o_que_o_r3_tirou(pdf_do_corpus, proibido):
    assert proibido.lower() not in texto_do_pdf(pdf_do_corpus).lower(), proibido


def test_grupos_viram_faixas_de_titulo(corpus_busca):
    texto = ("# Dia 1\n"
             "PT/AHU/CU/021/0006/00390\n"
             "# Dia 2\n"
             "AHU_ACL_CU_014, Cx. 13, D. 1063\n")
    bruto = la.gerar_pdf(la.ler_lista(texto, corpus_busca), corpus_busca)
    lido = _sem_espacos(texto_do_pdf(bruto))
    assert "Dia1(1)" in lido and "Dia2(1)" in lido


def test_a_numeracao_e_continua_entre_grupos(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    texto = la.lista_de_documentos(corpus_busca, uids, agrupar_por="tipologia")
    bruto = la.gerar_pdf(la.ler_lista(texto, corpus_busca), corpus_busca)
    lido = _sem_espacos(texto_do_pdf(bruto))
    assert str(len(uids)) in lido


def test_pdf_de_uma_cota_de_fora_do_corpus_nao_quebra(corpus_busca):
    grupos = la.ler_lista("AHU_ACL_CU_099, Cx. 1, D. 1\ncota inventada\n", corpus_busca)
    texto = texto_do_pdf(la.gerar_pdf(grupos, corpus_busca))
    assert "não consta no catálogo AHU-Sul" in texto


def test_pdf_a_partir_da_selecao(corpus_busca):
    from ahu.selecao import Selecao

    s = Selecao()
    s.adicionar(corpus_busca.documentos["doc_uid"].tolist()[:3])
    bruto = la.pdf_da_selecao(corpus_busca, s)
    texto = _sem_espacos(texto_do_pdf(bruto))
    assert _sem_espacos(la.TITULO) in texto
    for numero in (1, 2, 3):
        assert str(numero) in texto


def test_pdf_de_selecao_vazia_sai_so_com_o_cabecalho(corpus_busca):
    from ahu.selecao import Selecao

    bruto = la.pdf_da_selecao(corpus_busca, Selecao())
    assert bruto.startswith(b"%PDF")
    assert la.TITULO in texto_do_pdf(bruto)


# ---------------------------------------------------------------------------
# A fonte embutida (R3)
# ---------------------------------------------------------------------------

def test_a_fonte_esta_no_repositorio_com_a_licenca():
    for estilo in ("", "B", "I"):
        caminho = mod_pdf.caminho_da_fonte(estilo)
        assert caminho.is_file() and caminho.stat().st_size > 100_000
        assert caminho.name.startswith("NotoSerif")
    assert (mod_pdf.PASTA_FONTES / "LICENSE-NotoSerif-OFL.txt").is_file()


def test_as_dejavu_sairam_com_o_dossie():
    """D7: a DejaVu servia ao dossiê, que saiu. Uma família só no repositório."""
    assert not list(mod_pdf.PASTA_FONTES.glob("DejaVu*"))
    assert sorted(p.name for p in mod_pdf.PASTA_FONTES.glob("*.ttf")) == [
        "NotoSerif-Bold.ttf", "NotoSerif-Italic.ttf", "NotoSerif-Regular.ttf"]


def test_estilo_desconhecido_erra_alto():
    with pytest.raises(mod_pdf.ErroDeFonte, match="Estilo desconhecido"):
        mod_pdf.caminho_da_fonte("Negrito")


def test_acentos_do_portugues_e_o_capitulum_passam():
    from fontTools.ttLib import TTFont

    with TTFont(mod_pdf.caminho_da_fonte()) as fonte:
        cmap = set()
        for tabela in fonte["cmap"].tables:
            cmap |= set(tabela.cmap)
    for caractere in "ÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüçºª—…«»":
        assert ord(caractere) in cmap, caractere
    assert 0x2E3F in cmap, "o ⸿ da identidade do EINS"


def test_a_caixa_de_marcar_e_desenhada_porque_a_fonte_nao_a_tem():
    """R3: a Noto Serif não traz glifo de caixa; por isso a caixa é um retângulo."""
    from fontTools.ttLib import TTFont

    with TTFont(mod_pdf.caminho_da_fonte()) as fonte:
        cmap = set()
        for tabela in fonte["cmap"].tables:
            cmap |= set(tabela.cmap)
    assert 0x2610 not in cmap and 0x25A1 not in cmap
    assert hasattr(mod_pdf, "caixa_de_marcar")


def test_o_pdf_nao_escreve_glifo_de_caixa(corpus_busca):
    grupos = la.ler_lista("PT/AHU/CU/021/0006/00390", corpus_busca)
    assert "☐".encode() not in la.gerar_pdf(grupos, corpus_busca)


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
    at.query_params["v"] = "2"
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
