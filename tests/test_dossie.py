"""Testes do WP 3.3: o dossiê em PDF (spec §8.2, D1, A8).

O A8 tem três partes, e cada uma tem o seu teste aqui:

1. **capa com proveniência** — `test_a8_capa_tem_proveniencia_e_citacao`;
2. **uma seção por documento** — `test_a8_uma_secao_por_documento`;
3. **`scribe_mediation_likely` com ressalva, e a interface sem ele** —
   `test_a8_mediacao_sai_como_rotulo_com_ressalva` e
   `test_d1_o_campo_bruto_nao_aparece_no_pdf`, mais o teste do WP 2.3 que varre a interface.

O texto do PDF é lido de volta com o `pypdf` (dependência só de teste).
"""
import pytest

from ahu import dossie as dos
from ahu import exportar as mod_exportar
from ahu import ficha as mod_ficha
from ahu.consulta import Consulta
from ahu.dados import carregar_sem_cache
from ahu.dossie import ErroDeDossie
from ahu.filtros import Filtros
from tests.conftest import APP_DADOS, tem_artefatos


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


def texto_do_pdf(bruto: bytes) -> str:
    import io

    import pypdf

    leitor = pypdf.PdfReader(io.BytesIO(bruto))
    return "\n".join(p.extract_text() for p in leitor.pages)


def paginas(bruto: bytes) -> int:
    import io

    import pypdf

    return len(pypdf.PdfReader(io.BytesIO(bruto)).pages)


def _sem_espacos(texto: str) -> str:
    """O extrator de texto do pypdf espalha espaços: compara-se sem eles."""
    return "".join(texto.split())


# ---------------------------------------------------------------------------
# Montagem das seções
# ---------------------------------------------------------------------------

def test_secao_por_documento_na_ordem_pedida(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:4]
    secoes = dos.montar_secoes(corpus_busca, list(reversed(uids)))
    assert [s.doc_uid for s in secoes] == list(reversed(uids))
    assert all(s.titulo and s.cotas for s in secoes)


def test_dossie_vazio_erra_alto(corpus_busca):
    with pytest.raises(ErroDeDossie, match="ao menos um documento"):
        dos.montar_secoes(corpus_busca, [])


def test_teto_de_200_erra_alto_com_mensagem_util(corpus_busca):
    with pytest.raises(ErroDeDossie, match="cabe 200"):
        dos.montar_secoes(corpus_busca, [f"u{i}" for i in range(dos.LIMITE + 1)])


def test_no_teto_exato_ainda_passa(corpus_busca):
    """O erro é acima de 200, não em 200."""
    uids = corpus_busca.documentos["doc_uid"].tolist()
    repetidos = (uids * 40)[:dos.LIMITE]
    assert len(dos.montar_secoes(corpus_busca, repetidos)) == dos.LIMITE


# ---------------------------------------------------------------------------
# A8 — a capa
# ---------------------------------------------------------------------------

def test_a8_capa_tem_proveniencia_e_citacao(corpus_busca):
    consulta = Consulta(texto="deserção de soldados", modo="conceitual", rigor="alto",
                        lente="sintaxe_diacronica", filtros=Filtros(tipologias=["CARTA"]),
                        semente=42)
    uids = corpus_busca.documentos["doc_uid"].tolist()[:3]
    prov = mod_exportar.proveniencia(corpus_busca, consulta, len(uids), origem="dossie")
    texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, uids, prov=prov))
    comprimido = _sem_espacos(texto)

    assert "deserção de soldados" in texto                 # a consulta
    assert "conceitual" in texto and "alto" in texto        # modo e rigor
    assert "sintaxe_diacronica" in texto                    # a lente
    assert "tipologias" in comprimido                       # os filtros
    assert "42" in texto                                    # a semente
    assert "Documentosnestedossiê:3" in comprimido          # o nº de resultados
    assert prov["gerado_em"][:10] in texto                  # a data
    assert prov["app_versao"]["versao"] in texto            # a versão do app
    assert prov["corpus"]["manifesto_sha256"] in comprimido  # o corpus
    assert "Comocitar" in comprimido
    assert mod_exportar.DOI_PREPRINT in comprimido          # o DOI


def test_capa_sem_consulta_nao_quebra(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()[:2]
    texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, uids))
    assert "sem texto" in texto or "corpus com os filtros" in texto


# ---------------------------------------------------------------------------
# A8 — uma seção por documento
# ---------------------------------------------------------------------------

def test_a8_uma_secao_por_documento(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, uids))
    for numero, uid in enumerate(uids, start=1):
        ficha = mod_ficha.montar(corpus_busca, uid)
        # O número da seção e a primeira parte do título de cada ficha estão no PDF.
        assert f"{numero}." in texto, numero
        pedaco = ficha.resumo[:30]
        assert _sem_espacos(pedaco) in _sem_espacos(texto), uid


def test_secao_traz_cotas_resumo_e_links(corpus_busca):
    uid = corpus_busca.documentos["doc_uid"].iloc[4]      # o que tem cota e link no conftest
    texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, [uid]))
    comprimido = _sem_espacos(texto)
    ficha = mod_ficha.montar(corpus_busca, uid)
    for valor in ficha.cotas.values():
        assert _sem_espacos(str(valor)) in comprimido
    assert "DigitArq" in texto and "Projeto Resgate" in texto


def test_o_pdf_tem_mais_de_uma_pagina_e_fecha_bem(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    bruto = dos.gerar_pdf(corpus_busca, uids)
    assert bruto.startswith(b"%PDF") and bruto.rstrip().endswith(b"%%EOF")
    assert paginas(bruto) >= 2                            # capa + documentos


# ---------------------------------------------------------------------------
# A8 / D1 — a mediação de escrivão
# ---------------------------------------------------------------------------

def test_mediacao_vira_rotulo_em_portugues(corpus_busca):
    """D1/§8.2: "Provável"/"Pouco provável", nunca true/false."""
    extra = corpus_busca.pdf_extra
    original = extra["scribe_mediation_likely"].copy()
    try:
        extra.loc[extra.index[0], "scribe_mediation_likely"] = True
        extra.loc[extra.index[1], "scribe_mediation_likely"] = False
        secoes = dos.montar_secoes(corpus_busca, extra["doc_uid"].tolist()[:2])
        assert [s.mediacao for s in secoes] == ["Provável", "Pouco provável"]
    finally:
        extra["scribe_mediation_likely"] = original


def test_a8_mediacao_sai_como_rotulo_com_ressalva(corpus_busca):
    extra = corpus_busca.pdf_extra
    original = extra["scribe_mediation_likely"].copy()
    try:
        extra.loc[extra.index[0], "scribe_mediation_likely"] = True
        texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, [extra["doc_uid"].iloc[0]]))
        comprimido = _sem_espacos(texto)
        assert _sem_espacos(dos.TITULO_MEDIACAO) in comprimido
        assert "Provável" in texto
        # A ressalva está no PDF, e nomeia o que importa: inferência, preprint, não é prova.
        assert "Ressalvametodológica" in comprimido
        assert "preprint" in texto
        assert _sem_espacos("não um fato estabelecido") in comprimido
        assert _sem_espacos("nunca como prova") in comprimido
        assert "1,13" in comprimido and "705" in comprimido   # a auditoria do §2.1
    finally:
        extra["scribe_mediation_likely"] = original


def test_d1_o_campo_bruto_nao_aparece_no_pdf(corpus_busca):
    """O rótulo aparece; o **nome do campo** e os booleanos não."""
    uids = corpus_busca.documentos["doc_uid"].tolist()[:5]
    texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, uids))
    comprimido = _sem_espacos(texto)
    for campo in mod_ficha.CAMPOS_PROIBIDOS:
        assert campo not in comprimido, campo
    assert "True" not in comprimido and "False" not in comprimido


def test_documento_sem_dado_de_mediacao_nao_ganha_rotulo(corpus_busca):
    extra = corpus_busca.pdf_extra
    uid = extra["doc_uid"].iloc[0]
    original = extra["scribe_mediation_likely"].copy()
    try:
        extra["scribe_mediation_likely"] = extra["scribe_mediation_likely"].astype("object")
        extra.loc[extra.index[0], "scribe_mediation_likely"] = None
        assert dos.montar_secoes(corpus_busca, [uid])[0].mediacao is None
    finally:
        extra["scribe_mediation_likely"] = original


def test_a1_nenhum_nivel_proibido_no_dossie(corpus_busca):
    uids = corpus_busca.documentos["doc_uid"].tolist()
    texto = _sem_espacos(texto_do_pdf(dos.gerar_pdf(corpus_busca, uids)))
    for nivel in mod_ficha.NIVEIS_PROIBIDOS:
        assert nivel not in texto, nivel


# ---------------------------------------------------------------------------
# Acentos: a fonte serifada embutida
# ---------------------------------------------------------------------------

def test_acentos_do_portugues_saem_certos_no_dossie(corpus_busca):
    docs = corpus_busca.documentos
    uid = docs["doc_uid"].iloc[0]
    original = docs["description"].copy()
    try:
        docs.loc[docs.index[0], "description"] = (
            "REQUERIMENTO de João Gonçalves à Mesa de Inspeção: ÁÀÂÃÉÊÍÓÔÕÚÜÇ — «aspas» º ª")
        texto = texto_do_pdf(dos.gerar_pdf(corpus_busca, [uid]))
        assert "João Gonçalves" in texto and "Inspeção" in texto
        assert "ÁÀÂÃÉÊÍÓÔÕÚÜÇ" in _sem_espacos(texto)
    finally:
        docs["description"] = original


# ---------------------------------------------------------------------------
# No corpus de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_dossie_de_duzentos_documentos_reais(corpus_real):
    """O teto, nos dados de verdade: ele sai, e sai com uma seção por documento."""
    uids = corpus_real.documentos["doc_uid"].tolist()[:dos.LIMITE]
    bruto = dos.gerar_pdf(corpus_real, uids, Consulta(texto="deserção", modo="exata"))
    assert bruto.startswith(b"%PDF")
    assert paginas(bruto) > 20
    texto = _sem_espacos(texto_do_pdf(bruto))
    assert f"{dos.LIMITE}." in texto                       # a última seção foi numerada
    for campo in mod_ficha.CAMPOS_PROIBIDOS:
        assert campo not in texto


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_os_dois_rotulos_de_mediacao_existem_no_corpus(corpus_real):
    """O corpus tem os dois casos, e o dossiê mostra cada um com o seu rótulo."""
    extra = corpus_real.pdf_extra
    com = extra.loc[extra["scribe_mediation_likely"], "doc_uid"].tolist()[:2]
    sem = extra.loc[~extra["scribe_mediation_likely"], "doc_uid"].tolist()[:2]
    assert com and sem
    secoes = dos.montar_secoes(corpus_real, com + sem)
    assert [s.mediacao for s in secoes] == ["Provável", "Provável",
                                            "Pouco provável", "Pouco provável"]
