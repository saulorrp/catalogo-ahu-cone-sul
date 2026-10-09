"""Testes do R1 (plano v3.1): o que foi retirado tem de ficar retirado.

Um teste de remoção parece bobo até alguém reintroduzir o que saiu. Estes cobram as decisões
**D7** (exportação é uma coisa só: o PDF da lista do AHU), **D8** (a mediação de escrivão sai do
documento), **D9** (as lentes saíram) e **D10** (nenhum identificador interno na interface).
"""
import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
APP = RAIZ / "app.py"


@pytest.fixture(scope="module")
def fonte_do_app() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def textos_da_interface(fonte_do_app) -> list[str]:
    """Todas as cadeias literais do `app.py` — é o que pode chegar à tela."""
    arvore = ast.parse(fonte_do_app)
    return [n.value for n in ast.walk(arvore)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


# ---------------------------------------------------------------------------
# D7: os módulos de exportação saíram
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("caminho", [
    "ahu/exportar.py", "ahu/dossie.py", "tests/test_exportar.py", "tests/test_dossie.py",
])
def test_os_arquivos_de_exportacao_e_dossie_nao_existem(caminho):
    assert not (RAIZ / caminho).exists(), caminho


def test_o_app_nao_importa_exportacao_nem_dossie(fonte_do_app):
    for proibido in ("import exportar", "import dossie", "mod_exportar", "mod_dossie"):
        assert proibido not in fonte_do_app, proibido


def test_a_proveniencia_sobreviveu_para_o_plano_do_eins():
    """D11: o plano `eins-plano-coleta/1` precisa dizer de que corpus saiu."""
    from ahu import proveniencia

    assert hasattr(proveniencia, "proveniencia") and hasattr(proveniencia, "camadas_ativas")
    assert proveniencia.DOI_PREPRINT == "10.1590/SciELOPreprints.16461"


def test_a_selecao_nao_e_mais_baixada_nem_carregada(fonte_do_app):
    """D7: o JSON da seleção saiu da interface."""
    assert "file_uploader" not in fonte_do_app
    assert "selecao_ahu_sul.json" not in fonte_do_app


def test_os_unicos_downloads_de_pdf_sao_a_lista_do_ahu(textos_da_interface):
    """Seleção, cotas coladas e lotes produzem o mesmo tipo de PDF, e só ele.

    Os nomes montados por f-string chegam aqui em pedaços (`"…_lote_"` e `".pdf"`), e por isso
    o teste olha o prefixo em vez do nome inteiro.
    """
    arquivos = sorted({t for t in textos_da_interface
                       if t.endswith((".pdf", ".csv")) or "lista_consulta_AHU" in t})
    assert arquivos == [".pdf", "lista_consulta_AHU.pdf",
                        "lista_consulta_AHU_lote_"], arquivos
    assert not any(t.endswith(".csv") for t in textos_da_interface)


def test_o_unico_json_oferecido_e_o_plano_do_eins(textos_da_interface):
    arquivos = [t for t in textos_da_interface if t.endswith(".json")]
    assert arquivos == ["plano_coleta.eins.json"], arquivos


# ---------------------------------------------------------------------------
# D8: a mediação de escrivão sai do documento
# ---------------------------------------------------------------------------

def test_o_campo_do_escrivao_nao_aparece_em_lugar_nenhum_do_codigo_do_app(fonte_do_app):
    assert "scribe_mediation" not in fonte_do_app


def test_o_campo_do_escrivao_so_aparece_como_proibicao():
    """O campo continua no dado (`pdf_extra.parquet`), e em `ahu/` só existe como proibição.

    `dados.py` e `ficha.py` o nomeiam para **barrá-lo** (`CAMPOS_SO_PDF`, `CAMPOS_PROIBIDOS`).
    Qualquer outro módulo que o mencione estaria a caminho de exibi-lo.
    """
    from ahu.dados import CAMPOS_SO_PDF
    from ahu.ficha import CAMPOS_PROIBIDOS

    assert CAMPOS_SO_PDF == CAMPOS_PROIBIDOS == ("scribe_mediation_likely",)
    guardas = {"dados.py", "ficha.py"}
    for modulo in (RAIZ / "ahu").glob("*.py"):
        if modulo.name in guardas:
            continue
        texto = modulo.read_text(encoding="utf-8")
        assert "scribe_mediation" not in texto, modulo.name


# ---------------------------------------------------------------------------
# D9: as lentes saíram
# ---------------------------------------------------------------------------

def test_nao_existe_mais_lente_no_codigo():
    from ahu import consulta, filtros

    assert not hasattr(filtros, "LENTES") and not hasattr(filtros, "aplicar_lente")
    assert not hasattr(consulta.Consulta, "com_lente")
    assert consulta.VERSAO_URL == 2


def test_a_interface_nao_fala_mais_de_lentes(textos_da_interface):
    for texto in textos_da_interface:
        assert "lente" not in texto.lower(), texto


# ---------------------------------------------------------------------------
# D10: nenhum identificador interno na interface
# ---------------------------------------------------------------------------

# Um literal que **é** o nome da coluna é chave de dicionário, não rótulo. O que a D10 proíbe é
# o identificador dentro de uma frase, de um rótulo ou de um texto de ajuda.
CHAVES = {"doc_uid", "pessoa_uid", "doc_uids", "pessoa_uids"}


def test_nenhum_rotulo_da_interface_nomeia_um_identificador(textos_da_interface):
    """D10: o pesquisador vê cota, ano, resumo e nomes — nunca um uid."""
    for texto in textos_da_interface:
        if texto in CHAVES:
            continue
        baixo = texto.lower()
        assert "doc_uid" not in baixo, texto
        assert "pessoa_uid" not in baixo, texto
        assert "id público" not in baixo, texto


def test_as_tabelas_mostradas_nao_tem_coluna_de_identificador(corpus_busca):
    from ahu import amostras, pessoas, selecao

    uids = corpus_busca.documentos["doc_uid"].tolist()[:3]

    s = selecao.Selecao()
    s.adicionar(uids)
    tabelas = [
        s.tabela(corpus_busca),
        pessoas.tabela(pessoas.listar(corpus_busca)),
        pessoas.tabela_de_coletivos(pessoas.listar_coletivos(corpus_busca)),
        amostras.tabela_dos_documentos(corpus_busca, uids),
    ]
    for tabela in tabelas:
        colunas = [str(c).lower() for c in tabela.columns]
        assert not any("uid" in c for c in colunas), colunas
        # E nenhum valor de célula é um identificador do corpus.
        valores = "\n".join(tabela.astype(str).to_numpy().ravel().tolist())
        assert "ahu-sul:" not in valores and "pes:" not in valores


def test_a_issue_do_github_ainda_leva_o_uid(corpus_busca):
    """D10 tem uma exceção explícita: a *issue* vai para o mantenedor, não para o usuário."""
    from ahu import pessoas

    pessoa = pessoas.encontrar(corpus_busca, "Lavradio")[0]
    pagina = pessoas.pagina(corpus_busca, pessoa.pessoa_uid)
    from urllib.parse import unquote_plus
    assert pessoa.pessoa_uid in unquote_plus(pagina.url_issue)
