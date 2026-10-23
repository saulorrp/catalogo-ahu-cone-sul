"""Testes do WP 2.3: ficha, links e a estrutura do app (A1).

O A1 é verificado percorrendo **todas** as fichas montadas e procurando campo e nível proibidos.
É por isso que `ahu/ficha.py` monta a ficha como dado, sem Streamlit: o teste varre o corpus
inteiro em segundos, sem subir interface.
"""
import ast
import re
from pathlib import Path

import pytest

from ahu import ficha as mod_ficha
from ahu import links
from ahu.dados import carregar_sem_cache
from ahu.ficha import CAMPOS_PROIBIDOS, NIVEIS_PROIBIDOS, marcar_termos, montar
from tests.conftest import APP_DADOS, tem_artefatos

RAIZ = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Links (§9)
# ---------------------------------------------------------------------------

def test_link_do_digitarq_usa_o_codigo_de_referencia():
    url = links.digitarq("PT/AHU/CU/021/0006/00390")
    assert url == ("https://digitarq.arquivos.pt/search?"
                   "query=PT%2FAHU%2FCU%2F021%2F0006%2F00390&isAdvancedSearch=false")


def test_sem_cota_nao_ha_link_do_digitarq():
    assert links.digitarq(None) is None
    assert links.digitarq("") is None
    assert links.digitarq("UNFORMATTED_CODE") is None   # a cota quebrada do catálogo


def test_link_do_resgate_e_o_portal():
    assert links.resgate("AHU_ACL_CU_021, Cx. 6, D. 390.") == "https://resgate.bn.gov.br/"


# ---------------------------------------------------------------------------
# Destaque dos termos (§5.5, §9)
# ---------------------------------------------------------------------------

def test_marcar_termos_destaca_a_palavra_do_texto_original():
    assert marcar_termos("CARTA sobre soldados", ["soldados"]) == "CARTA sobre **soldados**"


def test_marcar_termos_casa_grafia_antiga():
    """Quem buscou 'Melo' vê 'Mello' marcado (§5.1)."""
    assert marcar_termos("pedido de Manoel de Mello", ["melo"]) == "pedido de Manoel de **Mello**"


def test_marcar_termos_sem_termos_devolve_o_texto():
    assert marcar_termos("CARTA", []) == "CARTA"
    assert marcar_termos("", ["x"]) == ""


# ---------------------------------------------------------------------------
# Ficha (§6.2, §6.3, §9)
# ---------------------------------------------------------------------------

def test_ficha_tem_titulo_com_ano_tipologia_pessoa_e_pasta(corpus_busca):
    uid = corpus_busca.documentos["doc_uid"].iloc[0]
    f = montar(corpus_busca, uid)
    linha = corpus_busca.documentos.iloc[0]
    assert str(int(linha["ano_isolado"])) in f.titulo
    assert str(linha["tipologia"]) in f.titulo
    assert str(linha["folder"]) in f.titulo


def test_ficha_traz_cotas_e_links(corpus_busca):
    uid = corpus_busca.documentos["doc_uid"].iloc[4]   # o documento com cota conhecida
    f = montar(corpus_busca, uid)
    assert f.cotas["Código de referência"] == "PT/AHU/CU/021/0006/00390"
    assert f.cotas["Código atual"].startswith("AHU_ACL_CU_021")
    assert "digitarq" in f.links["digitarq"]
    assert f.links["resgate"].startswith("https://resgate.bn.gov.br")


def test_ficha_separa_as_inferencias_de_ia(corpus_busca):
    f = montar(corpus_busca, corpus_busca.documentos["doc_uid"].iloc[0])
    assert "SRSP" in f.inferencias
    assert "Categoria social do remetente" in f.inferencias
    # O resumo do AHU não é inferência: fica fora do bloco.
    assert f.resumo not in f.inferencias.values()


def test_ficha_marca_tipologia_a_revisar(corpus_busca):
    fichas = [montar(corpus_busca, u) for u in corpus_busca.documentos["doc_uid"]]
    assert all(isinstance(f.tipologia_a_revisar, bool) for f in fichas)


def test_a1_nenhuma_ficha_mostra_campo_ou_nivel_proibido(corpus_busca):
    """A1: nem `scribe_mediation_likely`, nem BAIXA/REJEITADO, em nenhuma ficha."""
    for uid in corpus_busca.documentos["doc_uid"]:
        texto = montar(corpus_busca, uid).texto_visivel()
        for campo in CAMPOS_PROIBIDOS:
            assert campo not in texto, (uid, campo)
        for nivel in NIVEIS_PROIBIDOS:
            assert not re.search(rf"\b{nivel}\b", texto), (uid, nivel)


def test_a1_pessoa_sem_nascimento_exibivel_aparece_com_rotulo(corpus_busca):
    """D2: a pessoa continua na ficha; o que não aparece é o nascimento."""
    escondidas = corpus_busca.pessoas[~corpus_busca.pessoas["nascimento_exibivel"].astype(bool)]
    if escondidas.empty:
        pytest.skip("este corpus não tem biografia escondida")
    uid = escondidas["pessoa_uid"].iloc[0]
    vinculos = corpus_busca.doc_pessoa
    docs = vinculos.loc[vinculos["pessoa_uid"] == uid, "doc_uid"]
    if docs.empty:
        pytest.skip("a pessoa escondida não tem documento neste corpus")
    f = montar(corpus_busca, docs.iloc[0])
    p = next(p for p in f.pessoas if p.pessoa_uid == uid)
    assert p.nivel is None and p.nascimento is None and p.fonte is None
    assert p.rotulo    # §6.3: diz o que se sabe, sem dizer "Desconhecida"
    assert "esconhecida" not in p.rotulo


def test_ficha_de_documento_sem_pessoa_funciona(corpus_busca):
    sem_pessoa = set(corpus_busca.documentos["doc_uid"]) - set(corpus_busca.doc_pessoa["doc_uid"])
    uid = sem_pessoa.pop() if sem_pessoa else corpus_busca.documentos["doc_uid"].iloc[0]
    f = montar(corpus_busca, uid)
    assert f.titulo and isinstance(f.pessoas, list)


def test_ficha_de_doc_uid_inexistente_avisa(corpus_busca):
    with pytest.raises(KeyError, match="doc_uid desconhecido"):
        montar(corpus_busca, "t:naoexiste")


# ---------------------------------------------------------------------------
# A estrutura do app (sem subir Streamlit)
# ---------------------------------------------------------------------------

def test_app_py_existe_e_e_python_valido():
    arvore = ast.parse((RAIZ / "app.py").read_text(encoding="utf-8"))
    nomes = {n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}
    assert {"main", "aba_busca", "aba_sobre", "desenhar_ficha", "barra_lateral"} <= nomes


def test_app_tem_as_abas_da_spec_e_o_conversor():
    import app

    assert app.SECOES == ("Catálogo", "AHU-Sul enriquecido")
    assert app.ABAS_CATALOGO == ("Busca", "Conversor", "Referências", "Sobre e método")
    assert app.ABAS_ENRIQUECIDAS == ("Busca enriquecida", "Pessoas", "Amostras", "Painel")


def test_app_nao_tem_logica_de_negocio():
    """AGENTS.md: a lógica fica em `ahu/`. O app.py só desenha."""
    texto = (RAIZ / "app.py").read_text(encoding="utf-8")
    for proibido in ("BM25", "get_scores", "read_parquet", "pickle", "rank_bm25",
                     "SentenceTransformer", "np."):
        assert proibido not in texto, proibido
    assert "from ahu" in texto


def test_app_so_le_app_dados():
    """AGENTS.md 3: o app não abre `corpus/` nem monta caminho de arquivo."""
    texto = (RAIZ / "app.py").read_text(encoding="utf-8")
    for proibido in ("corpus/", "Path(", "open("):
        assert proibido not in texto, proibido


def test_app_nunca_nomeia_o_campo_do_pdf():
    """D1: `scribe_mediation_likely` não aparece em lugar nenhum da interface."""
    assert "scribe_mediation_likely" not in (RAIZ / "app.py").read_text(encoding="utf-8")


def test_app_v2_preservado_no_legado():
    antigo = RAIZ / "legado" / "app_v2.py"
    assert antigo.is_file() and antigo.stat().st_size > 10_000
    ast.parse(antigo.read_text(encoding="utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# A1 sobre o corpus de verdade: todas as 167.344 fichas
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a1_em_todas_as_fichas_reais():
    dados = carregar_sem_cache(APP_DADOS)
    niveis_vistos = set()
    com_pessoa = 0
    for uid in dados.documentos["doc_uid"]:
        f = montar(dados, uid)
        texto = f.texto_visivel()
        for campo in CAMPOS_PROIBIDOS:
            assert campo not in texto, (uid, campo)
        for nivel in NIVEIS_PROIBIDOS:
            assert not re.search(rf"\b{nivel}\b", texto), (uid, nivel)
        for p in f.pessoas:
            if p.nivel:
                niveis_vistos.add(p.nivel)
            if p.pessoa_uid:
                com_pessoa += 1
    assert niveis_vistos <= {"ALTA", "MEDIA"}
    assert niveis_vistos == {"ALTA", "MEDIA"}   # as duas aparecem de fato
    assert com_pessoa > 5000


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_fichas_reais_tem_titulo_e_cota():
    dados = carregar_sem_cache(APP_DADOS)
    sem_titulo, sem_cota, sem_link, sem_link_brasil = 0, 0, 0, 0
    for uid, pasta in zip(dados.documentos["doc_uid"], dados.documentos["folder"]):
        f = montar(dados, uid)
        sem_titulo += not f.titulo
        sem_cota += not f.cotas
        if f.links.get("digitarq") is None:
            if uid.startswith("ahu-sul:"):
                sem_link += 1
            elif pasta not in ("Reino", "Códices"):
                sem_link_brasil += 1
    assert sem_titulo == 0 and sem_cota == 0
    # No ahu-brasil, a Série Reino e os Códices não têm cota por série, caixa e documento, e
    # por isso nenhum link do DigitArq. Fora deles, sobram as cotas com erro de digitação.
    assert sem_link_brasil <= 200
    # Só **um** documento fica sem link do DigitArq, e isso é correto: a sua única cota é antiga
    # e por região ("AHU-São Paulo-MGouveia, cx. 2, doc. 224"), sem número de série, e daí não
    # sai código CRAV nenhum. Os outros 12 que tinham `UNFORMATTED_CODE` no `reference_code`
    # passaram a ter link pelo `new_code` do Resgate, desde o WP 3.3a.
    assert sem_link == 1


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_bloco_de_ia_tem_os_campos_do_6_2_e_nada_mais():
    dados = carregar_sem_cache(APP_DADOS)
    f = montar(dados, dados.documentos["doc_uid"].iloc[0])
    assert set(f.inferencias) == {"Categoria social do remetente", "Vetor",
                                  "Impacto da tipologia", "SRSP", "Justificativa"}
    assert isinstance(f.inferencias["SRSP"], int)
    assert mod_ficha.TITULO_IA == "Inferências (IA) — DeepSeek v3"


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a1_biografia_escondida_vira_rotulo_nas_fichas_reais():
    """D2/§6.3: a pessoa aparece, o nascimento não, e o rótulo diz o que se sabe."""
    dados = carregar_sem_cache(APP_DADOS)
    escondidas = dados.pessoas[~dados.pessoas["nascimento_exibivel"].astype(bool)]
    assert len(escondidas) == 1106       # 1.359 - 253 exibíveis

    vinculos = dados.doc_pessoa
    uid = next(u for u in escondidas["pessoa_uid"]
               if not vinculos.loc[vinculos["pessoa_uid"] == u].empty)
    doc_uid = vinculos.loc[vinculos["pessoa_uid"] == uid, "doc_uid"].iloc[0]
    p = next(p for p in montar(dados, doc_uid).pessoas if p.pessoa_uid == uid)

    assert p.nome and p.nivel is None and p.nascimento is None and p.fonte is None
    assert p.rotulo in ("Não identificado", "Identificado, local não encontrado",
                        "Nascimento não confirmado")


# ---------------------------------------------------------------------------
# Navegar sem texto: a lente ou o filtro sozinhos valem como consulta
# ---------------------------------------------------------------------------

def test_sem_texto_a_resposta_e_o_corpus_filtrado(corpus_busca):
    from ahu.consulta import Consulta, executar
    from ahu.filtros import Filtros

    tudo = executar(corpus_busca, Consulta())
    assert tudo.total == len(corpus_busca.documentos)
    assert tudo.resultados[0].posicoes == {"corpus": 1}

    so_filtro = executar(corpus_busca, Consulta(filtros=Filtros(tipologias=["CARTA"])))
    assert 0 < so_filtro.total < tudo.total
    for r in so_filtro.resultados:
        assert corpus_busca.documentos.iloc[r.linha]["tipologia"] == "CARTA"


def test_sem_texto_um_filtro_do_deepseek_sozinho_filtra(corpus_busca):
    """D9: as lentes saíram; o filtro que elas preenchiam continua valendo sozinho."""
    from ahu.consulta import Consulta, executar
    from ahu.filtros import FAIXAS_SRSP, Filtros

    so_srsp = executar(corpus_busca, Consulta(filtros=Filtros(srsp_de=FAIXAS_SRSP["alto"][0])))
    assert so_srsp.total < len(corpus_busca.documentos)


def test_ordenacao_por_ano_poe_sem_data_no_fim(corpus_busca):
    import pandas as pd

    from ahu.consulta import Consulta, executar

    anos = corpus_busca.documentos["ano_isolado"].copy()
    try:
        corpus_busca.documentos.loc[corpus_busca.documentos.index[2], "ano_isolado"] = None
        for ordem in ("ano_asc", "ano_desc"):
            r = executar(corpus_busca, Consulta(ordenacao=ordem))
            ultimo = corpus_busca.documentos.iloc[r.resultados[-1].linha]["ano_isolado"]
            assert pd.isna(ultimo), ordem
    finally:
        corpus_busca.documentos["ano_isolado"] = anos


# ---------------------------------------------------------------------------
# O app de verdade, pelo harness do Streamlit
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_sobe_sem_excecao_e_tem_as_abas():
    """Roda o `app.py` de verdade, sem navegador e sem carregar o e5 (busca vazia)."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Catálogo AHU-Brasil"
    app = __import__("app")
    assert [t.label for t in at.tabs] == list(app.ABAS_CATALOGO)
    # A aba Sobre mostra as camadas lidas do manifesto, não escritas no código.
    textos = " ".join(m.value for m in at.markdown)
    assert "Camadas e versões ativas" in textos
    assert "MAE 1.13" in textos or "MAE 1,13" in textos
    # O catálogo geral não mostra os filtros da coleção enriquecida.
    assert "O que são as inferências" not in textos

    at.radio(key="secao").set_value(app.SECOES[1]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == list(app.ABAS_ENRIQUECIDAS)
    textos = " ".join(m.value for m in at.markdown)
    # R5: popover de saiba mais com explicação e ressalva metodológica do preprint
    assert "O que são as inferências" in textos
    assert "Essa inferência, contudo, é metodologicamente delicada" in textos


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_busca_por_cota_sem_carregar_o_modelo():
    """Uma cota não passa pelo motor semântico: a página responde sem baixar o e5."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    at.query_params["v"] = "2"
    at.query_params["q"] = "AHU_ACL_CU_017, Cx. 5, D. 487"
    at.query_params["m"] = "exata"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    textos = " ".join(m.value for m in at.markdown)
    assert "**1** documento(s)" in textos and "pela cota" in textos
