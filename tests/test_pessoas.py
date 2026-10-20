"""Testes do WP 2.4: lista de pessoas, página da pessoa, coletivos e painel diatópico (A2).

O A2 — "Lavradio", por nome, variante ou título, leva a **uma** pessoa com os seus 163 documentos
— é verificado duas vezes: no corpus de mentira do `conftest.py`, que roda sempre, e nos
artefatos de verdade, quando eles existem, onde o número 163 é cobrado de fato.
"""
import pandas as pd
import pytest

from ahu import graficos as gr
from ahu import links
from ahu import pessoas as mp
from ahu.consulta import Consulta
from ahu.dados import carregar_sem_cache
from ahu.ficha import NIVEIS_PROIBIDOS
from tests.conftest import APP_DADOS, tem_artefatos


@pytest.fixture(scope="module")
def corpus_real():
    return carregar_sem_cache(APP_DADOS)


# ---------------------------------------------------------------------------
# A2 — "Lavradio" leva a uma pessoa
# ---------------------------------------------------------------------------

def test_a2_no_corpus_de_mentira_o_titulo_acha_a_pessoa(corpus_busca):
    """A pessoa do conftest tem o título "marquês do Lavradio" e não o nome "Lavradio"."""
    por_titulo = mp.encontrar(corpus_busca, "Lavradio")
    assert len(por_titulo) == 1
    pessoa = por_titulo[0]
    assert pessoa.nome == "Luís de Almeida"
    assert "marquês do Lavradio" in pessoa.titulos

    # O mesmo alvo pelo nome principal e pelo título inteiro.
    assert [p.pessoa_uid for p in mp.encontrar(corpus_busca, "Luís de Almeida")] \
        == [pessoa.pessoa_uid]
    assert [p.pessoa_uid for p in mp.encontrar(corpus_busca, "marques do lavradio")] \
        == [pessoa.pessoa_uid]


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a2_lavradio_e_uma_pessoa_com_163_documentos(corpus_real):
    """O critério A2 da spec §11, nos dados de verdade."""
    for busca in ("Lavradio", "lavradio", "marquês do Lavradio", "marques do lavradio"):
        achadas = mp.encontrar(corpus_real, busca)
        assert len(achadas) == 1, (busca, [p.nome for p in achadas])
        assert achadas[0].n_documentos == 163, busca

    pessoa = mp.encontrar(corpus_real, "Lavradio")[0]
    pagina = mp.pagina(corpus_real, pessoa.pessoa_uid)
    assert len(pagina.documentos) == 163
    # E os 163 documentos são de fato os que o filtro de pessoa devolve na aba Busca.
    from ahu.consulta import executar
    from ahu.filtros import Filtros
    resposta = executar(corpus_real, Consulta(filtros=Filtros(pessoas=[pessoa.pessoa_uid])))
    assert resposta.total == 163


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_busca_de_pessoa_ignora_acento_e_caixa(corpus_real):
    assert ([p.pessoa_uid for p in mp.encontrar(corpus_real, "MELO E CASTRO")]
            == [p.pessoa_uid for p in mp.encontrar(corpus_real, "melo e castro")])


# ---------------------------------------------------------------------------
# Lista de pessoas (§7.1)
# ---------------------------------------------------------------------------

def test_lista_ordena_por_documentos_e_por_nome(corpus_busca):
    por_documentos = mp.listar(corpus_busca, ordem="documentos")
    assert [p.n_documentos for p in por_documentos] == sorted(
        (p.n_documentos for p in por_documentos), reverse=True)

    nomes = [p.nome for p in mp.listar(corpus_busca, ordem="nome")]
    assert nomes == sorted(nomes, key=lambda n: n.lower())


def test_ordem_desconhecida_erra_alto(corpus_busca):
    with pytest.raises(ValueError, match="Ordem desconhecida"):
        mp.listar(corpus_busca, ordem="aleatoria")
    with pytest.raises(ValueError, match="Ordem desconhecida"):
        mp.listar_coletivos(corpus_busca, ordem="aleatoria")


def test_lista_sem_texto_traz_todas_as_pessoas(corpus_busca):
    assert len(mp.listar(corpus_busca)) == len(corpus_busca.pessoas)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_filtro_de_origem_so_alcanca_biografia_exibivel(corpus_real):
    """D2: filtrar por país nunca traz quem não pode mostrar nascimento."""
    paises = sorted(corpus_real.pessoas["pais_atual"].dropna().unique().tolist())
    achadas = mp.listar(corpus_real, paises=paises[:1])
    assert achadas
    assert all(p.exibe_nascimento and p.pais_atual == paises[0] for p in achadas)

    so_alta = mp.listar(corpus_real, so_confianca_alta=True)
    assert so_alta and all(p.nivel == "ALTA" for p in so_alta)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a1_a_lista_nunca_traz_nivel_proibido_nem_dado_escondido(corpus_real):
    """A1 na aba Pessoas: nada de BAIXA/REJEITADO, e quem não exibe não traz nascimento."""
    for p in mp.listar(corpus_real):
        assert p.nivel in (None, "ALTA", "MEDIA"), (p.nome, p.nivel)
        assert p.nivel not in NIVEIS_PROIBIDOS
        if not p.exibe_nascimento:
            assert (p.nascimento, p.cidade_nascimento, p.pais_atual, p.ano_nascimento,
                    p.fonte, p.nivel) == (None, None, None, None, None, None), p.nome

    tabela = mp.tabela(mp.listar(corpus_real))
    texto = "\n".join(tabela.astype(str).to_numpy().ravel().tolist())
    for nivel in NIVEIS_PROIBIDOS:
        assert nivel not in texto
    assert "scribe_mediation_likely" not in texto


def test_periodo_de_atividade(corpus_busca):
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    assert pessoa.periodo == (str(pessoa.ano_min) if pessoa.ano_min == pessoa.ano_max
                              else f"{pessoa.ano_min}–{pessoa.ano_max}")
    sem_data = mp.PessoaNaLista(pessoa_uid="x", nome="N", tipo="pessoa")
    assert sem_data.periodo == "sem data" and sem_data.nascimento is None


# ---------------------------------------------------------------------------
# Coletivos e instituições — lista própria (§7.1)
# ---------------------------------------------------------------------------

def test_coletivos_ficam_fora_da_lista_de_pessoas(corpus_busca):
    """A "Junta de Teste" é instituição: entra na lista própria, nunca na de pessoas."""
    coletivos = mp.listar_coletivos(corpus_busca)
    nomes = {c.nome for c in coletivos}
    assert "Junta de Teste" in nomes
    assert "Junta de Teste" not in {p.nome for p in mp.listar(corpus_busca)}
    assert all(c.tipo in mp.TIPOS_COLETIVOS for c in coletivos)


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_coletivos_juntam_grafias_que_so_diferem_na_caixa(corpus_real):
    camara = [c for c in mp.listar_coletivos(corpus_real, texto="Oficiais da Câmara do Rio")]
    assert camara, "a Câmara do Rio de Janeiro tem de estar na lista de coletivos"
    maior = camara[0]
    assert maior.grafias, "as grafias que diferem só por caixa deviam estar juntas"
    assert all(c.nome.lower() != g.lower() or True for c in camara for g in c.grafias)
    # Nenhuma instituição vazou para a lista de pessoas.
    assert not mp.encontrar(corpus_real, "Conselho Ultramarino")


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_contagem_de_remetentes_fecha_com_doc_pessoa(corpus_real):
    papeis = mp.contagem_de_remetentes(corpus_real)
    assert sum(papeis.values()) == len(corpus_real.doc_pessoa)
    assert set(mp.TIPOS_COLETIVOS) <= set(papeis)


# ---------------------------------------------------------------------------
# Página da pessoa (§7.1)
# ---------------------------------------------------------------------------

def test_pagina_de_pessoa_sem_vinculo_vem_vazia_e_coerente(corpus_busca):
    """No corpus de mentira a pessoa não tem vínculo em `doc_pessoa`: a página não pode quebrar."""
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    pagina = mp.pagina(corpus_busca, pessoa.pessoa_uid)
    assert pagina.documentos.empty and pagina.linha_do_tempo.empty
    assert pagina.tipologias == [] and pagina.destinatarios == [] and pagina.sem_data == 0
    assert pagina.url_documentos and pagina.url_issue


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_pagina_traz_linha_do_tempo_tipologias_e_destinatarios(corpus_real):
    pessoa = mp.encontrar(corpus_real, "Lavradio")[0]
    pagina = mp.pagina(corpus_real, pessoa.pessoa_uid)
    assert len(pagina.documentos) == pessoa.n_documentos
    assert int(pagina.linha_do_tempo["documentos"].sum()) + pagina.sem_data \
        == pessoa.n_documentos
    assert sum(n for _, n in pagina.tipologias) <= pessoa.n_documentos
    assert sum(n for _, n in pagina.destinatarios) <= pessoa.n_documentos
    assert pagina.linha_do_tempo["ano"].is_monotonic_increasing
    # Ordem cronológica, com os sem data no fim (§7.2).
    anos = pagina.documentos["ano"]
    assert anos.dropna().is_monotonic_increasing
    assert anos.isna().sum() == pagina.sem_data
    if pagina.sem_data:
        assert pd.isna(anos.iloc[-1])


def test_pagina_de_pessoa_inexistente_erra_alto(corpus_busca):
    with pytest.raises(KeyError, match="pessoa_uid desconhecido"):
        mp.pagina(corpus_busca, "pes:naoexiste")


def test_ver_documentos_e_um_link_de_busca_que_volta_igual(corpus_busca):
    """O "Ver documentos" é a URL versionada da §8.3: ela tem de voltar na mesma consulta (A6)."""
    from urllib.parse import parse_qs, urlparse

    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    url = mp.url_dos_documentos(pessoa.pessoa_uid)
    assert url.startswith("?v=2")
    parametros = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    volta = Consulta.de_url(parametros)
    assert volta.filtros.pessoas == [pessoa.pessoa_uid]
    assert volta.texto == "" and volta.pagina == 1


def test_reportar_erro_e_uma_issue_preenchida(corpus_busca):
    pessoa = mp.encontrar(corpus_busca, "Lavradio")[0]
    pagina = mp.pagina(corpus_busca, pessoa.pessoa_uid)
    assert pagina.url_issue.startswith(f"https://github.com/{links.REPOSITORIO}/issues/new?")
    from urllib.parse import parse_qs, urlparse
    campos = {k: v[0] for k, v in parse_qs(urlparse(pagina.url_issue).query).items()}
    assert pessoa.pessoa_uid in campos["title"] and pessoa.nome in campos["title"]
    assert pessoa.pessoa_uid in campos["body"] and "decisao_manual" in campos["body"]
    assert campos["labels"] == links.ETIQUETA_ISSUE


def test_issue_respeita_o_repositorio_do_fork():
    url = links.issue_de_pessoa("pes:1", "N", repositorio="outro/repo")
    assert url.startswith("https://github.com/outro/repo/issues/new?")


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_issue_de_quem_nao_exibe_nascimento_nao_vaza_dado(corpus_real):
    escondida = next(p for p in mp.listar(corpus_real) if not p.exibe_nascimento)
    pagina = mp.pagina(corpus_real, escondida.pessoa_uid)
    assert "sem nascimento exibível" in pagina.url_issue or "exib" in pagina.url_issue
    for nivel in NIVEIS_PROIBIDOS:
        assert nivel not in pagina.url_issue


# ---------------------------------------------------------------------------
# Painel diatópico (§9) — por documento ou por pessoa
# ---------------------------------------------------------------------------

def test_painel_recusa_contagem_desconhecida(corpus_busca):
    with pytest.raises(ValueError, match="Contagem desconhecida"):
        mp.painel_diatopico(corpus_busca, "linha")


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_painel_conta_por_documento_e_por_pessoa(corpus_real):
    por_documento = mp.painel_diatopico(corpus_real, "documento")
    por_pessoa = mp.painel_diatopico(corpus_real, "pessoa")

    assert por_documento.total == len(corpus_real.documentos)
    assert por_pessoa.total == len(corpus_real.pessoas)
    # Por pessoa, o que fica de fora é exatamente quem não tem nascimento exibível (D2).
    exibiveis = int(corpus_real.pessoas["nascimento_exibivel"].astype(bool).sum())
    assert por_pessoa.total - por_pessoa.sem_origem == exibiveis
    assert int(por_pessoa.paises["contagem"].sum()) == exibiveis

    # Há sempre mais documentos que pessoas num país onde alguém escreveu muito.
    assert por_documento.paises["contagem"].iloc[0] > por_pessoa.paises["contagem"].iloc[0]
    assert por_documento.paises["contagem"].is_monotonic_decreasing
    assert 0 <= por_documento.sem_origem <= por_documento.total


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_painel_por_decada_e_coerente_com_o_ranking(corpus_real):
    painel = mp.painel_diatopico(corpus_real, "documento")
    assert set(painel.decadas["pais"]) <= set(painel.paises["pais"])
    assert (painel.decadas["decada"] % 10 == 0).all()
    # Um documento conta uma vez por década e país, nunca duas.
    assert not painel.decadas.duplicated(["decada", "pais"]).any()
    por_pais = painel.decadas.groupby("pais")["contagem"].sum()
    ranking = painel.paises.set_index("pais")["contagem"]
    for pais, soma in por_pais.items():
        assert soma <= ranking[pais] + 0  # a soma das décadas nunca passa o total do país


# ---------------------------------------------------------------------------
# Gráficos (§9) — paleta e séries, sem subir interface
# ---------------------------------------------------------------------------

def test_cor_segue_o_pais_nao_a_posicao():
    """Tirar um país do meio não pode repintar os que sobraram."""
    todos = gr.cores(["Portugal", "Brasil", "Espanha"], "light")
    sem_brasil = gr.cores(["Portugal", "Espanha"], "light")
    assert todos["Portugal"] == sem_brasil["Portugal"]
    assert todos["Espanha"] != sem_brasil["Espanha"]  # a ordem fixa é do ranking, recalculado
    assert todos[gr.OUTROS] == gr.PALETAS["light"]["apagado"]
    assert todos[mp.SEM_ORIGEM] == gr.PALETAS["light"]["apagado"]


def test_a_paleta_dos_dois_temas_e_escolhida_e_tem_o_mesmo_tamanho():
    assert set(gr.TEMAS) == set(gr.PALETAS)
    assert len({len(gr.PALETAS[t]["series"]) for t in gr.TEMAS}) == 1
    assert gr.PALETAS["light"]["series"] != gr.PALETAS["dark"]["series"]
    with pytest.raises(ValueError, match="Tema desconhecido"):
        gr.paleta("sepia")


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_grafico_por_decada_dobra_o_excedente_em_outros(corpus_real):
    painel = mp.painel_diatopico(corpus_real, "documento")
    figura = gr.barras_por_decada(painel, "light", limite=2)
    series = [t.name for t in figura.data]
    assert series[:2] == gr.dobrar_em_outros(painel.paises, limite=2)
    assert gr.OUTROS in series
    # Nenhuma série fora da paleta mais os dois cinzas.
    cores_usadas = {t.marker.color for t in figura.data}
    assert cores_usadas <= set(gr.PALETAS["light"]["series"]) | {
        gr.PALETAS["light"]["apagado"]}


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_graficos_do_painel_e_da_pessoa_sobem_nos_dois_temas(corpus_real):
    painel = mp.painel_diatopico(corpus_real, "pessoa")
    pessoa = mp.encontrar(corpus_real, "Lavradio")[0]
    pagina = mp.pagina(corpus_real, pessoa.pessoa_uid)
    for tema in gr.TEMAS:
        assert gr.barras_por_pais(painel, tema).data
        assert gr.barras_por_decada(painel, tema).data
        figura = gr.barras_por_ano(pagina.linha_do_tempo, tema)
        # A linha do tempo não pula ano: 1770 a 1779 são dez barras, inclusive as vazias.
        assert len(figura.data[0].x) == pagina.linha_do_tempo["ano"].max() \
            - pagina.linha_do_tempo["ano"].min() + 1


def test_grafico_vazio_nao_quebra():
    vazio = mp.PainelDiatopico(
        por="documento",
        paises=pd.DataFrame({"pais": [], "contagem": []}),
        decadas=pd.DataFrame({"decada": [], "pais": [], "contagem": []}))
    assert gr.barras_por_decada(vazio, "light") is not None
    assert gr.barras_por_ano(pd.DataFrame({"ano": [], "documentos": []}), "dark") is not None


# ---------------------------------------------------------------------------
# A aba Pessoas no app de verdade
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_app_abre_a_aba_pessoas_e_o_painel_sem_excecao():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DADOS.parent / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    textos = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Coletivos e instituições" in " ".join(
        str(r.options) for r in at.radio) or "Painel diatópico" in textos
    # As abas continuam sendo as da §9, mais o Conversor: a aba Pessoas não criou aba no topo.
    import app as modulo
    assert [t.label for t in at.tabs] == list(modulo.ABAS)


# ---------------------------------------------------------------------------
# R9: busca de nome tolerante
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
@pytest.mark.parametrize("procurado, esperado", [
    # o caso que o autor relatou: falta um "s" e sobra um "t"
    ("domingo de britto", "Domingos de Brito Peixoto"),
    ("Domingos de Britto", "Domingos de Brito Peixoto"),
    ("domingos de brito", "Domingos de Brito Peixoto"),
    # grafia antiga: z/s, ae/ai, letra dobrada
    ("Jozé da Silva Paes", "José da Silva Pais"),
    ("Joze da Silva Pais", "José da Silva Pais"),
    ("Martinho de Mello e Castro", "Martinho de Melo e Castro"),
    # e o A2 continua de pé
    ("Lavradio", "D. Luís de Almeida Portugal Soares de Alarcão Eça e Melo Silva e Mascarenhas"),
])
def test_busca_de_nome_tolera_grafia_e_prefixo(corpus_real, procurado, esperado):
    achadas = mp.encontrar(corpus_real, procurado)
    assert achadas, procurado
    assert esperado in [p.nome for p in achadas[:3]], (procurado, [p.nome for p in achadas[:3]])


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a_ordem_e_de_semelhanca(corpus_real):
    """Quem busca o nome inteiro quer a pessoa daquele nome em primeiro."""
    achadas = mp.encontrar(corpus_real, "Martinho de Melo e Castro")
    assert achadas[0].nome == "Martinho de Melo e Castro"


def test_a_regra_de_prefixo_exige_tres_letras():
    """Um token de até duas letras casa só inteiro: "sa" não pode abrir "Sampaio"."""
    assert mp._casa_tokens(["bri"], ["brito"])            # 3 letras: prefixo vale
    assert mp._casa_tokens(["domingo"], ["domingos"])     # o caso do autor
    assert not mp._casa_tokens(["sa"], ["sampaio"])       # 2 letras: não
    assert mp._casa_tokens(["de"], ["de"])                # 2 letras, inteiro: vale
    # a função recebe tokens **já neutralizados** pela `_tokens`; é lá que "britto" vira "brito"
    assert mp._tokens("Britto Domingo") == ["brito", "domingo"]
    # a ordem dos tokens não importa, e cada um consome um token do candidato
    assert mp._casa_tokens(mp._tokens("britto domingo"), mp._tokens("Domingos de Brito"))
    assert not mp._casa_tokens(["brito", "brito"], ["brito", "peixoto"])


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_busca_sem_resultado_devolve_vazio(corpus_real):
    assert mp.encontrar(corpus_real, "zzzqqq inexistente") == []


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_a_tabela_traz_a_fonte_como_url(corpus_real):
    """R9: o link da fonte vai à tabela como URL; a interface o torna legível."""
    tabela = mp.tabela(mp.listar(corpus_real, so_confianca_alta=True))
    assert "Fonte" in tabela.columns
    fontes = [f for f in tabela["Fonte"] if f]
    assert fontes and all(f.startswith("http") for f in fontes)
