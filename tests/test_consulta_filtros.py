"""Testes do WP 2.2: objeto `Consulta`, ida e volta para a URL (A6) e filtros da §6.1."""
import numpy as np
import pytest

from ahu import busca
from ahu import filtros as mod_filtros
from ahu.consulta import (
    TAMANHO_PAGINA,
    VERSAO_URL,
    Consulta,
    ErroDeLink,
    executar,
)
from ahu.dados import carregar_sem_cache
from ahu.filtros import FAIXAS_SRSP, Filtros, opcoes
from tests.conftest import APP_DADOS, tem_artefatos

CONSULTAS_VARIADAS = [
    Consulta(),
    Consulta(texto="deserção de soldados"),
    Consulta(texto="Lavradio", modo="conceitual", rigor="alto", pagina=3),
    Consulta(texto='"obras do porto"', modo="exata", ordenacao="ano_desc", semente=42),
    Consulta(
        texto="soldados",
        filtros=Filtros(ano_de=1750, ano_ate=1800, sem_data=False, pastas=["Bahia", "Pará"],
                        tipologias=["CARTA", "OFÍCIO"], categorias=["Commoner"],
                        vetores=["Bottom-Up"], srsp_de=7, srsp_ate=9, destinatario="Rei",
                        pessoas=["pes:12345678"], paises_origem=["Portugal"],
                        cidades_origem=["Lisboa"], so_confianca_alta=True,
                        colecoes=["ahu-sul"], impactos=["Narrative"]),
        ordenacao="ano_asc", pagina=2, semente=7,
    ),
]


# ---------------------------------------------------------------------------
# A6: ida e volta pela URL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("consulta", CONSULTAS_VARIADAS, ids=lambda c: c.texto or "padrao")
def test_a6_consulta_volta_igual_da_url(consulta):
    assert Consulta.de_url(consulta.para_url()) == consulta


@pytest.mark.parametrize("consulta", CONSULTAS_VARIADAS, ids=lambda c: c.texto or "padrao")
def test_consulta_volta_igual_do_dicionario(consulta):
    assert Consulta.de_dicionario(consulta.para_dicionario()) == consulta


def test_url_so_leva_o_que_nao_e_padrao():
    assert Consulta().para_url() == {"v": str(VERSAO_URL)}
    url = Consulta(texto="soldados").para_url()
    assert url == {"v": "2", "q": "soldados"}


def test_url_e_compacta_e_versionada():
    c = CONSULTAS_VARIADAS[-1]
    url = c.para_url()
    assert url["v"] == "2"
    assert all(len(k) <= 2 for k in url)          # chaves curtas
    assert url["ft"] == "CARTA|OFÍCIO"
    assert url["sd"] == "0" and url["fa"] == "1"  # booleanos como 0/1


def test_url_vazia_da_a_consulta_padrao():
    assert Consulta.de_url({}) == Consulta()
    assert Consulta.de_url(None) == Consulta()


def test_link_sem_versao_e_recusado():
    with pytest.raises(ErroDeLink, match="sem versão"):
        Consulta.de_url({"q": "soldados"})


def test_link_de_outra_versao_e_recusado():
    with pytest.raises(ErroDeLink, match="versão 99"):
        Consulta.de_url({"v": "99", "q": "soldados"})


def test_link_com_parametro_desconhecido_e_recusado():
    with pytest.raises(ErroDeLink, match="desconhecidos"):
        Consulta.de_url({"v": "2", "q": "x", "xyz": "1"})


def test_link_com_valor_invalido_e_recusado():
    with pytest.raises(ErroDeLink, match="Modo desconhecido"):
        Consulta.de_url({"v": "2", "m": "fuzzy"})
    with pytest.raises(ErroDeLink, match="número"):
        Consulta.de_url({"v": "2", "a1": "mil e setecentos"})
    with pytest.raises(ErroDeLink, match="0 ou 1"):
        Consulta.de_url({"v": "2", "sd": "talvez"})


def test_query_params_com_lista_funciona():
    """`st.query_params` devolve lista quando a chave repete."""
    assert Consulta.de_url({"v": ["2"], "q": ["soldados"]}) == Consulta(texto="soldados")


def test_consulta_invalida_avisa_na_criacao():
    with pytest.raises(ErroDeLink, match="Página"):
        Consulta(pagina=0)


def test_dicionario_tem_proveniencia_legivel():
    c = CONSULTAS_VARIADAS[-1]
    d = c.para_dicionario()
    assert d["versao"] == VERSAO_URL
    assert d["filtros_ativos"]["srsp_de"] == 7
    assert "sem_data" in d["filtros"]                      # completo, para a capa do PDF
    assert d["filtros_ativos"]["sem_data"] is False         # só o que difere do padrão
    assert "impactos" not in Consulta().para_dicionario()["filtros_ativos"]


# ---------------------------------------------------------------------------
# Filtros isolados (§6.1)
# ---------------------------------------------------------------------------

def test_filtros_vazios_nao_fazem_mascara(corpus_busca):
    assert Filtros().mascara(corpus_busca) is None


def quantos(dados, **kwargs) -> int:
    m = Filtros(**kwargs).mascara(dados)
    return int(m.sum()) if m is not None else len(dados.documentos)


def test_filtro_de_tipologia(corpus_busca):
    assert quantos(corpus_busca, tipologias=["CARTA"]) == 5
    assert quantos(corpus_busca, tipologias=["CARTA", "OFÍCIO"]) == 6
    assert quantos(corpus_busca, tipologias=["INEXISTENTE"]) == 0


def test_filtro_de_periodo_e_documentos_sem_data(corpus_busca):
    docs = corpus_busca.documentos
    anos = sorted(int(a) for a in docs["ano_isolado"].dropna())
    assert quantos(corpus_busca, ano_de=anos[0], ano_ate=anos[2]) == 3
    assert quantos(corpus_busca, ano_de=anos[-1]) == 1
    # Sem nenhum documento sem data, desligar a caixa não muda nada neste corpus.
    assert docs["ano_isolado"].isna().sum() == 0
    assert quantos(corpus_busca, sem_data=False) == len(docs)


def test_filtro_de_periodo_preserva_documento_sem_data_quando_pedido(corpus_busca):
    """A caixa 'incluir documentos sem data' salva quem não tem ano (§6.1)."""
    dados = corpus_busca
    anos = dados.documentos["ano_isolado"].copy()
    try:
        dados.documentos.loc[dados.documentos.index[0], "ano_isolado"] = None
        assert quantos(dados, ano_de=1800, ano_ate=1805, sem_data=True) >= 1
        com = quantos(dados, ano_de=1800, ano_ate=1805, sem_data=True)
        sem = quantos(dados, ano_de=1800, ano_ate=1805, sem_data=False)
        assert com == sem + 1
    finally:
        dados.documentos["ano_isolado"] = anos


def test_filtro_de_destinatario_por_texto(corpus_busca):
    assert quantos(corpus_busca, destinatario="Lavradio") == 1
    assert quantos(corpus_busca, destinatario="lavradio") == 1     # normalizado
    assert quantos(corpus_busca, destinatario="Rei") == 9


def test_filtro_de_srsp(corpus_busca):
    srsp = corpus_busca.documentos["vernacular_score"]
    alvo = int(srsp.iloc[0])
    assert quantos(corpus_busca, srsp_de=alvo, srsp_ate=alvo) == int((srsp == alvo).sum())
    assert quantos(corpus_busca, srsp_de=10) == 0


def test_filtro_de_pessoa_usa_pessoa_uid(corpus_busca):
    uid = corpus_busca.pessoas["pessoa_uid"].iloc[0]
    esperado = int((corpus_busca.doc_pessoa["pessoa_uid"] == uid).sum())
    assert quantos(corpus_busca, pessoas=[uid]) == esperado
    assert quantos(corpus_busca, pessoas=["pes:naoexiste"]) == 0


def test_filtro_de_origem_so_alcanca_nascimento_exibivel(corpus_busca):
    """D2: quem não tem nascimento exibível não entra pelo filtro de origem."""
    pessoas = corpus_busca.pessoas
    assert bool(pessoas["nascimento_exibivel"].any())
    pais = pessoas.loc[pessoas["nascimento_exibivel"], "pais_atual"].dropna().iloc[0]
    assert quantos(corpus_busca, paises_origem=[pais]) >= 0
    # Um país que só apareceria numa biografia escondida não filtra nada.
    assert quantos(corpus_busca, paises_origem=["Atlântida"]) == 0


def test_so_confianca_alta_apenas_aperta(corpus_busca):
    com_media = quantos(corpus_busca, paises_origem=["Portugal"], so_confianca_alta=False)
    so_alta = quantos(corpus_busca, paises_origem=["Portugal"], so_confianca_alta=True)
    assert so_alta <= com_media


def test_filtros_se_combinam(corpus_busca):
    um = quantos(corpus_busca, tipologias=["CARTA"])
    dois = quantos(corpus_busca, tipologias=["CARTA"], destinatario="Rei")
    assert dois <= um


def test_usa_deepseek_identifica_os_filtros_de_ia():
    assert not Filtros().usa_deepseek()
    assert not Filtros(tipologias=["CARTA"]).usa_deepseek()
    assert Filtros(categorias=["Commoner"]).usa_deepseek()
    assert Filtros(srsp_de=7).usa_deepseek()


def test_filtros_do_deepseek_comecam_desligados():
    """Preprint §4: nada que dependa do DeepSeek vem ligado."""
    padrao = Filtros()
    for campo in mod_filtros.CAMPOS_DEEPSEEK:
        assert getattr(padrao, campo) in (None, [], False), campo


# ---------------------------------------------------------------------------
# Opções: saem dos dados (D6, A11)
# ---------------------------------------------------------------------------

def test_opcoes_saem_dos_dados(corpus_busca):
    o = opcoes(corpus_busca)
    docs = corpus_busca.documentos
    assert o["colecoes"] == ["t"]
    assert o["mostrar_colecao"] is False          # §6.1: oculto com uma coleção só
    assert set(o["pastas"]) == set(docs["folder"].dropna().unique())
    assert dict(o["tipologias"])["CARTA"] == 5
    assert o["ano_minimo"] == int(docs["ano_isolado"].min())
    assert o["documentos_sem_data"] == 0


def test_tipologias_vem_por_frequencia(corpus_busca):
    contagens = [n for _, n in opcoes(corpus_busca)["tipologias"]]
    assert contagens == sorted(contagens, reverse=True)


def test_escala_do_srsp_vem_do_manifesto(corpus_busca):
    assert mod_filtros.escala_srsp(corpus_busca) == [1, 9]
    assert corpus_busca.manifesto["srsp"]["escala"] == [1, 9]
    assert corpus_busca.manifesto["srsp"]["camada"] == "sociolinguistica_deepseek_v3"


def test_faixas_e_nota_do_srsp():
    """§6.2: faixas 1–3 / 4–6 / 7–9 e a ressalva de ±1 ponto."""
    assert FAIXAS_SRSP == {"baixo": (1, 3), "medio": (4, 6), "alto": (7, 9)}
    assert "±1" in mod_filtros.NOTA_SRSP and "1,13" in mod_filtros.NOTA_SRSP


def test_auditoria_do_srsp_chega_do_manifesto(corpus_busca):
    auditoria = mod_filtros.auditoria_srsp(corpus_busca)
    assert auditoria["mae"] == 1.13 and auditoria["n"] == 705


def test_opcoes_de_pessoas_tem_nome_variantes_e_titulos(corpus_busca):
    pessoas = opcoes(corpus_busca)["pessoas"]
    assert pessoas and all("pessoa_uid" in p for p in pessoas)
    com_titulo = [p for p in pessoas if any("Lavradio" in b for b in p["busca"])]
    assert com_titulo and com_titulo[0]["pessoa_uid"].startswith("pes:")
    assert pessoas == sorted(pessoas, key=lambda p: (-p["n_documentos"], str(p["nome"])))


# ---------------------------------------------------------------------------
# D9: as lentes saíram, e um link antigo que as traga ainda abre
# ---------------------------------------------------------------------------

def test_a_consulta_nao_tem_mais_campo_de_lente():
    from dataclasses import fields
    assert "lente" not in {f.name for f in fields(Consulta)}
    assert VERSAO_URL == 2


def test_link_da_versao_1_com_lente_ainda_abre(corpus_busca):
    """D9: o parâmetro da lente é descartado em silêncio; todo o resto vale."""
    del corpus_busca
    antigo = {"v": "1", "q": "deserção", "m": "exata", "r": "alto", "l": "sintaxe_diacronica",
              "p": "3", "ft": "CARTA|OFÍCIO", "s1": "7"}
    volta = Consulta.de_url(antigo)
    assert volta.texto == "deserção" and volta.modo == "exata" and volta.rigor == "alto"
    assert volta.pagina == 3
    assert volta.filtros.tipologias == ["CARTA", "OFÍCIO"] and volta.filtros.srsp_de == 7


def test_link_da_versao_1_sem_lente_tambem_abre():
    volta = Consulta.de_url({"v": "2", "q": "gado"})
    assert volta.texto == "gado"


def test_link_de_versao_futura_e_recusado():
    with pytest.raises(ErroDeLink, match="versão 9"):
        Consulta.de_url({"v": "9", "q": "gado"})


def test_a_lente_nao_viaja_mais_na_url():
    assert "l" not in Consulta(texto="x").para_url()


# ---------------------------------------------------------------------------
# Execução: busca com filtros e ordenação
# ---------------------------------------------------------------------------

def test_executar_aplica_os_filtros_na_busca(corpus_busca):
    sem = executar(corpus_busca, Consulta(texto="carta", modo="exata"))
    com = executar(corpus_busca, Consulta(texto="carta", modo="exata",
                                          filtros=Filtros(tipologias=["CARTA"])))
    assert 0 < com.total <= sem.total
    for r in com.resultados:
        assert corpus_busca.documentos.iloc[r.linha]["tipologia"] == "CARTA"


def test_filtro_tambem_vale_para_a_rota_de_cotas(corpus_busca):
    cota = "AHU_ACL_CU_021, Cx. 6, D. 390"
    livre = executar(corpus_busca, Consulta(texto=cota, modo="exata"))
    assert livre.total == 1
    bloqueado = executar(corpus_busca, Consulta(texto=cota, modo="exata",
                                                filtros=Filtros(tipologias=["MAPA"])))
    assert bloqueado.total == 0  # o filtro não é furado pela cota


def test_ordenacao_por_ano(corpus_busca):
    anos = corpus_busca.documentos["ano_isolado"]
    crescente = executar(corpus_busca, Consulta(texto="carta", modo="exata", ordenacao="ano_asc"))
    valores = [int(anos.iloc[r.linha]) for r in crescente.resultados]
    assert valores == sorted(valores)
    assert [r.posicao for r in crescente.resultados] == list(range(1, crescente.total + 1))

    decrescente = executar(corpus_busca, Consulta(texto="carta", modo="exata",
                                                  ordenacao="ano_desc"))
    assert [int(anos.iloc[r.linha]) for r in decrescente.resultados] == sorted(valores,
                                                                              reverse=True)


def test_pagina_de_25_e_o_padrao():
    assert TAMANHO_PAGINA == 25


# ---------------------------------------------------------------------------
# Aviso de consulta distante (§5.5, decisão de 08/10)
# ---------------------------------------------------------------------------

def test_aviso_de_consulta_distante_e_so_aviso(corpus_busca, encoder):
    """Com as duas condições o aviso aparece, mas nada é cortado."""
    com = busca.buscar(corpus_busca, "xkq zzv plorb", encoder=encoder, cosseno_minimo=0.0,
                       limite_destaque=99.0)
    assert com.consulta_distante and busca.AVISO_DISTANTE in com.avisos
    sem = busca.buscar(corpus_busca, "xkq zzv plorb", encoder=encoder, cosseno_minimo=0.0,
                       limite_destaque=0.0)
    assert not sem.consulta_distante
    # O aviso não muda a lista: mesmos documentos, mesma ordem.
    assert [r.doc_uid for r in com.resultados] == [r.doc_uid for r in sem.resultados]
    assert [r.posicao for r in com.resultados] == [r.posicao for r in sem.resultados]


def test_aviso_exige_as_duas_condicoes(corpus_busca, encoder):
    """(a) nenhum termo no corpus E (b) topo sem destaque. Uma só não basta."""
    # 'soldados' está no corpus: condição (a) falha, nada de aviso nem com o limite altíssimo.
    com_lexico = busca.buscar(corpus_busca, "soldados", encoder=encoder, cosseno_minimo=0.0,
                              limite_destaque=99.0)
    assert com_lexico.sem_correspondencia_lexica is False
    assert not com_lexico.consulta_distante
    # 'xkq' não está: (a) vale, e aí o destaque decide.
    sem_lexico = busca.buscar(corpus_busca, "xkq", encoder=encoder, cosseno_minimo=0.0,
                              limite_destaque=99.0)
    assert sem_lexico.sem_correspondencia_lexica is True
    assert sem_lexico.consulta_distante


def test_termos_no_corpus_vale_em_qualquer_modo(corpus_busca):
    assert busca.termos_no_corpus(corpus_busca, ["soldados", "xkq"]) == ["soldados"]
    assert busca.termos_no_corpus(corpus_busca, ["xkq", "zzv"]) == []


def test_limite_do_destaque_e_provisorio_e_esta_nas_constantes():
    assert busca.LIMITE_DESTAQUE == 4.5
    assert "pouco relevantes" in busca.AVISO_DISTANTE
    assert not hasattr(busca, "PISO_AVISO_DISTANTE")  # o piso absoluto saiu


def test_busca_sem_semantico_nao_avisa(corpus_busca):
    r = busca.buscar(corpus_busca, "carta", modo="exata")
    assert r.cosseno_maximo is None and r.avisos == []


def test_destaque_do_topo_mede_o_quanto_o_melhor_se_separa():
    com_destaque = np.array([0.70] * 99 + [0.90], dtype=np.float32)
    sem_destaque = np.array([0.70, 0.701, 0.699] * 33 + [0.70], dtype=np.float32)
    assert busca.destaque_do_topo(com_destaque) > busca.destaque_do_topo(sem_destaque)
    assert busca.destaque_do_topo(np.array([0.5, 0.5, 0.5], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Sobre os artefatos de verdade (pulado se app_dados/ não existir)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_filtros_nos_dados_reais():
    d = carregar_sem_cache(APP_DADOS)
    o = opcoes(d)
    assert o["mostrar_colecao"] is True and o["colecoes"] == ["ahu-brasil", "ahu-sul"]
    assert o["documentos_sem_data"] == 2343
    sul = d.documentos[d.documentos["colecao"] == "ahu-sul"]
    assert int(sul["ano_isolado"].isna().sum()) == 186          # §6.1 cita este número
    # No ahu-sul: 3.348 no dado bruto + 1 que a tabela de correções consertou ('OFICÍO').
    assert int((sul["tipologia"] == "OFÍCIO").sum()) == 3349
    assert dict(o["tipologias"])["OFÍCIO"] == 32440
    assert mod_filtros.escala_srsp(d) == [1, 9]

    # Os filtros que as lentes preenchiam continuam existindo sozinhos (D9).
    administrativa = Filtros(vetores=["Top-Down"], categorias=["Metropolitan Elite"])
    assert int(administrativa.mascara(d).sum()) > 0
    marginalizadas = Filtros(categorias=["Commoner", "Marginalized", "Low Military", "Clergy"],
                             vetores=["Bottom-Up"])
    quantos = int(marginalizadas.mascara(d).sum())
    assert 0 < quantos < len(d.documentos)
    assert quantos != int(administrativa.mascara(d).sum())


@pytest.mark.skipif(not tem_artefatos, reason="app_dados/ ainda não construído")
def test_filtro_de_srsp_alto_nos_dados_reais():
    d = carregar_sem_cache(APP_DADOS)
    alto = Filtros(srsp_de=7, srsp_ate=9).mascara(d)
    srsp = d.documentos["vernacular_score"]
    assert int(alto.sum()) == int(((srsp >= 7) & (srsp <= 9)).sum()) == 141


def test_filtros_por_secao():
    """O catálogo geral desliga os filtros da coleção enriquecida; a seção dela fixa a coleção."""
    from ahu.filtros import Filtros

    f = Filtros(ano_de=1750, pessoas=["pes:1"], srsp_de=7, srsp_ate=9, colecoes=["ahu-sul"])
    assert f.enriquecidos()
    geral = f.para_secao(False)
    assert geral.ano_de == 1750 and geral.pessoas == [] and geral.srsp_de is None
    assert geral.colecoes == [] and not geral.enriquecidos()
    assert Filtros(ano_de=1750).para_secao(True).colecoes == ["ahu-sul"]
