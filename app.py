"""app.py — Classificador AHU-Sul v3: interface (spec §6, §7.1, §9).

Só Streamlit aqui. Toda a lógica está em `ahu/`: busca, filtros, consulta, ficha, links, pessoas
e gráficos.
O app lê **apenas** `app_dados/` (AGENTS.md 3), por `ahu.dados.carregar`, que recusa iniciar com
artefato desalinhado (A4).

Abas (§9): Busca · Pessoas · Amostras · Painel · Conversor · Sobre e método. Busca e ficha são o
WP 2.3; Pessoas e o painel diatópico, o WP 2.4; o Conversor de cotas veio depois da v3.0. As
que dependem da F3 e da F4 aparecem como "em breve", para que a navegação já seja a definitiva.

Rodar:
    streamlit run app.py
"""
from __future__ import annotations

import contextlib
import os

import pandas as pd
import streamlit as st

from ahu import amostras as mod_amostras
from ahu import busca
from ahu import consulta_onnx as mod_onnx
from ahu import conversor as mod_conversor
from ahu import estilo as mod_estilo
from ahu import ficha as mod_ficha
from ahu import filtros as mod_filtros
from ahu import graficos as mod_graficos
from ahu import links as mod_links
from ahu import lista_ahu as mod_lista
from ahu import pdf as mod_pdf
from ahu import pessoas as mod_pessoas
from ahu import selecao as mod_selecao
from ahu.consulta import TAMANHO_PAGINA, Consulta, ErroDeLink, executar, url_da_busca
from ahu.dados import ErroDados, carregar

TITULO = "Catálogo AHU-Brasil"
SUBTITULO = ("Busca e triagem nos resumos do Arquivo Histórico Ultramarino sobre o Brasil "
             "colonial, com as cotas para chegar ao documento no Projeto Resgate e no DigitArq")
ABAS = ("Busca", "Pessoas", "Amostras", "Painel", "Conversor", "Sobre e método", "Referências")

# O que a caixa da lista de consulta mostra vazia (WP 3.3a).
# O `display_text` do `LinkColumn` aceita uma expressão regular: o que estiver no grupo de
# captura vira o texto do link. Assim a coluna mostra `PT/AHU/CU/021/0006/00390` e **leva** à
# busca do DigitArq, sem a URL crua aparecer.
RE_REFERENCIA = r"query=(.*?)&"

URL_REPOSITORIO = "https://github.com/saulorrp/catalogo-ahu-cone-sul"
URL_METODO = URL_REPOSITORIO + "/blob/main/docs/METODO_COLETA.md"

EXEMPLO_DE_CONVERSAO = "AHU_ACL_CU_019, Cx. 3, D. 246"
EXEMPLO_DE_LOTE = """PT/AHU/CU/017/0069/06420
AHU_ACL_CU_021, Cx. 6, D. 390
AHU-São Paulo-MGouveia, cx. 2, doc. 126
AHU-Santa Catarina, cx. 13, doc. 46"""

EXEMPLO_DE_LISTA = """# Dia 1
AHU_ACL_CU_017, Cx. 5, D. 487 | ver anexo
PT/AHU/CU/019/0003/00230"""

# §9: proporcional para o texto, monoespaçada só para cotas e números.
# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------

def carregar_dados():
    try:
        return carregar()
    except ErroDados as e:
        st.error(f"**Os dados do app não estão prontos.**\n\n{e}")
        st.stop()

def tamanho_do_banco(dados) -> None:
    """Uma linha de números para dar a medida do banco antes da primeira busca."""
    docs = dados.documentos
    anos = docs["ano_isolado"].dropna()
    por_colecao = docs["colecao"].value_counts()
    colunas = st.columns(4)
    colunas[0].metric("Documentos", milhar(len(docs)))
    colunas[1].metric("Séries (capitanias e afins)", docs["folder"].nunique())
    if len(anos):
        colunas[2].metric("Período", f"{int(anos.min())}–{int(anos.max())}")
    sul = int(por_colecao.get("ahu-sul", 0))
    colunas[3].metric("Enriquecidos (AHU-Sul)", milhar(sul))
    st.caption(
        f"O banco tem duas camadas. A coleção **ahu-sul** ({milhar(sul)} documentos sobre o Sul "
        "e o Prata, a maioria na série Rio de Janeiro) traz metadados enriquecidos: pessoas "
        "identificadas, tipologia revisada e variáveis sociolinguísticas. A coleção "
        f"**ahu-brasil** ({milhar(len(docs) - sul)} documentos) é o catálogo do AHU como "
        "publicado: resumo, data, série e cota. A aba Pessoas e os dados sociolinguísticos "
        "valem só para a ahu-sul; "
        "o filtro Coleção, na Busca, separa as duas."
    )

def milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def consulta_atual(dados) -> Consulta:
    """A consulta vem da URL na primeira carga e do estado da sessão depois (§8.3, A6).

    Na primeira carga a URL também pode trazer a seleção (R2), na chave `sel`. Ela é retirada
    antes de a `Consulta` ler o resto, que recusa parâmetro que não conhece.
    """
    if "consulta" not in st.session_state:
        parametros = dict(st.query_params)
        espelho = parametros.pop(mod_selecao.CHAVE_URL, None)
        try:
            st.session_state.consulta = Consulta.de_url(parametros)
        except ErroDeLink as e:
            st.warning(f"Link de busca não reconhecido ({e}) — começando uma busca nova.")
            st.session_state.consulta = Consulta()
        if espelho:
            recuperada, perdidos = mod_selecao.Selecao.de_url(espelho, dados)
            st.session_state.selecao = recuperada
            if perdidos:
                st.warning(f"{perdidos} documento(s) do link não existem mais neste corpus e "
                           "ficaram de fora da seleção.")
    return st.session_state.consulta


def guardar(consulta: Consulta) -> None:
    """Grava a consulta na sessão e na URL, **preservando o espelho da seleção** (R2)."""
    st.session_state.consulta = consulta
    parametros = consulta.para_url()
    espelho = st.session_state.get("espelho_selecao")
    if espelho:
        parametros[mod_selecao.CHAVE_URL] = espelho
    st.query_params.from_dict(parametros)


@contextlib.contextmanager
def gaveta(rotulo: str, nome: str, expandida: bool = False):
    """Uma **gaveta do EINS**: expansor sem moldura, com o fleurão por interruptor.

    O que a marca como gaveta é a `key` do `st.container` que a embrulha, que o Streamlit
    transforma na classe `st-key-…` — é essa classe que o CSS de `ahu/estilo.py` procura.

    Por que o invólucro, e não a `key` do próprio `st.expander`: no Streamlit 1.55 o expansor só
    ganha identificador (e portanto classe) quando é **stateful**, isto é, quando recebe
    `on_change`. E `on_change` faria cada abrir e fechar reexecutar a página inteira — justamente
    o que o R2 tirou. O `st.container`, ao contrário, emite a classe só por ter chave.
    """
    with st.container(key=mod_estilo.chave_de_gaveta(nome)):
        with st.expander(rotulo, expanded=expandida) as expansor:
            yield expansor


def tabela_estreita(tabela, **extras) -> None:
    """Uma tabela pequena: largura do conteúdo, centrada na página.

    `width="content"` existe no Streamlit 1.55 e faz a tabela parar de se esticar por toda a
    largura. A centragem vem de uma coluna vazia de cada lado — o Streamlit não tem alinhamento
    de elemento, e esta é a forma que não depende de CSS frágil.
    """
    esquerda, meio, direita = st.columns([1, 6, 1])
    del esquerda, direita
    with meio:
        st.dataframe(tabela, width="content", hide_index=True, **extras)


def selecao_atual() -> mod_selecao.Selecao:
    """A seleção vive só em `st.session_state` (§8.1, R4): nada é gravado no servidor."""
    if "selecao" not in st.session_state:
        st.session_state.selecao = mod_selecao.Selecao()
    return st.session_state.selecao


def espelhar_selecao(dados) -> None:
    """Põe a seleção na URL, para ela sobreviver a um recarregamento (R2).

    Chamada dos *callbacks* das caixas, não do corpo da página: é a mudança da seleção que
    precisa ser gravada, e gravar a URL não provoca reexecução.
    """
    selecao = selecao_atual()
    espelho = selecao.para_url(dados)
    st.session_state.espelho_selecao = espelho
    if espelho is None:
        st.query_params.pop(mod_selecao.CHAVE_URL, None)
    elif espelho:
        st.query_params[mod_selecao.CHAVE_URL] = espelho
    else:
        st.query_params.pop(mod_selecao.CHAVE_URL, None)


def ao_marcar(dados, doc_uid: str) -> None:
    """*Callback* da caixa "Selecionar": mexe no estado e **não** reexecuta a página.

    É esta função que faz um clique custar um fragmento em vez de duas buscas completas: ela não
    chama `st.rerun()`, e o fragmento que desenha a lista se redesenha sozinho.
    """
    selecao = selecao_atual()
    chave = f"sel_{doc_uid}"
    marcado = bool(st.session_state.get(chave))
    sobraram = selecao.alternar(doc_uid, marcado)
    if sobraram:
        st.session_state[chave] = False
        st.session_state.aviso_selecao = (
            f"A seleção já tem {mod_selecao.LIMITE} documentos, que é o máximo.")
    else:
        st.session_state.pop("aviso_selecao", None)
    espelhar_selecao(dados)


def resposta_da_consulta(dados, consulta: Consulta):
    """A resposta da busca, **memoizada na sessão** pela consulta (R2).

    Guarda uma entrada só, e a chave ignora a página: paginar não refaz a busca, e marcar uma
    caixa não chega nem a passar por aqui, porque o fragmento não reexecuta o corpo da página.
    Uma entrada basta e mantém a memória limitada — o Community Cloud tem 690 MiB garantidos.

    Não é `st.cache_data` de propósito: ela devolve **cópias** (serializa e desserializa a cada
    acesso), o que numa `Resposta` de 7.051 resultados custaria mais do que a busca que evita.
    """
    chave = tuple(sorted((k, v) for k, v in consulta.para_url().items() if k != "p"))
    guardado = st.session_state.get("resposta_memoizada")
    if guardado is not None and guardado[0] == chave:
        return guardado[1]
    st.session_state.modo_buscado = consulta.modo
    encoder = encoder_preguicoso(dados) if consulta.modo in ("automatico", "conceitual") else None
    resposta = executar(dados, consulta, encoder=encoder)
    st.session_state.resposta_memoizada = (chave, resposta)
    return resposta


def endereco_da_pagina() -> str | None:
    """O endereço onde o app está servido, para o link copiado ser absoluto (§8.3).

    `st.context.url` não existe em toda versão nem no harness de teste; na dúvida, o link sai
    relativo, que ainda funciona ao ser colado na mesma aba.
    """
    try:
        return st.context.url or None
    except Exception:
        return None


def tema() -> str:
    """Tema do navegador, para as figuras do WP 2.4 escolherem os seus tons (§9).

    `st.context.theme` não existe em toda versão nem em todo harness de teste; na dúvida, claro.
    """
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"

# A versão leve, para servidores gratuitos com pouca memória, liga isto (segredo do Streamlit
# Cloud): a busca fica só por palavras e por cota, e o modelo semântico nunca é carregado.
SEM_SEMANTICA = os.environ.get("AHU_SEM_SEMANTICA", "").strip().lower() in ("1", "sim", "true")
AVISO_SEM_SEMANTICA = ("Esta é a versão leve do catálogo: a busca por sentido (modo Conceitual) "
                       "não está disponível aqui, e a busca é feita por palavras e por cota.")


@st.cache_resource(show_spinner="Carregando o modelo semântico (uma vez por sessão)...")
def encoder_de_consulta(_dados):
    """O codificador da consulta, por **ONNX** (R7): nada de `torch` em execução.

    O modelo é baixado na primeira consulta conceitual e fica em cache. O
    `st.cache_resource` garante um por processo.
    """
    return mod_onnx.criar_encoder_de_dados(_dados)


def encoder_preguicoso(dados):
    """Embrulho que só prepara o modelo quando o motor semântico é de fato chamado.

    Sem isto, abrir o app — ou buscar uma cota, que nem usa o semântico — pagaria o
    carregamento à toa, e a carga fria da A10 estouraria. Se o download falhar, a busca
    conceitual avisa e as outras continuam funcionando.
    """
    if SEM_SEMANTICA:
        # Versão leve (Streamlit Community Cloud): sem o modelo, o app cabe nos 690 MiB.
        if st.session_state.get("modo_buscado") == "conceitual":
            st.info(AVISO_SEM_SEMANTICA, icon=":material/info:")
        return None

    def encoder(textos):
        try:
            return encoder_de_consulta(dados)(textos)
        except mod_onnx.ErroDeModelo as e:
            st.error(f"**A busca conceitual não está disponível.** {e}")
            st.stop()

    return encoder

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------

def barra_lateral(dados, consulta: Consulta) -> Consulta:
    """A tabela de filtros da §6.1. Devolve a consulta com o que o usuário mexeu."""
    opcoes = mod_filtros.opcoes(dados)
    f = consulta.filtros

    with st.sidebar:
        st.markdown("### Filtros")
        if opcoes["mostrar_colecao"]:
            f.colecoes = st.multiselect("Coleção", opcoes["colecoes"], default=f.colecoes)

        if opcoes["ano_minimo"] is not None:
            minimo, maximo = opcoes["ano_minimo"], opcoes["ano_maximo"]
            de, ate = st.slider("Período", minimo, maximo,
                                (f.ano_de or minimo, f.ano_ate or maximo))
            f.ano_de = None if de == minimo else de
            f.ano_ate = None if ate == maximo else ate
            f.sem_data = st.checkbox(
                f"Incluir documentos sem data ({opcoes['documentos_sem_data']})", value=f.sem_data)

        f.pastas = st.multiselect("Pasta / região", opcoes["pastas"], default=f.pastas)
        f.tipologias = st.multiselect(
            "Tipologia", [t for t, _ in opcoes["tipologias"]], default=f.tipologias,
            format_func=lambda t: f"{t} ({dict(opcoes['tipologias'])[t]})")

        pessoas = opcoes["pessoas"]
        rotulo_de = {p["pessoa_uid"]: f"{p['nome']} ({p['n_documentos']})" for p in pessoas}
        f.pessoas = st.multiselect(
            "Remetentes", [p["pessoa_uid"] for p in pessoas], default=f.pessoas,
            format_func=lambda u: rotulo_de.get(u, u),
            help="Nome principal, variantes e títulos.")
        f.destinatario = st.text_input("Destinatário contém", value=f.destinatario or "") or None

        with gaveta("Inferências do DeepSeek (começam desligadas)", "deepseek"):
            f.categorias = st.multiselect("Categoria social",
                                          [c for c, _ in opcoes["categorias"]], default=f.categorias)
            f.vetores = st.multiselect("Vetor", [v for v, _ in opcoes["vetores"]],
                                       default=f.vetores)
            escala = opcoes["escala_srsp"]
            usar_srsp = st.checkbox("Filtrar por SRSP",
                                    value=f.srsp_de is not None or f.srsp_ate is not None)
            if usar_srsp:
                de, ate = st.slider("SRSP", escala[0], escala[1],
                                    (f.srsp_de or escala[0], f.srsp_ate or escala[1]), step=1)
                f.srsp_de, f.srsp_ate = de, ate
            else:
                f.srsp_de = f.srsp_ate = None
            faixas = opcoes["faixas_srsp"]
            st.caption(f"Faixas: {faixas['baixo'][0]}–{faixas['baixo'][1]} · "
                       f"{faixas['medio'][0]}–{faixas['medio'][1]} · "
                       f"{faixas['alto'][0]}–{faixas['alto'][1]}.")
            with st.popover("? Saiba mais", width="stretch"):
                st.markdown(
                    "**O que são as inferências:**\n\n"
                    "Estimativas do modelo DeepSeek v3 em regime *zero-shot*, aplicadas ao resumo "
                    "do catálogo (e não ao manuscrito original). Elas produzem a categoria social "
                    "do remetente, o vetor de comunicação, o impacto da tipologia e a pontuação "
                    "SRSP de 1 a 9.\n\n"
                    "**Auditoria do SRSP (preprint §2.1):**\n\n"
                    "MAE 1,13 · r = 0,48 · viés +0,59 · N = 705 · avaliador único. "
                    "Diferenças de ±1 ponto não são distinguíveis.\n\n"
                    "**Mediação notarial:**\n\n"
                    "> \"Essa inferência, contudo, é metodologicamente delicada, por mais que seja uma "
                    "informação fundamental para a linguística diacrônica (uma vez que a escolaridade "
                    "variável desses notários coloniais atuava como um filtro permeável, ora favorecendo "
                    "a hipercorreção, ora permitindo o escape indeliberado da sintaxe local), a sua "
                    "classificação de forma puramente heurística é perigosa. Não é possível atestar tal "
                    "mediação de forma categórica sem um extenso estudo paleográfico e diplomático do "
                    "manuscrito original.\"\n\n"
                    "[Preprint no SciELO Preprints](https://doi.org/10.1590/SciELOPreprints.16461)"
                )

        with gaveta("Origem do remetente", "origem"):
            f.paises_origem = st.multiselect("País (atual)", opcoes["paises_origem"],
                                             default=f.paises_origem)
            f.cidades_origem = st.multiselect("Cidade de nascimento", opcoes["cidades_origem"],
                                              default=f.cidades_origem)
            f.so_confianca_alta = st.checkbox(
                "Só confiança ALTA", value=f.so_confianca_alta,
                help="Desmarcada: ALTA e MÉDIA. Nunca BAIXA nem rejeitada.")

        if st.button("Limpar filtros", width="stretch"):
            guardar(Consulta(texto=consulta.texto, modo=consulta.modo, rigor=consulta.rigor))
            st.rerun()

    consulta.filtros = f
    return consulta

def barra_de_busca(consulta: Consulta) -> Consulta:
    """§9: a busca no topo."""
    with st.form("busca", clear_on_submit=False):
        colunas = st.columns([6, 2, 2, 1])
        texto = colunas[0].text_input(
            "Busca", value=consulta.texto, placeholder='deserção de soldados · "obras do porto" '
                                                       '· AHU_ACL_CU_017, Cx. 5, D. 487',
            label_visibility="collapsed")
        modo = colunas[1].selectbox("Modo", busca.MODOS, index=busca.MODOS.index(consulta.modo),
                                    format_func=str.capitalize, label_visibility="collapsed")
        rigor = colunas[2].selectbox(
            "Rigor", busca.RIGORES, index=busca.RIGORES.index(consulta.rigor),
            format_func=lambda r: f"Rigor {r}", label_visibility="collapsed")
        enviado = colunas[3].form_submit_button("Buscar", width="stretch")

    if enviado:
        from ahu.consulta import asdict_raso
        return Consulta(**{**asdict_raso(consulta), "texto": texto, "modo": modo, "rigor": rigor,
                           "pagina": 1})
    return consulta

# ---------------------------------------------------------------------------
# Resultados, seleção e lista (R2) — tudo dentro de um fragmento
# ---------------------------------------------------------------------------

def desenhar_ficha(dados, doc_uid: str, termos: list[str], resultado=None) -> None:
    """Uma ficha, com a caixa de seleção fora do expansor (§8.1).

    A caixa usa `on_change`: ela mexe na seleção pelo *callback* e **não** chama `st.rerun()`.
    Como a ficha é desenhada dentro de `secao_de_resultados`, que é um fragmento, marcar uma
    caixa redesenha só o fragmento — a busca não é refeita (R2).
    """
    f = mod_ficha.montar(dados, doc_uid, termos)
    selecao = selecao_atual()
    chave = f"sel_{doc_uid}"
    # O estado da caixa segue a seleção: depois de "selecionar tudo" ou de "limpar", a caixa
    # tem de contar a verdade. Escrever o estado antes de instanciar o widget é o jeito
    # suportado de fazer isso.
    desejado = doc_uid in selecao
    if st.session_state.get(chave) != desejado:
        st.session_state[chave] = desejado

    caixa, titulo = st.columns([1, 20])
    caixa.checkbox(
        "Selecionar", key=chave, label_visibility="collapsed",
        on_change=ao_marcar, args=(dados, doc_uid),
        help="Selecionar este documento. A seleção atravessa buscas.")

    with titulo.expander(f.titulo, expanded=False):
        st.markdown(f.resumo_marcado)
        if f.tipologia_a_revisar:
            st.caption("Tipologia fora do vocabulário canônico: a revisar.")

        st.markdown("**Cotas**")
        for etiqueta, valor in f.cotas.items():
            colunas = st.columns([3, 9])
            colunas[0].markdown(f"<span class='rotulo'>{etiqueta}</span>", unsafe_allow_html=True)
            colunas[1].code(valor, language=None)
        destinos = [f"[Ficha no DigitArq]({f.links['digitarq']})"] if f.links.get("digitarq") else []
        destinos.append(f"[Portal do Projeto Resgate]({f.links['resgate']})")
        st.markdown(" · ".join(destinos))

        if f.destinatario:
            st.markdown(f"**Destinatário:** {f.destinatario}")

        if f.pessoas:
            st.markdown("**Remetentes**")
            for p in f.pessoas:
                linha = f"- {p.nome} — *{p.tipo}*"
                if p.exibe_nascimento:
                    linha += f" · nasc. {p.nascimento}" if p.nascimento else ""
                    if p.pais_epoca:
                        linha += f" · à época: {p.pais_epoca}"
                    linha += f" · confiança {p.nivel}"
                    if p.origem_do_dado:
                        linha += f" · origem do dado: {p.origem_do_dado}"
                    if p.fonte:
                        linha += f" · [fonte]({p.fonte})"
                elif p.rotulo:
                    linha += f" · {p.rotulo}"
                st.markdown(linha)

        if resultado is not None:
            evidencia = [f"posição {resultado.posicao}"]
            evidencia += [f"{m}: {p}º" for m, p in sorted(resultado.posicoes.items())]
            if resultado.cosseno is not None:
                evidencia.append(f"cosseno {resultado.cosseno}")
            if resultado.termos:
                evidencia.append("termos: " + ", ".join(resultado.termos))
            st.caption(" · ".join(evidencia))

        if f.inferencias:
            with gaveta(mod_ficha.TITULO_IA, f"ia_{doc_uid}"):
                for etiqueta, valor in f.inferencias.items():
                    st.markdown(f"**{etiqueta}:** {valor}")


@st.fragment
def secao_de_resultados(dados, consulta: Consulta, resposta) -> None:
    """A lista de resultados, a seleção e a lista do AHU, num fragmento só (R2).

    Tudo que depende da seleção vive aqui dentro, e é por isso que um clique numa caixa não
    reexecuta a página: nem a busca, nem a barra lateral, nem as outras abas.
    """
    aviso = st.session_state.pop("aviso_selecao", None)
    if aviso:
        st.warning(aviso)

    painel_de_selecao(dados, resposta)
    painel_da_lista_ahu(dados, resposta)

    if not resposta.total:
        st.info("Nenhum documento. Tente menos filtros, outro modo ou um rigor mais baixo.")
        return

    paginas = max(1, -(-resposta.total // TAMANHO_PAGINA))
    pagina = min(consulta.pagina, paginas)
    for resultado in resposta.pagina(pagina, TAMANHO_PAGINA):
        desenhar_ficha(dados, resultado.doc_uid, resposta.termos_consulta, resultado)

    st.divider()
    st.caption("Resumos e cotas: Arquivo Histórico Ultramarino e Projeto Resgate "
               "\"Barão do Rio Branco\".")


def aba_busca(dados, consulta: Consulta) -> None:
    consulta = barra_de_busca(consulta)
    consulta = barra_lateral(dados, consulta)
    guardar(consulta)

    resposta = resposta_da_consulta(dados, consulta)

    ativos = consulta.filtros.ativos()
    cabecalho = f"**{resposta.total}** documento(s)"
    if resposta.cota:
        cabecalho += " · encontrado(s) pela cota"
    if ativos:
        cabecalho += f" · {len(ativos)} filtro(s) ativo(s)"
    st.markdown(cabecalho)

    for aviso in resposta.avisos:
        st.markdown(f"<div class='aviso-distante'>{aviso}</div>", unsafe_allow_html=True)

    if resposta.truncado_por_motor:
        st.caption(f"Cada motor contribui com os {busca.LIMITE_MOTOR} primeiros documentos.")

    painel_do_link(consulta)

    # Ordenação e página ficam **fora** do fragmento: as duas mudam a `Consulta`, e a consulta
    # tem de ir para a URL para o link copiado reproduzir a mesma lista (A6). A reexecução que
    # elas provocam não refaz a busca, porque `resposta_da_consulta` ignora a página na chave.
    colunas = st.columns([3, 6, 3])
    ordenacoes = ("relevancia", "ano_asc", "ano_desc")
    ordenacao = colunas[0].selectbox(
        "Ordenar por", ordenacoes, index=ordenacoes.index(consulta.ordenacao),
        format_func=lambda o: {"relevancia": "Relevância", "ano_asc": "Ano (mais antigo)",
                               "ano_desc": "Ano (mais recente)"}[o])
    if ordenacao != consulta.ordenacao:
        from ahu.consulta import asdict_raso
        guardar(Consulta(**{**asdict_raso(consulta), "ordenacao": ordenacao, "pagina": 1}))
        st.rerun()

    paginas = max(1, -(-resposta.total // TAMANHO_PAGINA))
    if paginas > 1:
        pagina = colunas[2].number_input("Página", 1, paginas,
                                         min(consulta.pagina, paginas), step=1)
        if pagina != consulta.pagina:
            from ahu.consulta import asdict_raso
            guardar(Consulta(**{**asdict_raso(consulta), "pagina": int(pagina)}))
            st.rerun()

    secao_de_resultados(dados, consulta, resposta)


# ---------------------------------------------------------------------------
# Link da busca (WP 3.4, §8.3, A6)
# ---------------------------------------------------------------------------

def painel_do_link(consulta: Consulta) -> None:
    """"Copiar link desta busca": abrir o link reproduz a busca, a página e a semente (A6)."""
    with gaveta("Copiar link desta busca", "link"):
        st.caption(
            "O link carrega a consulta inteira — texto, modo, rigor, filtros, ordenação, "
            "página e semente. Abrir de novo reproduz a mesma lista, na mesma ordem. Use o ícone "
            "de copiar no canto da caixa.")
        st.code(url_da_busca(consulta, endereco_da_pagina()), language=None)


# ---------------------------------------------------------------------------
# Seleção (WP 3.1, §8.1)
# ---------------------------------------------------------------------------

def painel_de_selecao(dados, resposta) -> None:
    """A seleção, mínima (R2): uma linha por documento, ✕ para tirar, limpar, e o PDF.

    Sem ordenação e sem texto explicativo: o painel é para conferir o que vai ao arquivo, não
    para estudar. Ele vive dentro do fragmento, de modo que o contador e as linhas acompanham
    cada clique sem reexecutar a página.
    """
    selecao = selecao_atual()
    quantos = len(selecao)
    titulo = f"Seleção — {quantos} documento(s)" if quantos else "Seleção vazia"
    with gaveta(titulo, "selecao", expandida=bool(quantos)):
        acoes = st.columns([3, 2, 2])
        if acoes[0].button(f"Selecionar os {resposta.total} do resultado",
                           width="stretch", disabled=not resposta.total,
                           key="selecionar_resultado"):
            sobraram = selecao.adicionar([r.doc_uid for r in resposta.resultados])
            if sobraram:
                st.session_state.aviso_selecao = (
                    f"{sobraram} documento(s) não couberam: o teto é {mod_selecao.LIMITE}.")
            espelhar_selecao(dados)
            st.rerun(scope="fragment")
        if acoes[1].button("Limpar seleção", width="stretch", disabled=not quantos,
                           key="limpar_selecao"):
            selecao.limpar()
            espelhar_selecao(dados)
            st.rerun(scope="fragment")

        if not quantos:
            return

        if st.session_state.get("espelho_selecao") is None:
            st.caption("Seleção grande demais para o link; gere o PDF para guardá-la.")

        for linha in selecao.linhas(dados):
            numero, corpo, tirar = st.columns([1, 18, 1])
            numero.markdown(f"<span class='numero'>{linha['n']}</span>", unsafe_allow_html=True)
            corpo.markdown(
                f"<span class='numero'>{linha['ano'] or '—'}</span> · "
                f"<span class='cota'>{linha['cota'] or '—'}</span><br>"
                f"<span class='rotulo'>{mod_pdf.encurtar(linha['resumo'], 140)}</span>",
                unsafe_allow_html=True)
            if tirar.button("✕", key=f"tirar_{linha['uid']}", help="Tirar da seleção"):
                selecao.remover(linha["uid"])
                st.session_state.pop(f"sel_{linha['uid']}", None)
                espelhar_selecao(dados)
                st.rerun(scope="fragment")

        st.divider()
        pdf_pronto(f"Gerar PDF da seleção ({quantos})", "pdf_selecao",
                   lambda: mod_lista.pdf_da_selecao(dados, selecao),
                   "lista_consulta_AHU.pdf")


# ---------------------------------------------------------------------------
# Lista para consulta no AHU (WP 3.3a)
# ---------------------------------------------------------------------------

def pdf_pronto(rotulo: str, chave: str, montar, nome_do_arquivo: str) -> None:
    """Gera um PDF **sob demanda** e só então oferece o download.

    O `st.download_button` monta o arquivo a cada execução da página; para um PDF isso custaria
    segundos em cada clique, e muito mais no oitavo de núcleo do Community Cloud. Daí os dois
    passos: um botão que monta e guarda na sessão, e o download do que já está pronto.
    """
    if st.button(rotulo, width="stretch", key=f"btn_{chave}"):
        with st.spinner("Montando o PDF..."):
            try:
                st.session_state[chave] = montar()
            except mod_pdf.ErroDeFonte as e:
                st.session_state.pop(chave, None)
                st.error(f"**Não foi possível gerar o PDF.** {e}")
    pronto = st.session_state.get(chave)
    if pronto:
        st.download_button(
            f"Baixar {nome_do_arquivo} ({len(pronto) // 1024} KB)", data=pronto,
            file_name=nome_do_arquivo, mime="application/pdf", width="stretch",
            key=f"baixar_{chave}")


def painel_da_lista_ahu(dados, resposta) -> None:
    """A lista de consulta presencial, do pacote antigo, agora sobre `app_dados/`.

    Duas entradas, como no painel antigo: a seleção do WP 3.1 e uma lista de cotas colada.
    """
    selecao = selecao_atual()
    if "lista_ahu_texto" not in st.session_state:
        st.session_state.lista_ahu_texto = ""

    with gaveta("Colar cotas", "cotas"):
        st.caption(
            "Uma cota por linha, em qualquer formato. `#` abre um grupo, `|` acrescenta nota ou "
            "link. Cotas de outras capitanias também funcionam.")

        enviar = st.columns([3, 3, 2])
        agrupar = enviar[1].selectbox(
            "Agrupar por", (None, "folder", "tipologia", "ano_isolado"),
            format_func=lambda c: {None: "sem grupos", "folder": "Pasta / região",
                                   "tipologia": "Tipologia", "ano_isolado": "Ano"}[c],
            key="agrupar_lista")

        def acrescentar(uids) -> None:
            """Acrescenta ao que já está na caixa, em vez de substituir."""
            novo = mod_lista.lista_de_documentos(dados, uids, agrupar)
            antigo = st.session_state.lista_ahu_texto
            st.session_state.lista_ahu_texto = "\n".join(x for x in (antigo, novo) if x.strip())
            st.rerun()

        if enviar[0].button(f"Enviar a seleção ({len(selecao)})", width="stretch",
                            disabled=not len(selecao)):
            acrescentar(selecao.ordenar(dados, st.session_state.get("ordem_selecao", "chegada")))
        if enviar[2].button(f"Enviar o resultado ({resposta.total})", width="stretch",
                            disabled=not resposta.total):
            acrescentar([r.doc_uid for r in resposta.resultados])

        texto = st.text_area(
            "Documentos (uma cota por linha; `#` abre um grupo, `|` acrescenta nota ou link)",
            value=st.session_state.lista_ahu_texto, height=220, key="area_lista_ahu",
            placeholder=EXEMPLO_DE_LISTA)
        st.session_state.lista_ahu_texto = texto
        if not texto.strip():
            return

        grupos = mod_lista.ler_lista(texto, dados)
        r = mod_lista.relatorio(grupos)
        st.info(f"{r['total']} documento(s) · {r['no_catalogo']} completado(s) pelo catálogo "
                f"AHU-Sul · {r['com_digitarq']} com link do DigitArq · {r['grupos']} grupo(s).")
        if r["sem_referencia"]:
            st.warning("Cotas não reconhecidas (confira a digitação): "
                       + "; ".join(r["sem_referencia"][:10]))

        pdf_pronto("Gerar PDF desta lista", "lista_pdf",
                   lambda: mod_lista.gerar_pdf(grupos, dados),
                   "lista_consulta_AHU.pdf")


# ---------------------------------------------------------------------------
# Pessoas (WP 2.4, §7.1)
# ---------------------------------------------------------------------------

def abrir_pessoa(pessoa_uid: str | None) -> None:
    st.session_state.pessoa_aberta = pessoa_uid


def aba_pessoas(dados) -> None:
    if st.session_state.get("pessoa_aberta"):
        pagina_da_pessoa(dados, st.session_state.pessoa_aberta)
        return

    st.subheader("Pessoas")
    qual = st.radio(
        "Lista", ("Pessoas", "Coletivos e instituições"), horizontal=True,
        label_visibility="collapsed",
        help="Coletivos e instituições ficam numa lista própria: não são pessoas e não têm "
             "biografia (§7.1).")
    if qual == "Pessoas":
        lista_de_pessoas(dados)
    else:
        lista_de_coletivos(dados)


def lista_de_pessoas(dados) -> None:
    opcoes = mod_filtros.opcoes(dados)
    colunas = st.columns([5, 3, 3])
    texto = colunas[0].text_input(
        "Nome, variante ou título", placeholder="Lavradio · Melo e Castro",
        help="Casa por nome principal, variante gráfica ou título.")
    ordem = colunas[1].selectbox(
        "Ordenar por", mod_pessoas.ORDENS,
        format_func=lambda o: mod_pessoas.ROTULOS_ORDEM[o])
    with gaveta("Filtros de origem (só biografias exibíveis)", "origem_pessoas"):
        esquerda, direita = st.columns(2)
        paises = esquerda.multiselect("País (atual)", opcoes["paises_origem"])
        cidades = direita.multiselect("Cidade de nascimento", opcoes["cidades_origem"])
        so_alta = st.checkbox("Só confiança ALTA", help="Desmarcada: ALTA e MÉDIA. Nunca BAIXA.")

    achadas = mod_pessoas.listar(dados, texto=texto, paises=paises, cidades=cidades,
                                 so_confianca_alta=so_alta, ordem=ordem)
    colunas[2].metric("Pessoas", f"{len(achadas)} de {len(dados.pessoas)}")
    if not achadas:
        st.info("Nenhuma pessoa com esse nome ou com essa origem.")
        return

    tabela = mod_pessoas.tabela(achadas)
    escolha = st.dataframe(
        tabela, width="stretch", hide_index=True, on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Fonte": st.column_config.LinkColumn(
                "Fonte", display_text="abrir", width="small",
                help="A página de onde o dado de nascimento foi lido."),
            "Nº de documentos": st.column_config.NumberColumn(width="small"),
        },
    )
    st.caption("Clique numa linha para abrir a página da pessoa.")
    linhas = escolha.selection.rows if hasattr(escolha, "selection") else []
    if linhas:
        abrir_pessoa(achadas[linhas[0]].pessoa_uid)
        st.rerun()


def lista_de_coletivos(dados) -> None:
    st.caption('"Conselho Ultramarino", "Oficiais da Câmara…": remetentes que não são pessoas. '
               "Grafias que diferem só por caixa ou acento aparecem numa linha só.")
    colunas = st.columns([5, 3, 3])
    texto = colunas[0].text_input("Nome", placeholder="Câmara · Conselho", key="busca_coletivo")
    ordem = colunas[1].selectbox(
        "Ordenar por", tuple(mod_pessoas.ORDENS_COLETIVOS),
        format_func=lambda o: mod_pessoas.ROTULOS_ORDEM[o], key="ordem_coletivo")
    coletivos = mod_pessoas.listar_coletivos(dados, texto=texto, ordem=ordem)
    colunas[2].metric("Coletivos e instituições", len(coletivos))
    if not coletivos:
        st.info("Nenhum coletivo ou instituição com esse nome.")
        return
    st.dataframe(mod_pessoas.tabela_de_coletivos(coletivos), width="stretch",
                 hide_index=True)
    papeis = mod_pessoas.contagem_de_remetentes(dados)
    sem_entidade = sum(papeis.get(t, 0) for t in mod_pessoas.TIPOS_SEM_ENTIDADE)
    st.caption(f"Fora destas listas há {sem_entidade} vínculo(s) de remetente sem entidade "
               "(não identificado ou nome incompleto): eles não são pessoa nem instituição.")


def pagina_da_pessoa(dados, pessoa_uid: str) -> None:
    try:
        pagina = mod_pessoas.pagina(dados, pessoa_uid)
    except KeyError:
        st.warning("Essa pessoa não está mais nos dados. Voltando à lista.")
        abrir_pessoa(None)
        return
    p = pagina.pessoa

    if st.button("← Voltar à lista"):
        abrir_pessoa(None)
        st.rerun()

    st.subheader(p.nome)
    st.markdown(f"<span class='rotulo'>{p.n_documentos} documento(s) · atividade "
                f"<span class='numero'>{p.periodo}</span></span>", unsafe_allow_html=True)

    for etiqueta, valores in (("Títulos", p.titulos), ("Cargos", p.cargos),
                              ("Variantes do nome", p.variantes)):
        if valores:
            st.markdown(f"**{etiqueta}:** " + " · ".join(valores))

    # D2: nascimento só para biografia exibível; para o resto, o rótulo da §6.3 e nada mais.
    if p.exibe_nascimento:
        linha = [f"**Nascimento:** {p.nascimento or '—'}", f"confiança {p.nivel}"]
        if p.pais_epoca:
            linha.append(f"à época: {p.pais_epoca}")
        if p.origem_do_dado:
            linha.append(f"origem do dado: {p.origem_do_dado}")
        if p.fonte:
            linha.append(f"[fonte]({p.fonte})")
        st.markdown(" · ".join(linha))
    elif p.rotulo:
        st.markdown(f"**Nascimento:** {p.rotulo}")

    acoes = st.columns([2, 3, 2])
    acoes[0].link_button("Ver documentos", pagina.url_documentos, width="stretch",
                         help="Abre a aba Busca com o filtro desta pessoa aplicado.")
    if acoes[1].button("Gerar amostra diacrônica", width="stretch",
                       help="Ordem cronológica, em lotes de 10, com espaçamento por ano (§7.2)."):
        st.session_state.amostra_pessoa = p.pessoa_uid
        st.session_state.tipo_de_amostra = "Diacrônica por pessoa"
        st.success(f"Amostra de **{p.nome}** preparada. Abra a aba **Amostras** para ver os "
                   "lotes e baixar o plano de coleta.")
    acoes[2].link_button("Reportar erro", pagina.url_issue, width="stretch",
                         help="Abre uma issue no GitHub já preenchida com o dado que o app "
                              "mostra hoje e a sua fonte.")

    st.markdown("#### Linha do tempo")
    if pagina.linha_do_tempo.empty:
        st.caption("Nenhum documento com data.")
    else:
        st.plotly_chart(mod_graficos.barras_por_ano(pagina.linha_do_tempo, tema()),
                        width="stretch")
    if pagina.sem_data:
        st.caption(f"{pagina.sem_data} documento(s) sem data não aparecem no gráfico.")

    esquerda, direita = st.columns(2)
    esquerda.markdown("#### Tipologias")
    esquerda.dataframe(pd.DataFrame(pagina.tipologias, columns=["Tipologia", "Documentos"]),
                       width="stretch", hide_index=True)
    direita.markdown("#### Destinatários")
    direita.dataframe(pd.DataFrame(pagina.destinatarios, columns=["Destinatário", "Documentos"]),
                      width="stretch", hide_index=True)

    with gaveta(f"Documentos em ordem cronológica ({len(pagina.documentos)})",
                "documentos_pessoa"):
        st.dataframe(
            pagina.documentos.drop(columns=["doc_uid"]).rename(columns={
                "ano": "Ano", "tipologia": "Tipologia", "destinatario": "Destinatário",
                "autoria_verificada": "Autoria verificada"}),
            width="stretch", hide_index=True)
        st.caption("A ficha completa de cada documento está na aba Busca, pelo botão "
                   "\"Ver documentos\".")


# ---------------------------------------------------------------------------
# Amostras (WP 3.5, §7.2–§7.4)
# ---------------------------------------------------------------------------

TIPOS_DE_AMOSTRA = ("Diacrônica por pessoa", "Estratificada sobre o resultado")


def aba_amostras(dados, consulta: Consulta) -> None:
    st.subheader("Amostras")
    tipos = list(TIPOS_DE_AMOSTRA)
    escolhido = st.session_state.get("tipo_de_amostra", tipos[0])
    tipo = st.radio("Tipo de amostra", tipos,
                    index=tipos.index(escolhido) if escolhido in tipos else 0,
                    horizontal=True, label_visibility="collapsed")
    st.session_state.tipo_de_amostra = tipo
    if tipo == tipos[0]:
        amostra_diacronica_na_tela(dados, consulta)
    else:
        amostra_estratificada_na_tela(dados, consulta)


def amostra_diacronica_na_tela(dados, consulta: Consulta) -> None:
    st.caption(
        "Os documentos de uma pessoa em ordem cronológica, em lotes de 10 (§7.2). O espaçamento "
        "limita quantos documentos entram por ano, para a amostra não se concentrar num só. "
        "Documento sem data não entra: numa série diacrônica ele não tem onde se colocar.")

    pessoas = mod_pessoas.ordenar(mod_pessoas.todas(dados), "documentos")
    rotulo = {p.pessoa_uid: f"{p.nome} ({p.n_documentos})" for p in pessoas}
    uids = [p.pessoa_uid for p in pessoas]
    aberta = st.session_state.get("amostra_pessoa")
    escolha = st.columns([5, 2, 2])
    pessoa_uid = escolha[0].selectbox(
        "Pessoa", uids, index=uids.index(aberta) if aberta in uids else 0,
        format_func=lambda u: rotulo.get(u, u))
    st.session_state.amostra_pessoa = pessoa_uid
    espacar = escolha[1].checkbox("Espaçar por ano", value=False,
                                  help="No máximo N documentos por ano (§7.2).")
    maximo = escolha[2].number_input("Máximo por ano", 1, 50, 2, step=1,
                                     disabled=not espacar) if espacar else None

    so_verificada = st.checkbox(
        "Só documentos com autoria verificada", value=True,
        help="§7.2: um nome, uma variante ou um título da pessoa aparece na descrição do "
             "documento. A comparação neutraliza acento e grafia antiga, de modo que 'Mello' "
             "casa com 'Melo'.")

    try:
        amostra = mod_amostras.amostra_diacronica(
            dados, pessoa_uid, maximo_por_ano=int(maximo) if espacar else None,
            so_autoria_verificada=so_verificada)
    except mod_amostras.ErroDeAmostra as e:
        st.error(f"**Não foi possível montar a amostra.** {e}")
        return

    r = amostra.relatorio
    colunas = st.columns(4)
    colunas[0].metric("Na amostra", len(amostra))
    colunas[1].metric("Lotes de 10", len(amostra.lotes))
    colunas[2].metric("Anos cobertos", len(r["anos"]))
    colunas[3].metric("Vínculos da pessoa", r["vinculados"])
    descartes = []
    if r["sem_data"]:
        descartes.append(f"{r['sem_data']} sem data")
    if r["sem_autoria_verificada"]:
        descartes.append(f"{r['sem_autoria_verificada']} sem autoria verificada")
    if r["cortados_pelo_maximo_por_ano"]:
        descartes.append(f"{r['cortados_pelo_maximo_por_ano']} cortados pelo espaçamento")
    if descartes:
        st.caption("Fora da amostra: " + " · ".join(descartes) + ".")

    if not len(amostra):
        st.info("Nenhum documento nesta amostra. Tente sem o espaçamento, ou sem exigir "
                "autoria verificada.")
        return

    tabela_estreita(pd.DataFrame(list(r["por_ano"].items()), columns=["Ano", "Documentos"]))
    mostrar_lotes(dados, amostra)
    baixar_amostra(dados, amostra, consulta, "amostra_diacronica")


def amostra_estratificada_na_tela(dados, consulta: Consulta) -> None:
    st.caption(
        "Sorteio sobre o **resultado atual da busca** (busca + filtros da aba Busca), com 1 ou 2 "
        "estratos e N documentos por estrato. A semente fica visível e editável: a mesma semente "
        "dá a mesma amostra (§7.3, A7).")

    resposta = executar(dados, consulta, encoder=None) if consulta.modo == "exata" else None
    if resposta is None:
        encoder = encoder_preguicoso(dados) if consulta.modo in ("automatico",
                                                                 "conceitual") else None
        resposta = executar(dados, consulta, encoder=encoder)
    uids = [r.doc_uid for r in resposta.resultados]
    st.markdown(f"Conjunto atual: **{len(uids)}** documento(s)"
                + (f" · consulta: *{consulta.texto}*" if consulta.texto else
                   " · o corpus com os filtros aplicados"))

    controles = st.columns([5, 2, 2])
    estratos = controles[0].multiselect(
        "Estratos (1 ou 2)", list(mod_amostras.ESTRATOS),
        default=["decada"], max_selections=2,
        format_func=lambda e: mod_amostras.ESTRATOS[e])
    por_estrato = controles[1].number_input("N por estrato", 1, 100, 3, step=1)
    semente = controles[2].number_input("Semente", 0, 10**9,
                                        consulta.semente if consulta.semente else 42, step=1)

    if semente != (consulta.semente or 42):
        from ahu.consulta import asdict_raso
        guardar(Consulta(**{**asdict_raso(consulta), "semente": int(semente)}))

    if not estratos:
        st.info("Escolha ao menos um estrato.")
        return
    if not uids:
        st.info("O resultado atual está vazio: não há o que sortear.")
        return

    try:
        amostra = mod_amostras.amostra_estratificada(
            dados, uids, estratos, int(por_estrato), int(semente))
    except mod_amostras.ErroDeAmostra as e:
        st.error(f"**Não foi possível sortear.** {e}")
        return

    r = amostra.relatorio
    colunas = st.columns(4)
    colunas[0].metric("Sorteados", len(amostra))
    colunas[1].metric("Estratos", r["estratos"])
    colunas[2].metric("Lotes de 10", len(amostra.lotes))
    colunas[3].metric("Estratos insuficientes", len(r["insuficientes"]))
    if r["insuficientes"]:
        st.warning(
            f"{len(r['insuficientes'])} estrato(s) tinham menos de {por_estrato} documento(s) e "
            f"entraram com o que havia: " + "; ".join(r["insuficientes"][:8])
            + ("…" if len(r["insuficientes"]) > 8 else ""))

    st.markdown("#### Relatório por estrato")
    tabela_estreita(mod_amostras.tabela_do_relatorio(amostra))
    mostrar_lotes(dados, amostra)
    baixar_amostra(dados, amostra, consulta, "amostra_estratificada")


def lotes_editados(amostra) -> mod_amostras.LotesEditados:
    """O que o pesquisador mexeu nos lotes, guardado na sessão.

    A chave inclui a identidade da amostra: refazer o sorteio — outra semente, outro estrato —
    joga fora as edições, que é o que se quer. Elas descreviam outra amostra.
    """
    assinatura = (amostra.tipo, amostra.pessoa_uid, tuple(amostra.estratos),
                  amostra.por_estrato, amostra.maximo_por_ano, amostra.semente,
                  tuple(len(lote) for lote in amostra.lotes))
    guardado = st.session_state.get("lotes_editados")
    if guardado is None or guardado[0] != assinatura:
        guardado = (assinatura, mod_amostras.LotesEditados.de_amostra(amostra))
        st.session_state.lotes_editados = guardado
    return guardado[1]


def mostrar_lotes(dados, amostra) -> None:
    """Os lotes: escolher documentos, reordenar, trocar de lote e baixar o PDF de cada um (R9)."""
    editados = lotes_editados(amostra)
    st.markdown("#### Lotes")
    st.caption(
        "Desmarque o que não vai ao arquivo, use ↑ e ↓ para mudar a ordem e **Mover** para "
        "passar um documento a outro lote. O botão de cada lote gera o PDF daquele lote, na "
        "ordem e com os documentos que estiverem marcados.")

    for i, lote in enumerate(editados.lotes):
        estrato = (amostra.estrato_do_lote[i]
                   if i < len(amostra.estrato_do_lote) else None)
        incluidos = editados.incluidos(i)
        titulo = f"Lote {i + 1} — {len(incluidos)} de {len(lote)} documento(s)"
        if estrato:
            titulo += f" · estrato {estrato}"

        with gaveta(titulo, f"lote_{i}", expandida=i == 0):
            if not lote:
                st.caption("Lote vazio.")
                continue

            cabecalho = st.columns([5, 2])
            with cabecalho[1]:
                pdf_pronto(f"PDF do lote {i + 1}", f"pdf_lote_{i}",
                           lambda i=i: mod_lista.pdf_da_selecao(
                               dados, mod_selecao.Selecao(uids=editados.incluidos(i))),
                           f"lista_consulta_AHU_lote_{i + 1}.pdf")

            tabela = mod_amostras.tabela_dos_documentos(dados, lote, editados.excluidos)
            editada = st.data_editor(
                tabela, key=f"editor_lote_{i}", hide_index=True, width="stretch",
                disabled=[c for c in mod_amostras.COLUNAS_DO_LOTE if c != "Incluir"],
                column_config={
                    "Incluir": st.column_config.CheckboxColumn("✓", width="small",
                                                               help="Vai ao PDF deste lote."),
                    "Nº": st.column_config.NumberColumn("Nº", width="small"),
                    # Ano: estreito, que é o que o autor pediu — quatro algarismos e nada mais.
                    "Ano": st.column_config.NumberColumn("Ano", width="small", format="%d"),
                    "Código de referência": st.column_config.LinkColumn(
                        "Código de referência", display_text=RE_REFERENCIA, width="medium",
                        help="Abre a ficha no DigitArq."),
                    "Pasta": st.column_config.TextColumn("Pasta", width="small"),
                    "Tipologia": st.column_config.TextColumn("Tipologia", width="small"),
                    "Resumo": st.column_config.TextColumn("Resumo", width="large"),
                })
            marcados = {linha["Nº"]: bool(linha["Incluir"])
                        for linha in editada.to_dict("records")}
            mudou = False
            for posicao, uid in enumerate(lote):
                desejado = marcados.get(posicao + 1, True)
                if desejado != (uid not in editados.excluidos):
                    editados.alternar(uid, desejado)
                    mudou = True
            if mudou:
                st.rerun(scope="fragment")

            ordem_linhas(dados, editados, i, lote)


def ordem_linhas(dados, editados, i: int, lote: list[str]) -> None:
    """↑ / ↓ e "mover para outro lote", para uma linha escolhida pelo número.

    **Não é arrastar com o rato.** Arrastar linhas exigiria um componente de terceiros com
    pacote JavaScript próprio; o motivo de não o usar está no `docs/HANDOFF.md`.
    """
    del dados
    controles = st.columns([3, 1, 1, 2, 2])
    numeros = list(range(1, len(lote) + 1))
    escolhido = controles[0].selectbox(
        "Linha", numeros, key=f"linha_lote_{i}",
        format_func=lambda n: f"Nº {n}")
    posicao = escolhido - 1

    if controles[1].button("↑", key=f"sobe_{i}", disabled=posicao == 0,
                           help="Subir esta linha"):
        editados.mover(i, posicao, -1)
        st.rerun(scope="fragment")
    if controles[2].button("↓", key=f"desce_{i}", disabled=posicao >= len(lote) - 1,
                           help="Descer esta linha"):
        editados.mover(i, posicao, +1)
        st.rerun(scope="fragment")

    outros = [j for j in range(len(editados.lotes)) if j != i]
    if outros:
        destino = controles[3].selectbox(
            "Para o lote", outros, key=f"destino_{i}",
            format_func=lambda j: f"Lote {j + 1}")
        if controles[4].button("Mover", key=f"move_{i}", width="stretch"):
            editados.trocar_de_lote(i, posicao, destino)
            st.rerun(scope="fragment")


def baixar_amostra(dados, amostra, consulta: Consulta, nome: str) -> None:
    """A amostra sai como plano de coleta do EINS (§7.4, D11) — e só.

    As exportações em CSV e JSON foram retiradas pela D7. O plano do EINS fica porque a D4 o
    exige: é por ele que o Acervo do EINS Lite cria um Processo por código de referência.
    """
    del nome
    st.markdown("#### Levar a amostra")
    colunas = st.columns(2)
    colunas[0].download_button(
        "Plano de coleta (EINS)", data=mod_amostras.plano_em_json(dados, amostra, consulta),
        file_name="plano_coleta.eins.json", mime="application/json",
        width="stretch",
        help="Esquema eins-plano-coleta/1 (§7.4): o Acervo do EINS Lite importa e cria um "
             "Processo por código de referência.")
    if colunas[1].button(f"Pôr os {len(amostra)} na seleção", width="stretch"):
        sobraram = selecao_atual().adicionar(amostra.doc_uids)
        if sobraram:
            st.warning(f"{sobraram} não couberam: o teto da seleção é {mod_selecao.LIMITE}.")
        st.rerun()


# ---------------------------------------------------------------------------
# Painel diatópico (WP 2.4, §9)
# ---------------------------------------------------------------------------

def aba_painel(dados) -> None:
    st.subheader("Painel diatópico")
    por = st.radio("Contar por", mod_pessoas.POR, horizontal=True,
                   format_func=lambda x: f"Por {x}",
                   help="Por documento: quantos documentos têm remetente nascido no país. "
                        "Por pessoa: quantas pessoas distintas nasceram nele.")
    painel = mod_pessoas.painel_diatopico(dados, por)

    unidade = "Documentos" if por == "documento" else "Pessoas"
    colunas = st.columns(3)
    colunas[0].metric(f"{unidade} no corpus", f"{painel.total:,}".replace(",", "."))
    colunas[1].metric("Com origem conhecida",
                      f"{painel.total - painel.sem_origem:,}".replace(",", "."))
    colunas[2].metric("Sem origem conhecida", f"{painel.sem_origem:,}".replace(",", "."))
    st.caption(painel.nota)

    if painel.paises.empty:
        st.info("Nenhuma origem exibível nos dados ativos.")
        return

    # A tabela ao lado do gráfico não é enfeite: no tema claro três tons da paleta ficam abaixo
    # de 3:1 contra o branco, e o guia de visualização exige a leitura por número também.
    grafico, tabela = st.columns([3, 2])
    grafico.plotly_chart(mod_graficos.barras_por_pais(painel, tema()),
                         width="stretch")
    with tabela:
        st.dataframe(painel.paises.rename(columns={"pais": "País", "contagem": unidade}),
                     width="content", hide_index=True)

    st.markdown("#### Por década, com a cor do país de nascimento")
    st.plotly_chart(mod_graficos.barras_por_decada(painel, tema()), width="stretch")
    nomeados = mod_graficos.dobrar_em_outros(painel.paises)
    if len(painel.paises) > len(nomeados):
        st.caption(f"Os {len(nomeados)} países mais frequentes têm cor própria; "
                   "os demais entram como \"Outros\".")
    tabela_estreita(painel.decadas.rename(columns={
        "decada": "Década", "pais": "País", "contagem": unidade}))

    st.info("Em breve: o mapa das cidades de nascimento, com coordenadas da Wikidata "
            "(P625, WP 5.2).")


# ---------------------------------------------------------------------------
# Conversor de cotas
# ---------------------------------------------------------------------------

def enviar_do_conversor_a_lista(cotas: str) -> None:
    """Acrescenta as cotas convertidas à caixa "Colar cotas" da aba Busca.

    É um `on_click`, e não um `if st.button(...)`, porque roda **antes** do script: a caixa da
    lista já foi desenhada quando o script chega a esta aba, e o Streamlit recusa mudar o estado
    de um widget já desenhado na mesma execução.
    """
    antigo = st.session_state.get("lista_ahu_texto", "")
    novo = "\n".join(x for x in (antigo, cotas) if x.strip())
    st.session_state.lista_ahu_texto = novo
    st.session_state.area_lista_ahu = novo


def aba_conversor(dados) -> None:
    """Uma cota na entrada; o tipo reconhecido; as três cotas na saída, prontas para copiar.

    O desenho é o das cotas da ficha (`desenhar_ficha`): rótulo à esquerda, `st.code` à direita,
    que traz o botão de copiar de graça. Várias cotas de uma vez ficam numa gaveta abaixo.
    """
    st.subheader("Conversor de cotas")
    st.caption(
        "Cole uma cota em qualquer formato — código de referência (`PT/AHU/CU/…`), código atual "
        "(`AHU_ACL_CU_…`) ou código antigo (`AHU-<região>, cx., doc.`) — e o tipo é reconhecido "
        "sozinho. Funciona também com cotas que não estão no catálogo. O código antigo só se "
        "converte pela base, salvo Castro e Almeida e Mendes Gouveia, que guardam a numeração "
        "atual.")

    entrada = st.text_input("Cota", key="entrada_conversor", placeholder=EXEMPLO_DE_CONVERSAO)
    if entrada.strip():
        mostrar_conversao(dados, mod_conversor.converter(entrada, dados))

    with gaveta("Converter várias cotas de uma vez", "conversor_lote"):
        conversor_em_lote(dados)

    with gaveta("A cifra série-estado", "cifra"):
        st.caption("As séries que o conversor conhece pelo nome. Uma série fora desta tabela "
                   "ainda é convertida entre referência e código atual, mas sem o nome do estado.")
        tabela_estreita(mod_conversor.tabela_da_cifra())


def mostrar_conversao(dados, conversoes) -> None:
    escolhida = conversoes[0]
    if len(conversoes) > 1:
        escolhida = st.selectbox(
            f"Esta entrada corresponde a {len(conversoes)} documentos atuais",
            conversoes, key="escolha_conversor",
            format_func=lambda c: " · ".join(x for x in (c.resgate or c.referencia,
                                                         mod_conversor.rotulo_do_documento(dados, c.linha))
                                            if x))

    st.markdown(mod_conversor.descrever(escolhida))
    if not escolhida.convertida:
        st.warning(" ".join(escolhida.avisos))
        return

    cotas = {
        mod_conversor.ROTULO_FAMILIA["referencia"]: escolhida.referencia,
        mod_conversor.ROTULO_FAMILIA["resgate"]: escolhida.resgate,
        mod_conversor.ROTULO_FAMILIA["antiga"]: escolhida.antiga,
    }
    if escolhida.linha is not None:
        item = dados.documentos.iloc[escolhida.linha].get("document_id_and_date")
        if isinstance(item, str) and item.strip():
            cotas["Item e data"] = item
    for etiqueta, valor in cotas.items():
        colunas = st.columns([3, 9])
        colunas[0].markdown(f"<span class='rotulo'>{etiqueta}</span>", unsafe_allow_html=True)
        colunas[1].code(valor or "—", language=None)

    for aviso in escolhida.avisos:
        st.caption(aviso)
    links = mod_links.do_documento({"reference_code": escolhida.referencia,
                                    "new_code": escolhida.resgate})
    destinos = [f"[Ficha no DigitArq]({links['digitarq']})"] if links.get("digitarq") else []
    destinos.append(f"[Portal do Projeto Resgate]({links['resgate']})")
    st.markdown(" · ".join(destinos))
    if escolhida.linha is not None:
        st.caption("No catálogo AHU-Sul: " + mod_conversor.rotulo_do_documento(dados, escolhida.linha, 300))

    st.button("Enviar à lista do AHU", key="conversor_para_lista",
              on_click=enviar_do_conversor_a_lista,
              args=(mod_conversor.para_lista_ahu([escolhida]),),
              help="Acrescenta a cota à gaveta \"Colar cotas\" da aba Busca, de onde sai o PDF "
                   "da lista de consulta.")


def conversor_em_lote(dados) -> None:
    texto = st.text_area("Cotas (uma por linha, em qualquer formato)", key="area_conversor",
                         height=160, placeholder=EXEMPLO_DE_LOTE)
    if not texto.strip():
        return
    conversoes = mod_conversor.converter_lista(texto, dados)
    convertidas = sum(c.convertida for c in conversoes)
    st.info(f"{len(conversoes)} linha(s) · {convertidas} convertida(s) · "
            f"{len(conversoes) - convertidas} sem conversão.")
    tabela = mod_conversor.tabela(conversoes)
    # Os avisos vão abaixo da tabela, não numa coluna: oito colunas não cabem ao lado da barra
    # lateral, e um aviso escondido à direita é um aviso que ninguém lê.
    st.dataframe(
        tabela, width="stretch", hide_index=True,
        column_order=[c for c in tabela.columns if c != "Avisos"],
        column_config={
            "DigitArq": st.column_config.LinkColumn("DigitArq", display_text="abrir",
                                                    width="small"),
        })
    avisos = dict.fromkeys(f"**{c.entrada or '(vazia)'}** — {a}"
                           for c in conversoes for a in c.avisos)
    if avisos:
        st.markdown("\n".join(f"- {a}" for a in avisos))
    st.button("Enviar à lista do AHU", key="conversor_lote_para_lista",
              disabled=not convertidas,
              on_click=enviar_do_conversor_a_lista,
              args=(mod_conversor.para_lista_ahu(conversoes),),
              help="Acrescenta as cotas convertidas à gaveta \"Colar cotas\" da aba Busca, "
                   "de onde sai o PDF da lista de consulta.")


# ---------------------------------------------------------------------------
# Sobre e método (§9)
# ---------------------------------------------------------------------------

def aba_sobre(dados) -> None:
    st.subheader("Sobre e método")
    st.markdown(
        "Este app é um instrumento de **triagem documental**: leva de uma pergunta a um conjunto "
        "de cotas que se possa justificar, citar e levar ao arquivo. Ele **não** contém as imagens "
        "dos manuscritos."
    )

    st.markdown("#### 1. Fontes")
    st.markdown(
        "O que são o AHU, o Projeto Resgate, a Biblioteca Nacional e o ACTD, de onde vêm os "
        "resumos e as cotas, e os links para os catálogos e os microfilmes estão na aba ao lado, "
        "**Referências**. O passo a passo da coleta do catálogo está em "
        f"[METODO_COLETA.md]({URL_METODO})."
    )

    st.markdown("#### 2. Como a busca funciona")
    st.markdown(
        "A busca combina três motores e funde os resultados pela **posição** de cada documento "
        "em cada motor, não por uma nota inventada.\n\n"
        "- **Cota.** Se a consulta é uma cota, em qualquer das três formas (Resgate, código de "
        "referência ou cota antiga), a resposta é o documento dessa cota e mais nada. Se a cota "
        "aparece no meio de uma frase, o documento vem primeiro, e o resto da frase segue a "
        "busca normal.\n"
        "- **Palavras.** Procura as palavras da consulta no resumo e nos campos do documento e "
        "ordena os resultados pela frequência e pela raridade de cada termo. Os campos têm pesos: pessoas 2,0, "
        "destinatário 1,5, tipologia 1,5, resumo "
        "1,0 e pasta 0,5. Grafias antigas e acentos são normalizados (`ph`→`f`, `y`→`i`). "
        "Aspas pedem a frase exata.\n"
        "- **Sentido (semântica).** Compara o significado da consulta com o de cada resumo, com o "
        "modelo multilíngue `multilingual-e5-small`. Acha documentos que tratam do assunto com "
        "outras palavras. O **rigor** define até onde, abaixo do documento mais próximo, um "
        "resultado ainda entra. Na versão gratuita do app, este motor fica desligado por falta "
        "de memória.\n\n"
        "Em consultas curtas (até duas palavras) as palavras pesam mais; em consultas longas, o "
        "sentido pesa mais. Cada motor contribui com no máximo "
        f"{busca.LIMITE_MOTOR} documentos, para que uma consulta vaga não devolva o banco "
        "inteiro. Os pesos e os limiares são provisórios até a avaliação com um conjunto de "
        "referência."
    )

    st.markdown("#### 3. A coleção AHU-Sul e as camadas de análise")
    manifesto = dados.manifesto
    contagens = manifesto.get("contagens", {})
    st.markdown(
        "A **ahu-sul** reúne os documentos do AHU sobre o Sul do Brasil e a região do Prata "
        "(Rio Grande, Santa Catarina, Colônia do Sacramento, Montevidéu, Buenos Aires, e os "
        "papéis do Rio de Janeiro e de São Paulo sobre o Sul). Sobre o resumo do catálogo, ela "
        "tem camadas que o resto do banco não tem:\n\n"
        "- **Pessoas.** Os remetentes e destinatários de cada resumo foram extraídos e "
        "consolidados num cadastro de pessoas, coletivos e instituições. Cada pessoa tem um "
        "identificador estável, e variantes de grafia do mesmo nome contam como a mesma pessoa.\n"
        "- **Biografias (Perplexity).** Para cada pessoa, o Perplexity buscou na web o local de "
        "nascimento e a fonte. O resultado foi conferido no Wikidata quando possível e classificado "
        "por confiança. O app só mostra os níveis **alta** e **média**.\n"
        "- **Sociolinguística (DeepSeek).** O modelo DeepSeek v3 leu cada resumo, sem exemplos "
        "prévios (*zero-shot*), e estimou a categoria social do remetente, o vetor da "
        "comunicação (de baixo para cima, de cima para baixo, horizontal), o grau de formulário "
        "da tipologia, a mediação provável de um escrivão e o **SRSP**, um índice de 1 a 9 da "
        "probabilidade de o original trazer sintaxe vernácula. O texto do prompt está registrado "
        "na camada. São estimativas: os filtros que dependem delas começam desligados.\n"
        "- **Tipologia.** A espécie documental (carta, ofício, requerimento…), normalizada e "
        "marcada quando fica fora do vocabulário canônico."
    )
    colunas = st.columns(3)
    colunas[0].metric("Documentos da ahu-sul",
                      milhar(int((dados.documentos["colecao"] == "ahu-sul").sum())))
    colunas[1].metric("Pessoas", milhar(contagens.get("pessoas", 0)))
    colunas[2].metric("Nascimento exibível", contagens.get("nascimento_exibivel", 0))

    srsp = manifesto.get("srsp", {})
    auditoria = srsp.get("auditoria", {})
    if auditoria:
        st.markdown(
            f"**Auditoria do SRSP** (preprint §2.1): escala inteira "
            f"{srsp.get('escala', [1, 9])[0]}–{srsp.get('escala', [1, 9])[1]} · "
            f"MAE {auditoria.get('mae')} · r = {auditoria.get('r')} · "
            f"viés +{auditoria.get('vies')} · N = {auditoria.get('n')} · "
            f"{auditoria.get('avaliadores')} avaliador. {auditoria.get('nota', '')}"
        )
    st.markdown(
        "Método completo: Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do "
        "AHU…* SciELO Preprints, 2026. "
        "[doi:10.1590/SciELOPreprints.16461](https://doi.org/10.1590/SciELOPreprints.16461). "
        "As chamadas pagas (DeepSeek, Perplexity) foram feitas só na construção do banco; o app "
        "não chama nenhum serviço pago."
    )

    st.markdown("##### Camadas e versões ativas")
    st.caption("Lidas do manifesto do corpus, não escritas no código.")
    camadas = manifesto.get("corpus", {}).get("manifesto", {}).get("camadas", {})
    usadas = manifesto.get("corpus", {}).get("camadas", {})
    st.table([
        {"Papel": papel, "Camada": nome, "Versão": camadas.get(nome, {}).get("versao", "—"),
         "Documentos": camadas.get(nome, {}).get("documentos", "—")}
        for papel, nome in usadas.items()
    ])
    st.caption(f"Modelo de vetores: {dados.modelo_embeddings.get('nome') or '—'}")

    st.markdown("#### Licença")
    st.markdown("Código: MIT. Dados derivados: CC BY 4.0 (provisório). "
                "Resumos e cotas: AHU e Projeto Resgate \"Barão do Rio Branco\".")

REFERENCIAS = """
#### Arquivo Histórico Ultramarino (AHU)
Arquivo em Lisboa, criado em 1931, que guarda a documentação da administração ultramarina
portuguesa. O núcleo usado aqui é o fundo do **Conselho Ultramarino** (1642–1833), o órgão que
despachava os assuntos das colônias: requerimentos, ofícios, consultas, cartas de governadores e
de câmaras. Os documentos avulsos estão organizados por **série** (em geral, uma capitania),
**caixa** e **documento**, e é isso que a cota registra. Hoje o AHU faz parte da Direção-Geral do
Livro, dos Arquivos e das Bibliotecas (DGLAB).
- Site do AHU: <https://ahu.dglab.gov.pt/>
- Catálogo do AHU no DigitArq, onde se busca pelo código de referência (`PT/AHU/CU/…`):
  <https://digitarq.ahu.arquivos.pt/>

#### Catálogos do AHU no ACTD
O **Arquivo Científico Tropical Digital** (ACTD), do antigo Instituto de Investigação Científica
Tropical (IICT, hoje Universidade de Lisboa), publica os catálogos dos documentos avulsos do
Conselho Ultramarino, série por série, em PDF com texto. É a **fonte principal** dos resumos e das
cotas da coleção ahu-brasil.
- ACTD: <https://actd.iict.pt/>
- Ficha de uma série: `https://actd.iict.pt/view/actd:CUc<série>` (ex.: `actd:CUc015`, Pernambuco)

#### Projeto Resgate "Barão do Rio Branco"
Projeto brasileiro, criado nos anos 1990, que localizou, catalogou e **microfilmou** a
documentação sobre o Brasil colonial guardada em arquivos estrangeiros, sobretudo no AHU. Os
catálogos, um por capitania, dão a cada documento a cota `AHU_ACL_CU_…`, usada neste app. As
imagens dos microfilmes foram digitalizadas e estão na internet.
- Página do projeto: <https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate>
- Instrumentos de pesquisa (catálogos por capitania):
  <https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/novos-instrumentos-de-pesquisa>
- Códices, avulsos e séries:
  <https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/catalogo-de-verbetes-codices-avulsos-e-series>
- Catálogos temáticos (mulheres, escravidão, irmandades, povos indígenas, semiárido, Rio da Prata):
  <https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/catalogos-tematicos>
- **Documentos digitalizados (microfilmes):**
  <https://resgate.bn.gov.br/docreader/docmulti.aspx?bib=resgate>. O servidor costuma recusar
  acessos de fora do Brasil.

#### Biblioteca Nacional (BN)
A Fundação Biblioteca Nacional, no Rio de Janeiro, mantém o portal do Projeto Resgate e as imagens
digitalizadas. Na sua Hemeroteca Digital estão os *Anais da Biblioteca Nacional*, onde saiu o
inventário de Castro e Almeida dos documentos sobre o Brasil no antigo Arquivo de Marinha e
Ultramar, base das séries "Castro e Almeida" da Bahia e do Rio de Janeiro.
- Biblioteca Nacional: <https://www.gov.br/bn/pt-br>
- BN Digital: <https://bndigital.bn.gov.br/>
- Hemeroteca Digital: <https://memoria.bn.gov.br/>

#### As três formas de cota
- **Cota do Resgate:** `AHU_ACL_CU_017, Cx. 69, D. 6420`. Série 017 (Rio de Janeiro), caixa 69,
  documento 6420.
- **Código de referência:** `PT/AHU/CU/017/0069/06420`. O mesmo documento, no padrão do DigitArq.
- **Cota antiga:** `AHU-Rio de Janeiro, cx. 76, doc. 41`. A numeração anterior ao Resgate, que
  aparece em trabalhos mais antigos. Só se converte pela base, porque a numeração mudou.

#### Este projeto
- Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…* SciELO Preprints, 2026.
  <https://doi.org/10.1590/SciELOPreprints.16461>
- Código e dados: <{repositorio}>
- Método de coleta, transcrição e cura das cotas: <{metodo}>
"""

def aba_referencias() -> None:
    st.subheader("Referências")
    st.markdown(REFERENCIAS.format(repositorio=URL_REPOSITORIO, metodo=URL_METODO))

# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=TITULO, page_icon=":material/history_edu:", layout="wide")
    st.markdown(mod_estilo.CSS, unsafe_allow_html=True)
    st.title(TITULO)
    st.caption(SUBTITULO)
    # No endereço antigo (streamlit.app), o segredo AHU_ENDERECO_NOVO aponta para o novo.
    novo = os.environ.get("AHU_ENDERECO_NOVO")
    if novo:
        st.info(f"O catálogo mudou de endereço e agora cobre o Brasil todo: [{novo}]({novo})",
                icon=":material/open_in_new:")

    dados = carregar_dados()
    tamanho_do_banco(dados)
    consulta = consulta_atual(dados)

    busca_, pessoas_, amostras_, painel_, conversor_, sobre_, referencias_ = st.tabs(ABAS)
    with busca_:
        aba_busca(dados, consulta)
    with pessoas_:
        aba_pessoas(dados)
    with amostras_:
        aba_amostras(dados, consulta)
    with painel_:
        aba_painel(dados)
    with conversor_:
        aba_conversor(dados)
    with sobre_:
        aba_sobre(dados)
    with referencias_:
        aba_referencias()

if __name__ == "__main__":
    main()
