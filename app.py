"""app.py — Classificador AHU-Sul v3: interface (spec §6, §7.1, §9).

Só Streamlit aqui. Toda a lógica está em `ahu/`: busca, filtros, consulta, ficha, links, pessoas
e gráficos.
O app lê **apenas** `app_dados/` (AGENTS.md 3), por `ahu.dados.carregar`, que recusa iniciar com
artefato desalinhado (A4).

Abas (§9): Busca · Pessoas · Amostras · Painel · Sobre e método. Busca e ficha são o WP 2.3;
Pessoas e o painel diatópico, o WP 2.4. As que dependem da F3 e da F4 aparecem como "em breve",
para que a navegação já seja a definitiva.

Rodar:
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ahu import amostras as mod_amostras
from ahu import busca
from ahu import ficha as mod_ficha
from ahu import filtros as mod_filtros
from ahu import graficos as mod_graficos
from ahu import lista_ahu as mod_lista
from ahu import pdf as mod_pdf
from ahu import pessoas as mod_pessoas
from ahu import selecao as mod_selecao
from ahu.consulta import TAMANHO_PAGINA, Consulta, ErroDeLink, executar, url_da_busca
from ahu.dados import ErroDados, carregar

TITULO = "Classificador AHU-Sul"
SUBTITULO = "Triagem documental sobre o catálogo do Arquivo Histórico Ultramarino (coleção AHU-Sul)"
ABAS = ("Busca", "Pessoas", "Amostras", "Painel", "Sobre e método")

# O que a caixa da lista de consulta mostra vazia (WP 3.3a).
EXEMPLO_DE_LISTA = """# Dia 1
AHU_ACL_CU_017, Cx. 5, D. 487 | ver anexo
PT/AHU/CU/019/0003/00230"""

# §9: proporcional para o texto, monoespaçada só para cotas e números.
ESTILO = """
<style>
  .cota, .numero { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .cota { font-size: 0.92rem; }
  .rotulo { color: #5f6368; font-size: 0.85rem; }
  .aviso-distante { background: #fff6e5; border-left: 4px solid #e8a33d; padding: 0.6rem 0.8rem;
                    border-radius: 4px; margin: 0.4rem 0 0.8rem; }
</style>
"""

# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------

def carregar_dados():
    try:
        return carregar()
    except ErroDados as e:
        st.error(f"**Os dados do app não estão prontos.**\n\n{e}")
        st.stop()

def consulta_atual() -> Consulta:
    """A consulta vem da URL na primeira carga e do estado da sessão depois (§8.3, A6)."""
    if "consulta" not in st.session_state:
        try:
            st.session_state.consulta = Consulta.de_url(dict(st.query_params))
        except ErroDeLink as e:
            st.warning(f"Link de busca não reconhecido ({e}) — começando uma busca nova.")
            st.session_state.consulta = Consulta()
    return st.session_state.consulta

def guardar(consulta: Consulta) -> None:
    st.session_state.consulta = consulta
    st.query_params.from_dict(consulta.para_url())


def selecao_atual() -> mod_selecao.Selecao:
    """A seleção vive só em `st.session_state` (§8.1, R4): nada é gravado no servidor."""
    if "selecao" not in st.session_state:
        st.session_state.selecao = mod_selecao.Selecao()
    return st.session_state.selecao


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

@st.cache_resource(show_spinner="Carregando o modelo semântico (uma vez por sessão)...")
def encoder_de_consulta(_dados):
    return busca.criar_encoder_consulta(_dados)


def encoder_preguicoso(dados):
    """Embrulho que só carrega o modelo quando o motor semântico é de fato chamado.

    Sem isto, abrir o app — ou buscar uma cota, que nem usa o semântico — pagaria o carregamento
    do e5 à toa, e a carga fria da A10 estouraria.
    """
    def encoder(textos):
        return encoder_de_consulta(dados)(textos)

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
            "Pessoa (remetente)", [p["pessoa_uid"] for p in pessoas], default=f.pessoas,
            format_func=lambda u: rotulo_de.get(u, u),
            help="Nome principal, variantes e títulos.")
        f.destinatario = st.text_input("Destinatário contém", value=f.destinatario or "") or None

        with st.expander("Inferências do DeepSeek (começam desligadas)"):
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
                       f"{faixas['alto'][0]}–{faixas['alto'][1]}. {opcoes['nota_srsp']}")

        with st.expander("Origem do remetente"):
            f.paises_origem = st.multiselect("País (atual)", opcoes["paises_origem"],
                                             default=f.paises_origem)
            f.cidades_origem = st.multiselect("Cidade de nascimento", opcoes["cidades_origem"],
                                              default=f.cidades_origem)
            f.so_confianca_alta = st.checkbox(
                "Só confiança ALTA", value=f.so_confianca_alta,
                help="Desmarcada: ALTA e MÉDIA. Nunca BAIXA nem rejeitada.")

        if st.button("Limpar filtros", use_container_width=True):
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
        enviado = colunas[3].form_submit_button("Buscar", use_container_width=True)

    if enviado:
        from ahu.consulta import asdict_raso
        return Consulta(**{**asdict_raso(consulta), "texto": texto, "modo": modo, "rigor": rigor,
                           "pagina": 1})
    return consulta

# ---------------------------------------------------------------------------
# Resultados e ficha
# ---------------------------------------------------------------------------

def desenhar_ficha(dados, doc_uid: str, termos: list[str], resultado=None) -> None:
    f = mod_ficha.montar(dados, doc_uid, termos)
    selecao = selecao_atual()

    # §8.1: a caixa fica **fora** do expansor, senão ela só existiria com a ficha aberta.
    caixa, titulo = st.columns([1, 20])
    marcado = caixa.checkbox(
        "Selecionar", value=doc_uid in selecao, key=f"sel_{doc_uid}", label_visibility="collapsed",
        help="Selecionar este documento. A seleção atravessa buscas.")
    if marcado != (doc_uid in selecao):
        sobraram = selecao.alternar(doc_uid, marcado)
        if sobraram:
            st.warning(f"A seleção já tem {mod_selecao.LIMITE} documentos, o máximo.")
        st.rerun()

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
            st.markdown("**Pessoas (remetentes)**")
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
            with st.expander(mod_ficha.TITULO_IA, expanded=False):
                for etiqueta, valor in f.inferencias.items():
                    st.markdown(f"**{etiqueta}:** {valor}")
                st.caption(mod_filtros.NOTA_SRSP)

def aba_busca(dados, consulta: Consulta) -> None:
    consulta = barra_de_busca(consulta)
    consulta = barra_lateral(dados, consulta)
    guardar(consulta)

    encoder = encoder_preguicoso(dados) if consulta.modo in ("automatico", "conceitual") else None
    resposta = executar(dados, consulta, encoder=encoder)

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
    painel_de_selecao(dados, consulta, resposta)
    painel_da_lista_ahu(dados, resposta)

    colunas = st.columns([3, 6, 3])
    ordenacao = colunas[0].selectbox(
        "Ordenar por", ("relevancia", "ano_asc", "ano_desc"),
        index=("relevancia", "ano_asc", "ano_desc").index(consulta.ordenacao),
        format_func=lambda o: {"relevancia": "Relevância", "ano_asc": "Ano (mais antigo)",
                               "ano_desc": "Ano (mais recente)"}[o])
    if ordenacao != consulta.ordenacao:
        from ahu.consulta import asdict_raso
        guardar(Consulta(**{**asdict_raso(consulta), "ordenacao": ordenacao, "pagina": 1}))
        st.rerun()

    paginas = max(1, -(-resposta.total // TAMANHO_PAGINA))
    pagina = min(consulta.pagina, paginas)
    if paginas > 1:
        pagina = colunas[2].number_input("Página", 1, paginas, pagina, step=1)
        if pagina != consulta.pagina:
            from ahu.consulta import asdict_raso
            guardar(Consulta(**{**asdict_raso(consulta), "pagina": int(pagina)}))
            st.rerun()

    if not resposta.total:
        st.info("Nenhum documento. Tente menos filtros, outro modo ou um rigor mais baixo.")
        return

    for resultado in resposta.pagina(pagina, TAMANHO_PAGINA):
        desenhar_ficha(dados, resultado.doc_uid, resposta.termos_consulta, resultado)

    st.divider()
    st.caption("Resumos e cotas: Arquivo Histórico Ultramarino e Projeto Resgate "
               "\"Barão do Rio Branco\".")


# ---------------------------------------------------------------------------
# Link da busca (WP 3.4, §8.3, A6)
# ---------------------------------------------------------------------------

def painel_do_link(consulta: Consulta) -> None:
    """"Copiar link desta busca": abrir o link reproduz a busca, a página e a semente (A6)."""
    with st.expander("Copiar link desta busca", expanded=False):
        st.caption(
            "O link carrega a consulta inteira — texto, modo, rigor, filtros, ordenação, "
            "página e semente. Abrir de novo reproduz a mesma lista, na mesma ordem. Use o ícone "
            "de copiar no canto da caixa.")
        st.code(url_da_busca(consulta, endereco_da_pagina()), language=None)


# ---------------------------------------------------------------------------
# Seleção (WP 3.1, §8.1)
# ---------------------------------------------------------------------------

def painel_de_selecao(dados, consulta: Consulta, resposta) -> None:
    """Ver, ordenar e esvaziar a seleção (§8.1)."""
    del consulta
    selecao = selecao_atual()
    titulo = (f"Seleção — {len(selecao)} documento(s)" if len(selecao)
              else "Seleção — vazia")
    with st.expander(titulo, expanded=False):
        st.caption(
            f"A seleção atravessa buscas e cabe até {mod_selecao.LIMITE} documentos. Ela vive "
            "só nesta sessão do navegador: nada é guardado no servidor.")

        acoes = st.columns(2)
        if acoes[0].button(f"Selecionar os {resposta.total} do resultado",
                           use_container_width=True, disabled=not resposta.total,
                           help="Acrescenta todos os documentos desta busca, até o teto."):
            sobraram = selecao.adicionar([r.doc_uid for r in resposta.resultados])
            if sobraram:
                st.warning(f"{sobraram} documento(s) não couberam: o teto é "
                           f"{mod_selecao.LIMITE}.")
            st.rerun()
        if acoes[1].button("Esvaziar a seleção", use_container_width=True,
                           disabled=not len(selecao)):
            selecao.limpar()
            st.rerun()

        if not len(selecao):
            return
        ordem = st.selectbox("Ordenar a seleção por", mod_selecao.ORDENS,
                             format_func=lambda o: mod_selecao.ROTULOS_ORDEM[o],
                             key="ordem_selecao")
        st.dataframe(selecao.tabela(dados, ordem), use_container_width=True, hide_index=True)



# ---------------------------------------------------------------------------
# Lista para consulta no AHU (WP 3.3a)
# ---------------------------------------------------------------------------

def pdf_pronto(rotulo: str, chave: str, montar, nome_do_arquivo: str) -> None:
    """Gera um PDF **sob demanda** e só então oferece o download.

    O `st.download_button` monta o arquivo a cada execução da página; para um PDF isso custaria
    segundos em cada clique, e muito mais no oitavo de núcleo do Community Cloud. Daí os dois
    passos: um botão que monta e guarda na sessão, e o download do que já está pronto.
    """
    if st.button(rotulo, use_container_width=True, key=f"btn_{chave}"):
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
            file_name=nome_do_arquivo, mime="application/pdf", use_container_width=True,
            key=f"baixar_{chave}")


def painel_da_lista_ahu(dados, resposta) -> None:
    """A lista de consulta presencial, do pacote antigo, agora sobre `app_dados/`.

    Duas entradas, como no painel antigo: a seleção do WP 3.1 e uma lista de cotas colada.
    """
    selecao = selecao_atual()
    if "lista_ahu_texto" not in st.session_state:
        st.session_state.lista_ahu_texto = ""

    with st.expander("Lista para consulta no AHU (PDF)", expanded=False):
        st.caption(
            "O PDF que se anexa ao pedido de consulta: A4 paisagem, uma linha por documento, com "
            "cota, código de referência, link do DigitArq, data, tipologia, remetente, resumo do "
            "catálogo e duas caixas para marcar à mão. Cotas de outras capitanias também "
            "funcionam — o código e o link saem da própria cota; o que falta, nesse caso, é só o "
            "que vem do catálogo.")

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

        if enviar[0].button(f"Enviar a seleção ({len(selecao)})", use_container_width=True,
                            disabled=not len(selecao)):
            acrescentar(selecao.ordenar(dados, st.session_state.get("ordem_selecao", "chegada")))
        if enviar[2].button(f"Enviar o resultado ({resposta.total})", use_container_width=True,
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

        campos = st.columns(2)
        cabecalho = {
            "titulo": campos[0].text_input(
                "Título", "Documentos para consulta presencial e registro fotográfico"),
            "pesquisador": campos[0].text_input("Pesquisador(a)"),
            "instituicao": campos[0].text_input("Instituição / vínculo"),
            "projeto": campos[1].text_input("Projeto"),
            "datas_consulta": campos[1].text_input("Datas agendadas no CRAV"),
            "observacao": campos[1].text_input("Observação (opcional)"),
        }
        pdf_pronto("Gerar a lista em PDF", "lista_pdf",
                   lambda: mod_lista.gerar_pdf(grupos, dados, cabecalho),
                   "lista_documentos_AHU.pdf")


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
    with st.expander("Filtros de origem (só biografias exibíveis)"):
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
        tabela, use_container_width=True, hide_index=True, on_select="rerun",
        selection_mode="single-row",
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
    st.dataframe(mod_pessoas.tabela_de_coletivos(coletivos), use_container_width=True,
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
    acoes[0].link_button("Ver documentos", pagina.url_documentos, use_container_width=True,
                         help="Abre a aba Busca com o filtro desta pessoa aplicado.")
    if acoes[1].button("Gerar amostra diacrônica", use_container_width=True,
                       help="Ordem cronológica, em lotes de 10, com espaçamento por ano (§7.2)."):
        st.session_state.amostra_pessoa = p.pessoa_uid
        st.session_state.tipo_de_amostra = "Diacrônica por pessoa"
        st.success(f"Amostra de **{p.nome}** preparada. Abra a aba **Amostras** para ver os "
                   "lotes e baixar o plano de coleta.")
    acoes[2].link_button("Reportar erro", pagina.url_issue, use_container_width=True,
                         help="Abre uma issue no GitHub já preenchida com o dado que o app "
                              "mostra hoje e a sua fonte.")

    st.markdown("#### Linha do tempo")
    if pagina.linha_do_tempo.empty:
        st.caption("Nenhum documento com data.")
    else:
        st.plotly_chart(mod_graficos.barras_por_ano(pagina.linha_do_tempo, tema()),
                        use_container_width=True)
    if pagina.sem_data:
        st.caption(f"{pagina.sem_data} documento(s) sem data não aparecem no gráfico.")

    esquerda, direita = st.columns(2)
    esquerda.markdown("#### Tipologias")
    esquerda.dataframe(pd.DataFrame(pagina.tipologias, columns=["Tipologia", "Documentos"]),
                       use_container_width=True, hide_index=True)
    direita.markdown("#### Destinatários")
    direita.dataframe(pd.DataFrame(pagina.destinatarios, columns=["Destinatário", "Documentos"]),
                      use_container_width=True, hide_index=True)

    with st.expander(f"Documentos em ordem cronológica ({len(pagina.documentos)})"):
        st.dataframe(
            pagina.documentos.drop(columns=["doc_uid"]).rename(columns={
                "ano": "Ano", "tipologia": "Tipologia", "destinatario": "Destinatário",
                "autoria_verificada": "Autoria verificada"}),
            use_container_width=True, hide_index=True)
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

    st.dataframe(pd.DataFrame(list(r["por_ano"].items()), columns=["Ano", "Documentos"]),
                 use_container_width=True, hide_index=True)
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
    st.dataframe(mod_amostras.tabela_do_relatorio(amostra), use_container_width=True,
                 hide_index=True)
    mostrar_lotes(dados, amostra)
    baixar_amostra(dados, amostra, consulta, "amostra_estratificada")


def mostrar_lotes(dados, amostra) -> None:
    st.markdown("#### Lotes")
    for numero, lote in enumerate(amostra.lotes, start=1):
        estrato = (amostra.estrato_do_lote[numero - 1]
                   if numero - 1 < len(amostra.estrato_do_lote) else None)
        titulo = f"Lote {numero} — {len(lote)} documento(s)"
        if estrato:
            titulo += f" · estrato {estrato}"
        with st.expander(titulo, expanded=numero == 1):
            st.dataframe(mod_amostras.tabela_dos_documentos(dados, lote),
                         use_container_width=True, hide_index=True)


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
        use_container_width=True,
        help="Esquema eins-plano-coleta/1 (§7.4): o Acervo do EINS Lite importa e cria um "
             "Processo por código de referência.")
    if colunas[1].button(f"Pôr os {len(amostra)} na seleção", use_container_width=True):
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
                         use_container_width=True)
    tabela.dataframe(painel.paises.rename(columns={"pais": "País", "contagem": unidade}),
                     use_container_width=True, hide_index=True)

    st.markdown("#### Por década, com a cor do país de nascimento")
    st.plotly_chart(mod_graficos.barras_por_decada(painel, tema()), use_container_width=True)
    nomeados = mod_graficos.dobrar_em_outros(painel.paises)
    if len(painel.paises) > len(nomeados):
        st.caption(f"Os {len(nomeados)} países mais frequentes têm cor própria; "
                   "os demais entram como \"Outros\".")
    st.dataframe(painel.decadas.rename(columns={
        "decada": "Década", "pais": "País", "contagem": unidade}),
        use_container_width=True, hide_index=True)

    st.info("Em breve: o mapa das cidades de nascimento, com coordenadas da Wikidata "
            "(P625, WP 5.2).")


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
    manifesto = dados.manifesto
    contagens = manifesto.get("contagens", {})
    colunas = st.columns(4)
    colunas[0].metric("Documentos", f"{contagens.get('documentos', 0):,}".replace(",", "."))
    colunas[1].metric("Pessoas", f"{contagens.get('pessoas', 0):,}".replace(",", "."))
    colunas[2].metric("Nascimento exibível", contagens.get("nascimento_exibivel", 0))
    colunas[3].metric("Modelo", (dados.modelo_embeddings.get("nome") or "—").split("/")[-1])

    st.markdown("#### Camadas e versões ativas")
    st.caption("Lidas do manifesto do corpus, não escritas no código.")
    camadas = manifesto.get("corpus", {}).get("manifesto", {}).get("camadas", {})
    usadas = manifesto.get("corpus", {}).get("camadas", {})
    st.table([
        {"Papel": papel, "Camada": nome, "Versão": camadas.get(nome, {}).get("versao", "—"),
         "Documentos": camadas.get(nome, {}).get("documentos", "—")}
        for papel, nome in usadas.items()
    ])

    srsp = manifesto.get("srsp", {})
    auditoria = srsp.get("auditoria", {})
    if auditoria:
        st.markdown("#### Auditoria do SRSP (preprint §2.1)")
        st.markdown(
            f"Escala inteira {srsp.get('escala', [1, 9])[0]}–{srsp.get('escala', [1, 9])[1]} · "
            f"MAE {auditoria.get('mae')} · r = {auditoria.get('r')} · "
            f"viés +{auditoria.get('vies')} · N = {auditoria.get('n')} · "
            f"{auditoria.get('avaliadores')} avaliador. {auditoria.get('nota', '')}"
        )

    st.markdown("#### O que sai daqui")
    st.markdown(
        "Uma coisa só: a **lista para consulta no AHU**, em PDF, montada da seleção ou de uma "
        "lista de cotas colada. A aba Amostras entrega também o plano de coleta do EINS Lite. "
        "As exportações em CSV e JSON e o dossiê em PDF foram retirados em 09/10/2026, por não "
        "serem usados.")

    st.markdown("#### Avaliação da busca")
    st.info("Em breve: a tabela v2 × v3 com nDCG@10, Recall@20 e MRR (WP 4.1).")

    st.markdown("#### Como citar")
    st.markdown(
        "Rocha, S. R. P. *Classificador AHU-Sul v3* (software e dados), 2026. "
        "A citação com o DOI do Zenodo entra na versão 3.0.0 (WP 4.4)."
    )
    st.markdown("#### Licença")
    st.markdown("Código: MIT. Dados derivados: CC BY 4.0 (provisório). "
                "Resumos e cotas: AHU e Projeto Resgate \"Barão do Rio Branco\".")

# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=TITULO, page_icon="📜", layout="wide")
    st.markdown(ESTILO, unsafe_allow_html=True)
    st.title(TITULO)
    st.caption(SUBTITULO)

    dados = carregar_dados()
    consulta = consulta_atual()

    busca_, pessoas_, amostras_, painel_, sobre_ = st.tabs(ABAS)
    with busca_:
        aba_busca(dados, consulta)
    with pessoas_:
        aba_pessoas(dados)
    with amostras_:
        aba_amostras(dados, consulta)
    with painel_:
        aba_painel(dados)
    with sobre_:
        aba_sobre(dados)

if __name__ == "__main__":
    main()
