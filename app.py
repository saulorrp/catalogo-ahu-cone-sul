"""app.py — Classificador AHU-Sul v3 (WP 2.3): interface de busca e ficha (spec §6, §9).

Só Streamlit aqui. Toda a lógica está em `ahu/`: busca, filtros, consulta, ficha, links.
O app lê **apenas** `app_dados/` (AGENTS.md 3), por `ahu.dados.carregar`, que recusa iniciar com
artefato desalinhado (A4).

Abas (§9): Busca · Pessoas · Amostras · Painel · Sobre e método. As que dependem da F3 e da F4
aparecem como "em breve", para que a navegação já seja a definitiva.

Rodar:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from ahu import busca
from ahu import ficha as mod_ficha
from ahu import filtros as mod_filtros
from ahu.consulta import TAMANHO_PAGINA, Consulta, ErroDeLink, executar
from ahu.dados import ErroDados, carregar

TITULO = "Classificador AHU-Sul"
SUBTITULO = "Triagem documental sobre o catálogo do Arquivo Histórico Ultramarino (coleção AHU-Sul)"
ABAS = ("Busca", "Pessoas", "Amostras", "Painel", "Sobre e método")

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
    """Lentes e a tabela de filtros da §6.1. Devolve a consulta com o que o usuário mexeu."""
    opcoes = mod_filtros.opcoes(dados)
    f = consulta.filtros

    with st.sidebar:
        st.markdown("### Lente metodológica")
        nomes = list(mod_filtros.LENTES)
        rotulos = [mod_filtros.LENTES[n]["rotulo"] for n in nomes]
        atual = consulta.lente or "livre"
        escolhida = nomes[st.radio(
            "Lente", range(len(nomes)), index=nomes.index(atual) if atual in nomes else 0,
            format_func=lambda i: rotulos[i], label_visibility="collapsed",
            help="A lente só preenche os filtros abaixo, que ficam editáveis.",
        )]
        st.caption(mod_filtros.LENTES[escolhida]["descricao"])
        if escolhida != atual:
            nova = consulta.com_lente(dados, None if escolhida == "livre" else escolhida)
            guardar(nova)
            st.rerun()

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
    with st.expander(f.titulo, expanded=False):
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
# Abas que ainda não existem (F3/F4)
# ---------------------------------------------------------------------------

def em_breve(titulo: str, o_que: str, pacote: str) -> None:
    st.subheader(titulo)
    st.info(f"Em breve: {o_que} ({pacote}).")

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
        em_breve("Pessoas", "lista de pessoas, página da pessoa e painel diatópico", "WP 2.4")
    with amostras_:
        em_breve("Amostras", "amostra diacrônica por pessoa e amostra estratificada com semente",
                 "WP 3.5")
    with painel_:
        em_breve("Painel", "contagem por documento ou por pessoa, por década e por país de "
                           "nascimento", "WP 2.4 e WP 5.2")
    with sobre_:
        aba_sobre(dados)

if __name__ == "__main__":
    main()
