"""graficos.py — WP 2.4: as figuras do painel e da página da pessoa (spec §9).

Sem Streamlit: são funções que devolvem figuras do Plotly, e por isso um teste pode conferir as
cores, a ordem da legenda e as séries sem subir interface.

Três figuras, e cada uma tem um trabalho diferente:

* **`barras_por_pais`** — ranking de magnitude: uma série só, uma cor só, barras horizontais com o
  número ao lado. Sem legenda, porque o título já diz o que é.
* **`barras_por_decada`** — identidade ao longo do tempo: barras empilhadas por década, coloridas
  pelo país de nascimento. Paleta categórica em **ordem fixa** (o país mais frequente pega sempre
  o primeiro tom), com legenda, e no máximo `LIMITE_CATEGORIAS` países: o resto vira "Outros".
  Cor segue o país, nunca a posição dele no gráfico — mudar o filtro não repinta os sobreviventes.
* **`barras_por_ano`** — a linha do tempo de uma pessoa: uma série, uma cor, um ano por barra.

A paleta é a de referência do guia de visualização, validada nos dois temas contra as superfícies
do Streamlit (`#ffffff` e `#0e1117`): faixa de luminosidade, piso de croma, separação para
protanopia e deuteranopia (ΔE OKLab ≥ 8 nos pares vizinhos) e contraste. No tema claro, três tons
ficam abaixo de 3:1 contra o branco; por isso **a interface mostra sempre a tabela ao lado do
gráfico** — é a compensação que o guia exige, e também o que a §9 pede para poder conferir número.

"Outros" e "sem origem conhecida" não são países: ficam no cinza neutro, fora da paleta
categórica, para não competirem com quem é dado de verdade.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ahu.pessoas import SEM_ORIGEM

TEMAS = ("light", "dark")
OUTROS = "Outros"
LIMITE_CATEGORIAS = 7       # além disto, a cor deixa de distinguir: o resto vira "Outros"

# Paleta categórica, em ordem fixa. Os dois temas são escolhidos, não um inverso automático do
# outro. Validada com `validate_palette.py` nos dois modos (ver o cabeçalho).
PALETAS = {
    "light": {
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"),
        "superficie": "#ffffff",
        "tinta": "#0b0b0b",
        "tinta_fraca": "#52514e",
        "apagado": "#898781",
        "grade": "#e1e0d9",
        "eixo": "#c3c2b7",
    },
    "dark": {
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
        "superficie": "#0e1117",
        "tinta": "#ffffff",
        "tinta_fraca": "#c3c2b7",
        "apagado": "#898781",
        "grade": "#2c2c2a",
        "eixo": "#383835",
    },
}

FONTE = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def paleta(tema: str = "light") -> dict:
    if tema not in PALETAS:
        raise ValueError(f"Tema desconhecido: {tema!r}. Use um de {', '.join(TEMAS)}.")
    return PALETAS[tema]


def dobrar_em_outros(contagem: pd.DataFrame, coluna: str = "pais",
                     limite: int = LIMITE_CATEGORIAS) -> list[str]:
    """As categorias que ganham cor própria, da maior para a menor. O resto é "Outros".

    `contagem` é o ranking já ordenado (`PainelDiatopico.paises`). O cinza neutro —
    "Outros" e "sem origem conhecida" — não ocupa tom da paleta.
    """
    nomes = [str(x) for x in contagem[coluna].tolist() if str(x) != SEM_ORIGEM]
    return nomes[:limite]


def cores(categorias: list[str], tema: str = "light") -> dict[str, str]:
    """Mapa categoria -> cor, em ordem fixa. Fora da lista, cinza neutro."""
    p = paleta(tema)
    mapa = {nome: p["series"][i % len(p["series"])] for i, nome in enumerate(categorias)}
    mapa[OUTROS] = p["apagado"]
    mapa[SEM_ORIGEM] = p["apagado"]
    return mapa


def _layout(fig: go.Figure, tema: str, altura: int, titulo_x: str = "",
            titulo_y: str = "") -> go.Figure:
    p = paleta(tema)
    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=p["superficie"],
        plot_bgcolor=p["superficie"],
        font=dict(family=FONTE, color=p["tinta_fraca"], size=13),
        bargap=0.25,
        barcornerradius=4,          # §marks: as pontas da barra arredondadas em 4px
        showlegend=False,
        hoverlabel=dict(font=dict(family=FONTE)),
    )
    for eixo, titulo in (("xaxis", titulo_x), ("yaxis", titulo_y)):
        fig.update_layout(**{eixo: dict(
            title=dict(text=titulo, font=dict(color=p["apagado"], size=12)),
            tickfont=dict(color=p["apagado"], size=12),
            gridcolor=p["grade"], linecolor=p["eixo"], zerolinecolor=p["eixo"],
        )})
    return fig


def barras_por_pais(painel, tema: str = "light") -> go.Figure:
    """Ranking de países, barras horizontais, uma cor só e o número na ponta."""
    p = paleta(tema)
    dados = painel.paises.iloc[::-1]        # o maior em cima
    unidade = "documentos" if painel.por == "documento" else "pessoas"
    fig = go.Figure(go.Bar(
        x=dados["contagem"], y=dados["pais"], orientation="h",
        marker=dict(color=p["series"][0]),
        text=dados["contagem"], textposition="outside",
        textfont=dict(color=p["tinta_fraca"], size=12),
        hovertemplate="%{y}: %{x} " + unidade + "<extra></extra>",
        cliponaxis=False,
    ))
    altura = max(180, 28 * len(dados) + 60)
    return _layout(fig, tema, altura, titulo_x=unidade.capitalize(),
                   titulo_y="País de nascimento")


def _decadas_completas(decadas: pd.DataFrame) -> list[int]:
    """Todas as décadas entre a primeira e a última, para o eixo não comprimir um vazio."""
    if decadas.empty:
        return []
    valores = decadas["decada"].dropna().astype(int)
    return list(range(int(valores.min()), int(valores.max()) + 10, 10))


def barras_por_decada(painel, tema: str = "light",
                      limite: int = LIMITE_CATEGORIAS) -> go.Figure:
    """Barras empilhadas por década, com a cor do país de nascimento (§9)."""
    p = paleta(tema)
    nomeadas = dobrar_em_outros(painel.paises, limite=limite)
    mapa = cores(nomeadas, tema)

    dados = painel.decadas.copy()
    if dados.empty:
        return _layout(go.Figure(), tema, 240, titulo_x="Década")
    dados["serie"] = [x if x in mapa and x in nomeadas else
                      (SEM_ORIGEM if x == SEM_ORIGEM else OUTROS)
                      for x in dados["pais"].astype(str)]
    eixo = _decadas_completas(dados)
    unidade = "documentos" if painel.por == "documento" else "pessoas"

    fig = go.Figure()
    # A ordem das séries é a do ranking: a cor de um país não muda quando outro entra ou sai.
    ordem = [*nomeadas, OUTROS, SEM_ORIGEM]
    for serie in ordem:
        recorte = dados[dados["serie"] == serie]
        if recorte.empty:
            continue
        por_decada = recorte.groupby("decada", observed=True)["contagem"].sum()
        fig.add_bar(
            name=serie, x=eixo, y=[int(por_decada.get(d, 0)) for d in eixo],
            marker=dict(color=mapa[serie],
                        # 2px da cor da superfície separam um segmento do outro.
                        line=dict(color=p["superficie"], width=2)),
            hovertemplate="%{x}s · " + serie + ": %{y} " + unidade + "<extra></extra>",
        )
    fig.update_layout(barmode="stack", showlegend=True, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, x=0,
        font=dict(color=p["tinta_fraca"], size=12), title=None))
    fig = _layout(fig, tema, 360, titulo_x="Década", titulo_y=unidade.capitalize())
    fig.update_layout(showlegend=True, xaxis=dict(
        type="category", title=dict(text="Década", font=dict(color=p["apagado"], size=12)),
        tickfont=dict(color=p["apagado"], size=12), gridcolor=p["grade"],
        linecolor=p["eixo"]))
    return fig


def barras_por_ano(linha_do_tempo: pd.DataFrame, tema: str = "light") -> go.Figure:
    """A linha do tempo de uma pessoa: um ano por barra, uma série, sem legenda."""
    p = paleta(tema)
    if linha_do_tempo.empty:
        return _layout(go.Figure(), tema, 200, titulo_x="Ano")
    anos = linha_do_tempo["ano"].astype(int)
    completos = list(range(int(anos.min()), int(anos.max()) + 1))
    por_ano = dict(zip(anos, linha_do_tempo["documentos"].astype(int)))
    fig = go.Figure(go.Bar(
        x=completos, y=[por_ano.get(a, 0) for a in completos],
        marker=dict(color=p["series"][0]),
        hovertemplate="%{x}: %{y} documento(s)<extra></extra>",
    ))
    return _layout(fig, tema, 240, titulo_x="Ano", titulo_y="Documentos")
