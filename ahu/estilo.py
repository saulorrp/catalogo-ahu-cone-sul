"""estilo.py — o único lugar com CSS (plano v3.1, R6).

O app injeta isto **uma vez**, no começo da página. Nada de `st.markdown("<style>…")` espalhado
pelas abas: quando o estilo mora em dois lugares, o segundo sempre discorda do primeiro.

## A gaveta do EINS

O expansor do Streamlit vem com um galão (`›` / `⌄`) e uma moldura. A gaveta do EINS
(`docs/relatorio_identidade_visual_eins.md` §4) é o contrário: **o fleurão é o interruptor** —
`❧ Título` fechada, `❦ Título` aberta —, o cabeçalho é chapado e em negrito, não há moldura, e o
conteúdo recua 16 px.

Aqui isso é feito **só com CSS**: o galão some, e o fleurão entra por `::before` no `<summary>`,
escolhido pelo estado do `<details>`. O glifo **nunca** está no texto do rótulo — assim ele não
passa pelo interpretador de Markdown do rótulo, não vira emoji e não aparece duas vezes ao lado do
galão.

## Os seletores, e por que estes

Lidos do pacote do **Streamlit 1.55.0** (`static/static/js/index.RuhrnD1v.js`), não adivinhados.
A árvore que o `st.expander` monta é:

```html
<div data-testid="stExpander" class="stExpander st-key-SUACHAVE">
  <details>                                    <!-- moldura e raio -->
    <summary>                                  <!-- display:flex; é o clicável -->
      <span>                                   <!-- cabeçalho, flex, com gap -->
        <span data-testid="stIconMaterial">keyboard_arrow_right</span>   <!-- o galão -->
        <div><div data-testid="stMarkdownContainer">…rótulo…</div></div>
      </span>
    </summary>
    <div data-testid="stExpanderDetails">…conteúdo…</div>
  </details>
</div>
```

Dois cuidados que valem na próxima atualização do Streamlit:

1. **`details[open]`** é escrito pelo próprio Streamlit (`detailsRef.current.open = …`). Ao
   **abrir** ele muda na hora; ao **fechar**, só quando a animação de altura termina (~500 ms).
   O fleurão acompanha o que se vê, que é o que se quer: enquanto o conteúdo ainda está
   recolhendo, a gaveta ainda está aberta;
2. o galão **não tem testid próprio**: ele é um ícone do Material (`stIconMaterial`) dentro do
   `<summary>`. Um `icon=` passado ao `st.expander` usaria `stExpanderIcon`, e por isso o seletor
   do galão é restrito ao `<summary>` — não escondemos ícone que o autor tenha pedido.

## O fleurão e as fontes

❧ (U+2767) e ❦ (U+2766) **não estão na Noto Serif** que vai embutida no PDF — ela tem o ⸿
(U+2E3F), mas não os fleurões. Eles vêm de fonte do sistema, e a pilha abaixo é escolhida para dar
sempre a forma de **texto**, nunca de emoji:

* `Segoe UI Symbol` — Windows (Edge, Chrome);
* `DejaVu Sans` — Linux;
* `Noto Sans Symbols 2`, `Noto Sans Symbols` — Android e Linux;
* `serif` — último recurso.

`font-variant-emoji: text` reforça a forma de texto onde o navegador a suporta. A cor é a do
texto, nunca o bordô: o fleurão é pontuação, não acento.
"""
from __future__ import annotations

# A pilha de fontes do fleurão. Fora daqui ninguém escolhe fonte.
FONTE_GLIFOS = ('"Segoe UI Symbol", "DejaVu Sans", "Noto Sans Symbols 2", '
                '"Noto Sans Symbols", "Apple Symbols", serif')

FECHADA = "❧"      # U+2767
ABERTA = "❦"       # U+2766

# As gavetas são embrulhadas num `st.container` cuja `key` começa assim; o Streamlit a transforma
# na classe `st-key-…`. O invólucro existe porque, no Streamlit 1.55, a `key` do **expansor** só
# vira identificador (e classe) quando ele é *stateful* — e isso faria cada clique reexecutar a
# página, que é o que o R2 tirou. O `st.container` emite a classe só por ter chave.
PREFIXO_GAVETA = "gaveta_"

CSS = f"""
<style>
/* ---------------------------------------------------------------------------
   Tipografia da §9: proporcional no texto, monoespaçada em cota e número.
   --------------------------------------------------------------------------- */
.cota, .numero {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.cota {{ font-size: 0.92rem; }}
.sic {{ color: #8b1a1a; font-size: 0.8rem; font-weight: 600; }}
.rotulo {{ color: var(--text-color, #5f6368); opacity: 0.75; font-size: 0.85rem; }}
.aviso-distante {{ background: rgba(232, 163, 61, 0.14); border-left: 4px solid #e8a33d;
                  padding: 0.6rem 0.8rem; border-radius: 4px; margin: 0.4rem 0 0.8rem; }}
/* Documento é papel: resumo com Noto Serif / serif */
.resumo, [data-testid="stExpanderDetails"] > div > [data-testid="stMarkdownContainer"] > p:first-child {{
  font-family: "Noto Serif", Georgia, Cambria, "Times New Roman", serif;
  font-size: 0.98rem;
  line-height: 1.55;
}}

/* ---------------------------------------------------------------------------
   O fleurão no lugar do galão, em TODO expansor — gaveta ou cartão.
   Um vocabulário só de recolher, como o plano pede.
   --------------------------------------------------------------------------- */

/* 1. fora o galão do Streamlit, **com o seu invólucro**.

      A árvore do galão é `<span>` (o invólucro, 20 px de largura fixa) > `<span
      data-testid="stIconMaterial">keyboard_arrow_down</span>`. Esconder só o de dentro deixaria
      os 20 px e o `gap` de 8 px do cabeçalho, e o fleurão ficaria longe do rótulo — foi o que
      aconteceu na primeira tentativa. Daí o `:has()`, ancorado no testid, que é o que há de
      estável aqui: o invólucro não tem testid nenhum, só classe gerada pelo Emotion.
      A segunda linha cobre a variante em SVG, que o Streamlit usa para outros ícones. */
[data-testid="stExpander"] > details > summary span:has(> [data-testid="stIconMaterial"]),
[data-testid="stExpander"] > details > summary span:has(> svg) {{
  display: none !important;
}}

/* 2. o fleurão entra como primeiro item flex do <summary>. */
[data-testid="stExpander"] > details > summary::before {{
  content: "{FECHADA}";
  font-family: {FONTE_GLIFOS};
  font-variant-emoji: text;
  font-size: 1.05rem;
  line-height: 1;
  flex: 0 0 auto;
  margin-right: 0.5rem;
  align-self: center;
  color: inherit;
  opacity: 0.85;
  /* `translate=no` e sem seleção: é um sinal, não texto para copiar. */
  user-select: none;
}}
[data-testid="stExpander"] > details[open] > summary::before {{
  content: "{ABERTA}";
}}

/* 3. foco de teclado visível e pontilhado (identidade do EINS §6). O Streamlit põe
      `outline:none` e um anel de sombra; aqui o anel sai e o pontilhado entra. */
[data-testid="stExpander"] > details > summary:focus-visible {{
  outline: 1px dotted currentColor;
  outline-offset: -2px;
  box-shadow: none;
}}

/* ---------------------------------------------------------------------------
   Fora o "Download as CSV" da barrinha das tabelas (D7, R9).
   A D7 diz que a exportação é uma coisa só: o PDF da lista do AHU. Uma tabela que oferece CSV
   no canto contradiz isso em silêncio. A busca e a tela cheia ficam — elas ajudam a ler.
   O botão não tem testid próprio; o que há de estável é o rótulo que o Streamlit lhe dá.
   --------------------------------------------------------------------------- */
[data-testid="stElementToolbarButton"][title="Download as CSV"],
[data-testid="stElementToolbarButton"][aria-label="Download as CSV"],
[data-testid="stElementToolbar"] button[title="Download as CSV"],
[data-testid="stElementToolbar"] button[aria-label="Download as CSV"] {{
  display: none !important;
}}

/* ---------------------------------------------------------------------------
   A gaveta: sem moldura, cabeçalho chapado em negrito, conteúdo recuado.
   Só dentro do `st.container` cuja chave começa por `{PREFIXO_GAVETA}`.
   --------------------------------------------------------------------------- */
[class*="st-key-{PREFIXO_GAVETA}"] [data-testid="stExpander"] > details {{
  border: none;
  border-radius: 0;
  background: transparent;
}}
[class*="st-key-{PREFIXO_GAVETA}"] [data-testid="stExpander"] > details > summary {{
  padding-left: 0;
  padding-right: 0;
  font-weight: 600;
  cursor: pointer;
  /* chapado: o Streamlit tinge o cabeçalho quando aberto, e a gaveta do EINS não tem fundo. */
  background: transparent !important;
}}
/* O rótulo é Markdown, e o `<p>` de dentro traz o seu próprio peso: sem isto o negrito do
   cabeçalho não chega ao texto. */
[class*="st-key-{PREFIXO_GAVETA}"] [data-testid="stExpander"] > details > summary p {{
  font-weight: inherit;
}}
[class*="st-key-{PREFIXO_GAVETA}"] [data-testid="stExpander"] > details
  > [data-testid="stExpanderDetails"] {{
  border-top: none;
  padding: 0.5rem 0 0.5rem 1rem;   /* 16 px de recuo à esquerda, como a gaveta do EINS */
}}
</style>
"""


def chave_de_gaveta(nome: str) -> str:
    """A `key` de um expansor que deve parecer uma gaveta do EINS.

    Passar por aqui, em vez de escrever a chave à mão, é o que garante que a classe que o CSS
    procura (`st-key-{PREFIXO_GAVETA}…`) e a chave do widget nunca se separem.
    """
    return f"{PREFIXO_GAVETA}{nome}"
