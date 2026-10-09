# Seletores do Streamlit de que o CSS depende

O CSS de [`ahu/estilo.py`](../ahu/estilo.py) alcança partes internas do Streamlit. Isto é o que há
de contrato, e o que conferir quando o Streamlit subir de versão.

**Versão de referência: Streamlit 1.55.0.** Tudo abaixo foi lido do pacote instalado
(`streamlit/static/static/js/index.RuhrnD1v.js`), não adivinhado. O arquivo do pacote muda de nome
a cada compilação; procure por `data-testid="stExpander"`.

## A árvore que `st.expander` monta

```html
<div data-testid="stExpander" class="stExpander st-key-SUACHAVE">   <!-- StyledExpandableContainer -->
  <details>                                   <!-- StyledDetails: é ela que tem moldura e raio -->
    <summary>                                 <!-- StyledSummary: display:flex, é o clicável -->
      <span>                                  <!-- StyledSummaryHeading: flex, gap de 8 px -->
        <span>                                <!-- StyledDynamicIcon: 20 px fixos (o invólucro) -->
          <span data-testid="stIconMaterial">keyboard_arrow_down</span>
        </span>
        <div>                                 <!-- StyledSummaryLabelWrapper -->
          <div data-testid="stMarkdownContainer"><p>…rótulo…</p></div>
        </div>
      </span>
    </summary>
    <div data-testid="stExpanderDetails">…conteúdo…</div>   <!-- StyledDetailsPanel: borderTop -->
  </details>
</div>
```

## Os seletores usados, e o que os sustenta

| Seletor | Para quê | Estabilidade |
|---|---|---|
| `[data-testid="stExpander"]` | raiz de todo expansor | **testid**, estável |
| `> details` / `> summary` | estado e cabeçalho | **tags HTML**; o Streamlit usa `<details>`/`<summary>` de verdade |
| `details[open]` | fechada ⇄ aberta | o Streamlit escreve `detailsRef.current.open` |
| `summary span:has(> [data-testid="stIconMaterial"])` | esconder o galão **com o invólucro** | testid no filho; o invólucro não tem testid, só classe do Emotion |
| `summary span:has(> svg)` | o mesmo, na variante em SVG | tag |
| `[data-testid="stExpanderDetails"]` | o painel do conteúdo | **testid**, estável |
| `[class*="st-key-gaveta_"]` | distinguir gaveta de cartão | classe pública, documentada pelo Streamlit |
| `summary [data-testid="stMarkdownContainer"] p` | o negrito chegar ao texto | **testid**, estável |

**Nada depende de classe do Emotion** (`st-emotion-cache-…`, `ewmru5c4`, `emntfgb2`): essas mudam
a cada compilação e aparecem aqui só como documentação da árvore.

## Três detalhes que custaram a descobrir

1. **O galão tem um invólucro de 20 px sem testid.** Esconder só o `stIconMaterial` deixava os
   20 px e o `gap` de 8 px do cabeçalho, e o fleurão ficava longe do rótulo. Daí o `:has()`.
2. **`details[open]` ao fechar demora.** Abrindo, o Streamlit põe `open` na hora; fechando, só
   quando a animação de altura termina (~500 ms). O fleurão acompanha o que se vê, que é o que se
   quer.
3. **A `key` do `st.expander` não gera classe.** No 1.55 o expansor só ganha identificador — e
   portanto a classe `st-key-…` — quando é *stateful*, isto é, quando recebe `on_change`. E
   `on_change` faria cada abrir e fechar reexecutar a página inteira, que é o que o R2 tirou. Por
   isso cada gaveta é embrulhada num `st.container(key=…)`, que emite a classe só por ter chave
   (está no próprio código do Streamlit: *"the ID is only used for extracting the key on the
   frontend and setting it as CSS class"*).

## O que conferir numa atualização do Streamlit

Rodar `pytest tests/test_estilo.py` cobre o lado do nosso código, mas **não** o lado do Streamlit.
Depois de atualizar, abrir o app e olhar:

1. o galão (`›` / `⌄`) **não** reaparece ao lado do fleurão;
2. o fleurão vira ❦ ao abrir e volta a ❧ ao fechar;
3. as gavetas continuam sem moldura, com o conteúdo recuado;
4. os cartões de resultado continuam **com** moldura;
5. o contorno pontilhado de foco aparece ao navegar com Tab.

Se algo falhar, reler a árvore no pacote novo: `grep -o 'data-testid="stExpander[^"]*"'` sobre
`streamlit/static/static/js/*.js`.
