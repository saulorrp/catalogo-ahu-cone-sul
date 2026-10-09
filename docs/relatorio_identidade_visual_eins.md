# Relatório de identidade visual — EINS Lite

Para reaproveitar em um projeto parente. Tudo foi extraído do código atual (referências no fim).

## 1. Princípio geral

- **Sóbrio, arquivístico, de ferramenta de pesquisa.** Nada de gradientes, vidro ou animações. Cantos pouco arredondados, bordas finas, fundo quase neutro.
- **Um único acento: bordô (`#800000`).** Todo o resto é neutro (cinza no tema escuro, creme no tema dia).
- **Glifos tipográficos de manuscrito no lugar de ícones**: ❧ ❦ ⸿. A identidade vem da paleografia, não de uma biblioteca de ícones.
- **Monoespaçada na tela (Consolas), serifada no papel (Noto Serif).** A tela é instrumento; o PDF é documento.
- **Tudo em tokens.** Nenhuma aba define cor própria. Tudo vem de `gui/theme.py`, aplicado por `gui/shell/style.py`.

## 2. Tokens de cor

| Token | Escuro | Dia |
|---|---|---|
| BG_BASE | `#0d0d0d` | `#f5f2eb` |
| BG_PANEL | `#161616` | `#ece8dd` |
| BG_CARD | `#1e1e1e` | `#ffffff` |
| BG_CARD_DEEP | `#111111` | `#f0ecdf` |
| BG_HOVER | `#2a2a2a` | `#e4dfd2` |
| BG_ACTIVE | `#252525` | `#ddd8ca` |
| BORDER / MID / BRIGHT | `#2a2a2a` / `#333333` / `#555555` | `#d0c9b4` / `#c8c0a8` / `#b0a890` |
| TXT_PRIMARY | `#e0e0e0` | `#1c1b18` |
| TXT_SECONDARY | `#aaaaaa` | `#5a5548` |
| TXT_MUTED / DIM | `#888888` / `#666666` | `#7a7264` / `#8a8272` |
| **ACCENT_PRIMARY** | **`#800000`** | **`#a83030`** |
| ACCENT_HOVER | `#990000` | `#800000` |
| ACCENT_RED (erro, remover) | `#b84040` | `#a83030` |
| ACCENT_BLUE (links) | `#4a88b8` | `#3a6a8a` |
| BTN_TEXT | `#ffffff` | `#ffffff` |

Observações:
- **O tema dia é "papel envelhecido"**: neutros levemente amarelados (`#f5f2eb`, `#ece8dd`), e não cinza puro.
- No tema dia o acento fica **mais claro** (`#a83030`) e o hover **mais escuro** (`#800000`). No escuro é o contrário.
- Os aliases `ACCENT_GREEN` e `ACCENT_AMBER` apontam para o bordô. São herança do EINS 3.0. Num projeto novo, use só `ACCENT_PRIMARY`.
- **Fundo da tela de imagem: `#777777` (L\*50), nos dois temas.** É um cinza neutro de avaliação, para não enganar a percepção de cor.
- **Paleta de anotação** (estilo One Dark, saturada, só para marcar sobre imagem; vem de `form_config.json`):

| Tipo | Cor |
|---|---|
| Brasão | `#e06c75` |
| Dano | `#d19a66` |
| Código verificador | `#e5c07b` |
| Marca d'água | `#98c379` |
| Assinatura | `#61afef` |
| Sinal de Fé | `#c678dd` |
| Ferramentas (régua, bordas) | `#56b6c2` |

A UI é neutra e bordô; a **cor viva fica reservada ao conteúdo marcado**.

## 3. Tipografia

- **UI:** `'Consolas', monospace`, 13 px.
  - Título de página: 16 px bold.
  - Nome na rail: 15 px bold.
  - Detalhe e legendas: 10–11 px, em `TXT_MUTED`.
  - Rótulos de seção: bold em `TXT_SECONDARY`, com linha inferior de 1 px.
- **Valores calculados:** monoespaçada, bold (`readout_style`).
- **PDF:** **Noto Serif Regular e Bold** (SIL OFL, embutida como subconjunto). Corpo 9 pt, entrelinha 1,35, rodapé 8 pt. A Noto Serif foi escolhida também por ter o glifo ⸿ (U+2E3F).
- **Estado vazio ou sem valor:** itálico em `TXT_DIM` ("sem descrição"). No PDF, `N/D`.

## 4. A Gaveta EINS (`EinsDrawer`)

Seção recolhível, sem moldura, sem caixa.

- **Cabeçalho:** um `QToolButton` plano (`border: none`, bold, alinhado à esquerda, cursor de mão).
- **Fechada:** `❧ Título`.
- **Aberta:** `❦ Título`.
- **Conteúdo:** recuado 16 px à esquerda, 8 px acima e abaixo. Sem fundo próprio.
- **Estado inicial: sempre fechada.**
- **API:** `addWidget`, `addLayout`, `toggle`, `set_open`, `is_open`, sinal `toggled(bool)`.
- **Mesmo vocabulário em outros lugares:** a tabela EXIF usa ❧/❦ nos grupos (`exif_table.py`), e o EINS 3.0 usava nos grupos do formulário.

**Semântica de uso (D47):** a gaveta carrega uma decisão, não só espaço.
- Gavetas servem para o que é **opcional ou secundário**: Direitos, FADGI, Dublin Core (derivado), "Alvos e relatório", "FADGI e medições".
- **Gaveta fechada = recurso desligado**, e fica de fora da saída (o PDF omite a linha).
- A parte essencial da tela fica sempre visível, fora de gavetas.

Para copiar: o arquivo tem 63 linhas, só PyQt6. Copie `widgets/eins_drawer.py` como está.

## 5. O Capitulum ⸿ como sinal de seleção

⸿ (U+2E3F) é o sinal de parágrafo medieval, que marcava onde começa uma seção do texto. No EINS ele é **o sinal de "isto está marcado/selecionado"**.

| Onde | Como aparece |
|---|---|
| **Botões de tipo de anotação** | `⸿` na cor do tipo + nome do tipo, em grade que quebra em colunas conforme a largura do painel. É um botão `checkable` que mostra o tipo ativo. |
| **Lista de anotações** | Uma linha por anotação: `⸿` (18 px) na cor do tipo, número, descrição acima e "tipo · medida" abaixo, ✕ discreto à direita. |
| **PDF (ficha do fólio)** | Cada anotação começa com ⸿ na cor do tipo, com recuo pendente (o texto alinha depois do sinal). |
| **Número no retângulo** | O mesmo número da lista aparece no canto do retângulo sobre a imagem. |

**Regra:** o ⸿ é sempre desenhado **na cor do tipo**, nunca em cor de acento. A cor identifica a categoria; o sinal identifica "isto é uma marca".

Implementação (`capitulum_icon(color, size=20)`):
- cria um `QPixmap` transparente;
- desenha o texto ⸿ centralizado, em bold, com `size - 2` px, na cor do tipo;
- devolve um `QIcon`, que serve a botões e a `QLabel`.

**Mesmo glifo na tela e no PDF**: a identidade atravessa a saída.

## 6. Seleção e estados interativos

| Estado | Tratamento |
|---|---|
| Estágio atual na rail | Fundo `BG_CARD` + **barra esquerda de 3 px em bordô** |
| Linha de lista selecionada | Fundo `BG_ACTIVE` + **barra esquerda de 3 px em bordô** |
| Hover de lista ou botão | `BG_HOVER` |
| Miniatura selecionada | Borda de 1 px em bordô + `BG_HOVER` |
| Cartão em rascunho | Borda **tracejada** de 2 px em bordô |
| Slot vazio | Borda tracejada de 1 px em `BORDER_MID`; no hover, borda bordô |
| Erro de campo | Borda de 1 px em `ACCENT_RED` |
| Desabilitado | Texto em `TXT_DIM` |
| Foco de teclado | Contorno pontilhado, sem parecer "atual" |

Duas ideias recorrentes: **barra esquerda bordô = "aqui estou"**, e **tracejado = "ainda não é definitivo"**.

## 7. Estrutura da janela (D57)

- **Rail** à esquerda, largura fixa de 188 px: logo (28 px), nome "EINS Lite" (15 px bold), etapas em ordem de fluxo (Alt+1…7), Processo aberto, alternador de tema, "⚙ Configurações".
- **Área principal** + **painel lateral** (mostrável/ocultável).
- **Barra de ação única** embaixo: ações da etapa, navegação de fólio (`◀ Anterior · Fólio n de N · Próximo ▶`), e à direita Exportar, ⚠ avisos e ▤ painel.
- **Painel de avisos** recolhível; avisos não abrem modais automáticos.
- **Barra de status** de uma linha.
- **Botões da barra** são símbolos com tooltip, para a barra não alargar a janela.
- Organização inspirada no Krokiet; o visual é próprio.

## 8. Formas e medidas

- **Raio de borda:** 4 px (botões, campos), 6 px (grupos, painéis internos, lista), 10 px (cartões de folha, alvos), 7–9 px (etiquetas e chips em pílula).
- **Bordas:** sempre 1 px `BORDER_MID`.
- **Padding:** botão 6×14, campo 6, rail 6×10.
- **Margens:** barra de ação 16/6, rail 10/12.
- **Etiqueta/chip:** pílula bordô com texto branco 10 px (`etiqueta_folha`).
- **Cartão "+" vazio:** borda tracejada de 2 px com círculo de 72 px; no hover, o círculo fica bordô.
- **Barra de progresso:** chunk bordô.

## 9. Logo e ícone

- Dois conjuntos em `resources/icons/`:
  - **Tema escuro:** `eins_icon.*` (símbolo claro).
  - **Tema dia:** `cover_symbol.*` (símbolo escuro).
- Há também `eins_icon_blink.svg`, uma variante de piscar.
- O ícone **troca com o tema**, para manter o contraste.
- O mesmo `cover_symbol.svg` é usado na capa do PDF (fundo branco).

## 10. Identidade no PDF

- Capa em A4, margem de 20 mm, ícone no topo e linhas separadoras.
- Rótulos em caixa-alta seguidos de dois-pontos, com valor ao lado (`FUNDO:`, `SÉRIE:`, `COTA:`).
- Ficha por fólio: cabeçalho, linhas chave-valor, anotações com ⸿.
- Rodapé: `{cota} · Folha N · fl. Nr · página p/P`, em 8 pt.
- Sem escala física falsa (D63) e sem logos de terceiros.

## 11. Voz e texto

- UI em **português**; código e docs em inglês.
- Tom seco e preciso: "Nenhum Processo aberto", "sem descrição", "N/D".
- Ajuda e dicas em `TXT_MUTED`, sem emoji.
- **Símbolos permitidos:** ❧ ❦ ⸿ ⚙ ⚠ ▤ ✕ ◀ ▶. Todos são glifos de fonte, não imagens.

## 12. O que levar ao projeto parente

1. **Obrigatório:** `theme.py` (tokens), `style.py` (QSS), o bordô como único acento, `eins_drawer.py`, `capitulum_icon`.
2. **Recomendado:** rail + barra de ação única, barra bordô de 3 px, tracejado para provisório, tema "dia papel".
3. **Decidir:** manter o ⸿ como marca de seleção, ou dar ao projeto parente um glifo próprio da mesma família (por exemplo ☙ ou ¶ medieval). A cor viva de cada categoria deve continuar separada do acento.
4. **Cuidados:**
   - `theme.py` e `components.py` são arquivos do EINS 3.0 que o EINS Lite não edita (o gate verifica). Copie-os, mas não os reformate.
   - A fonte de UI é Consolas (do Windows). Fora do Windows, o fallback é `monospace`.
   - Para o PDF, empacote Noto Serif com a licença OFL.

## Fontes consultadas

- [`theme.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/theme.py)
- [`style.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/shell/style.py)
- [`eins_drawer.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/widgets/eins_drawer.py)
- [`annotation_list.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/widgets/annotation_list.py)
- [`rail.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/shell/rail.py)
- [`action_bar.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/shell/action_bar.py)
- [`icons.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/gui/icons.py)
- [`pdf_layout.py`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/engines/pdf_layout.py)
- [`form_config.json`](file:///c:/Users/Saulo_r/Yandex.Disk/NEWVALT%205/eins%20lite/eins_lite/resources/form_config.json)
