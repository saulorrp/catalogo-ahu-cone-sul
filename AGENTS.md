# AGENTS.md — Classificador AHU-Sul v3

> **Comece lendo [`docs/HANDOFF.md`](docs/HANDOFF.md).** Ele diz em que ponto o trabalho está, o
> que vem a seguir, as decisões em vigor e as armadilhas desta máquina. Este arquivo aqui é o
> contrato do projeto; o HANDOFF é o estado dele.

Guia único para qualquer agente de IA (Claude, Gemini, Codex, etc.). `CLAUDE.md` e `GEMINI.md` só apontam para este arquivo.

## Propósito
Instrumento de **triagem documental** sobre o catálogo do AHU (Arquivo Histórico Ultramarino, coleção AHU-Sul).
Leva o pesquisador de uma pergunta a um conjunto de cotas que ele possa justificar, citar e levar ao arquivo.
Quatro tarefas: **T1 Encontrar · T2 Recortar · T3 Amostrar · T4 Levar e citar**.

## Documentos de referência
- **Seletores do Streamlit de que o CSS depende:** [`docs/seletores_streamlit.md`](docs/seletores_streamlit.md)
  — reler a cada atualização do Streamlit.
- **Remodelação em curso (v3.1):** [`docs/plano_remodelacao_v3_1.md`](docs/plano_remodelacao_v3_1.md)
  — aprovada em 09/10/2026, substitui as partes abertas da F3 e vem antes da F4 e do deploy.
  Identidade visual: [`docs/relatorio_identidade_visual_eins.md`](docs/relatorio_identidade_visual_eins.md).
- Spec: [`../spec_catalogo_ahu_v3.md`](../spec_catalogo_ahu_v3.md) (v2.0)
- Plano: [`../plano_implementacao_v3.md`](../plano_implementacao_v3.md)
- Publicação, memória medida e troca de modelo (WP 2.5): [`docs/publicacao.md`](docs/publicacao.md)
- Entrada do pipeline de biografias: [`../biografias_v2/`](../biografias_v2/) (p1–p5)
- App antigo (referência, não editar): [`../legacy app/`](../legacy%20app/), [`../lista_ahu_pacote/`](../lista_ahu_pacote/)

## Regras (inegociáveis)
1. Onde o código e o preprint divergirem, **vale o preprint**, salvo decisão registrada na spec §0 (D1).
2. **Nada pago em funcionamento (R1).** Chamadas pagas (DeepSeek, Perplexity) só na construção do corpus, na máquina do autor.
3. **O app só lê `app_dados/`** (spec §3.4). Nunca lê `corpus/` nem as camadas brutas.
4. **`scribe_mediation_likely` não aparece em lugar nenhum** (D8, que revê a D1). Não na
   interface, não nos filtros, não em PDF. Ele fica no dado (`pdf_extra.parquet`) e é explicado,
   com a ressalva do preprint §2 citada na íntegra, só no "saiba mais" das inferências. Em `ahu/`
   o nome dele só existe como **proibição** (`dados.CAMPOS_SO_PDF`, `ficha.CAMPOS_PROIBIDOS`), e
   há teste que falha se outro módulo o mencionar.
5. **Biografias: só ALTA e MÉDIA** (ou `decisao_manual = aceitar`). Nunca REJEITADO nem BAIXA (D2).
6. **IDs são `doc_uid` e `pessoa_uid`.** Nunca `P0001`, nem posição de linha. Eles são
   **internos** (D10): não entram em rótulo, tabela nem PDF — o pesquisador vê cota, ano, resumo
   e nomes. A única exceção é a *issue* de "Reportar erro", que vai ao mantenedor.
7. **Pastas, coleções e tipologias vêm dos dados, nunca do código** (D6).
8. Nenhum segredo no código. Chaves ficam no `.env` local (via `ApiKeyManager`) e nunca são versionadas.
9. Nomes: **SRSP**, valor **inteiro**, nunca dividido por 10. A escala é um **dado da camada**, não
   uma constante do código: sai de `"escala_srsp": [min, max]` no `camada.json` da camada ativa
   (hoje `[1, 9]` em `sociolinguistica_deepseek_v3`). Uma reclassificação com outra escala é uma
   **camada nova**, que declara a sua. O prompt do §2 é registro histórico e não se edita.
10. Dependências fixadas com versão, em número mínimo (R2). O `torch` é o **só-CPU**
    (`--extra-index-url https://download.pytorch.org/whl/cpu` e `torch==2.8.0+cpu`): o do PyPI em
    Linux arrasta as bibliotecas CUDA, que o app nunca usa.
11. Arquivos grandes de dados são tratados em lotes e com retomada; nunca duplicar o catálogo inteiro em memória sem necessidade.

## Estrutura
```
ahu/          módulos do app, sem Streamlit exceto app.py
ahu/fontes/   Noto Serif embutida no PDF (SIL OFL, licença na pasta)
ahu/estilo.py o único lugar com CSS (gaveta do EINS, fleurão por interruptor)
build/        scripts de construção (importar_base, importar_biografias_v2, construir_indice, construir_app)
avaliacao/    consultas de ouro, julgamento, métricas
corpus/       base + camadas (spec §3.1)
app_dados/    artefatos que o app lê (spec §3.4)
tests/        pytest
scripts/      gate.py
schemas/      JSON Schemas (eins-plano-coleta-1.json, do §7.4)
legado/       app antigo preservado (WP 2.3)
```

## Comandos
```
python scripts/gate.py            # ruff check + pytest; imprime PASS. Rodar antes de cada commit.
python -m pytest -q
python build/importar_base.py     # WP 1.1
python build/importar_camadas.py  # WP 1.2 (tipologia_v1 + sociolinguistica_deepseek_v3)
python build/importar_biografias_v2.py  # WP 1.3 (entidades_v2 + biografias_v2)
python build/construir_indice.py  # WP 1.4 (CPU: 97,6 min medidos nos 7.051 docs; com retomada)
python build/construir_indice.py --modelo intfloat/multilingual-e5-small --desativar-outras
                                  # WP 2.5: troca o modelo e desativa a camada antiga no manifesto
python build/construir_app.py     # WP 1.5 (artefatos de app_dados/; ~7 s)

streamlit run app.py              # o app novo, em http://localhost:8501
python scripts/medir_memoria.py   # WP 2.5: RSS e tempo da carga fria (A10)
```
A busca do WP 2.1 está em `ahu/busca.py`: `buscar(dados, texto, modo, rigor, encoder)`. Sem
`encoder` ela é só léxica (útil em teste); com `criar_encoder_consulta(dados)` o motor semântico
entra, e a primeira chamada carrega o e5 (~56 s).
A ordem acima é a ordem de construção. O `construir_app.py` lê as camadas que o
`corpus/manifest.json` marca como ativas, e é ele que faz valer a D1 e a D2 nos dados: o que não
é exibível não chega a `app_dados/`.
Ambiente: Python 3.11 declarado em `runtime.txt` (desenvolvimento local em 3.12 também funciona).

## Estado atual
| Fase | WP | Estado |
|---|---|---|
| F0 | 0.1 Repositório e convenções | feito |
| F0 | 0.2 Licença e README | feito |
| F0 | 0.3 Busca v2 de referência | pendente |
| F1 | 1.1 Base e `doc_uid` | feito (7.051 docs, 7.051 uids únicos) |
| F1 | 1.2 Tipologia e sociolinguística | feito (86 tipologias, 165 a revisar; SRSP 1–9) |
| F1 | 1.3 Entidades e biografias v2 | feito (7.229 entidades, 1.359 pessoas, 253 exibíveis) |
| F1 | 1.4 Índice semântico | feito (7.051 vetores e5-large, 97,6 min, sanidade 5/5) |
| F1 | 1.5 Artefatos do app | feito (A4 e A11 nos testes; carga em ~2 s) |
| **F1** | **completa** | **o corpus modular está pronto** |
| F2 | 2.1 Normalização e busca | feito (A3 passa; híbrida em ~330 ms) |
| F2 | 2.2 Consulta e filtros | feito (A6 na ida e volta; §6.1 inteira) |
| F2 | 2.3 Interface de busca e ficha | feito (A1 nas 7.051 fichas; app sobe em ~4 s) |
| F2 | 2.4 Pessoas | feito (A2: "Lavradio" -> 1 pessoa, 163 documentos) |
| F2 | 2.5 Publicação da v3 núcleo | **aberto**: índice e5-small no ar localmente; deploy adiado para depois da F3; A10 pendente do primeiro deploy |
| F3 | 3.1 Seleção | feito, **refeito no R2** (teto de 500, atravessa buscas, espelho na URL) |
| F3 | 3.2 Exportação CSV e JSON | **retirado pela D7** |
| F3 | 3.3 Dossiê em PDF | **retirado pela D7**; A8 aposentado |
| F3 | 3.3a Lista de consulta no AHU | feito, **refeito no R3** (7.050 de 7.051 cotas com código CRAV) |
| F3 | 3.4 Link da busca | feito (A6 com cinco buscas variadas, de ponta a ponta) |
| F3 | 3.5 Amostras e plano EINS | feito (A7 e A9) |
| **F3** | **remodelada pelo plano v3.1** | |
| v3.1 | R1 Remoções | feito (D7–D11) |
| v3.1 | R2 Seleção rápida | feito (0 buscas por clique; fragmento em ~46 ms) |
| v3.1 | R3 PDF da lista do AHU | feito (Noto Serif; linha nunca partida) |
| v3.1 | R9 Ajustes do autor | feito |
| v3.1 | R7 Caminho sem torch | feito (ONNX int8; pico 630 MiB) |
| v3.1 | R4 Rótulos da interface | feito (Remetentes, page_icon sem emoji; tradução diferida) |
| v3.1 | R5 "Saiba mais" | feito (popover com auditoria e ressalva do preprint §2) |
| v3.1 | R6 (resto) Bordô e tipografia | feito (.streamlit/config.toml, Noto Serif no resumo) |
| v3.1 | R8 Limpeza | pendente |
| F4 | restantes | pendente |

Atualize esta tabela ao concluir cada pacote.

**Pendências para o autor:**
- **publicar (WP 2.5), depois da F3.** A reindexação com o e5-small está feita (7.051 docs, 8,3
  min, sanidade 5/5, `embeddings_e5small_v1` ativa) e os artefatos de `app_dados/` já são de 384
  dimensões. O que falta é o `push` e o deploy, adiados por decisão do autor até a F3 terminar; os
  passos estão em [`docs/publicacao.md`](docs/publicacao.md). **A10 segue pendente do primeiro
  deploy**: tempo de resposta e carga fria só se medem na nuvem;
- **limite conhecido do aviso de consulta distante (§5.5).** A regra nova (nenhum termo no corpus
  **e** destaque < 4,5) acerta as nove consultas medidas, mas basta **um** termo existir no corpus
  para o aviso calar, mesmo que o casamento seja acidental: `military desertion` escapa porque
  `military`, depois da neutralização de grafia da §5.1, casa com um token português. Já
  `Fahnenflucht` e `greve dos trabalhadores` — conceitos legítimos sem palavra no corpus — recebem
  o aviso corretamente. Decidir no WP 4.2 se a condição (a) deve exigir mais que um termo;
- revisar `corpus/correcoes_tipologia.json`: as 165 tipologias ainda marcadas como a revisar
  (ex.: `PRIMEIRA`, `SEGUNDA`, `QUARTA`, `FORTALEZA`) são erros de extração que a tabela conserta;
- preencher a coluna `decisao_manual` do `revisao_biografias.csv` (hoje vazia nas 1.359 linhas) e
  rodar `python build/importar_biografias_v2.py` de novo: `aceitar` exibe uma biografia BAIXA ou
  REJEITADA, `rejeitar` esconde uma ALTA ou MÉDIA;
**Decisões de interface tomadas no WP 2.3, para o autor conferir na tela:**
- **sem texto, a resposta é o corpus com os filtros aplicados.** Sem isso uma lente sozinha, ou um
  filtro sozinho, devolvia zero documento — o que seria um bug visível na primeira tela.
- **consulta só de stopwords** (ex.: "de") mostra o mesmo corpus filtrado, com um aviso dizendo
  que a consulta foi ignorada, em vez de devolver zero sem explicação.
- **o modelo semântico carrega na primeira consulta que precisa dele**, não na abertura: abrir o
  app e buscar por cota não pagam os ~56 s do e5.

**Decidido no WP 2.5 (08/10): o modelo que vai para o ar é o `multilingual-e5-small`.** Medido
em `scripts/medir_memoria.py --com-app`, com o cache do Hugging Face quente (09/10, depois da F3),
o pico de RSS de uma consulta semântica é de **2.013 MiB** com o e5-large, **1.211 MiB** com o
e5-base e **948 MiB** com o e5-small; o mínimo garantido pelo Streamlit Community Cloud é 690 MiB
e o máximo documentado, 2,7 GiB. Só o small cabe, e sem folga. (A primeira medição dera 717 MiB
para o small, com o modelo recém-baixado e os pesos ainda não residentes: o número certo é 948.
A correção está em `docs/publicacao.md` §2.1.) A calibração fina (margens de rigor, pesos do RRF,
`LIMITE_DESTAQUE`) foi medida no e5-large e refaz-se no WP 4.2; os números estão todos em
constantes de `ahu/busca.py`. O raciocínio inteiro, a reindexação e os passos de publicação estão
em [`docs/publicacao.md`](docs/publicacao.md).

**Decidido e já no código (08/10):** os rótulos da §6.3 (tabela na spec); o tempo real da
indexação (97,6 min); a escala do SRSP como dado da camada (regra 9), com o prompt preservado como
registro histórico; e a `fonetica` seguindo o `p4_validar.py`, com a §5.1 corrigida para bater com
o código (`z→s` em qualquer posição, `oe→oi`).

**Limite conhecido do corpus modular:** uma coleção nova entra na base, na tipologia, na
sociolinguística e no índice sem mexer no código (A11, testado com `teste-mini`). As camadas de
pessoas são consolidadas por coleção; juntar a mesma pessoa em duas coleções é o WP 5.4, e até lá
`importar_biografias_v2.py` para em vez de sobrepor a camada de outra coleção.

## Convenção de trabalho
- Um pacote de trabalho por vez; terminar com testes passando (`gate.py` = PASS) e um commit.
- Não adicionar lógica de negócio em `app.py`; ela vai para `ahu/`.
