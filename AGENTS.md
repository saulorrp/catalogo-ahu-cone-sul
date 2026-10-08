# AGENTS.md — Classificador AHU-Sul v3

Guia único para qualquer agente de IA (Claude, Gemini, Codex, etc.). `CLAUDE.md` e `GEMINI.md` só apontam para este arquivo.

## Propósito
Instrumento de **triagem documental** sobre o catálogo do AHU (Arquivo Histórico Ultramarino, coleção AHU-Sul).
Leva o pesquisador de uma pergunta a um conjunto de cotas que ele possa justificar, citar e levar ao arquivo.
Quatro tarefas: **T1 Encontrar · T2 Recortar · T3 Amostrar · T4 Levar e citar**.

## Documentos de referência
- Spec: [`../spec_catalogo_ahu_v3.md`](../spec_catalogo_ahu_v3.md) (v2.0)
- Plano: [`../plano_implementacao_v3.md`](../plano_implementacao_v3.md)
- Entrada do pipeline de biografias: [`../biografias_v2/`](../biografias_v2/) (p1–p5)
- App antigo (referência, não editar): [`../legacy app/`](../legacy%20app/), [`../lista_ahu_pacote/`](../lista_ahu_pacote/)

## Regras (inegociáveis)
1. Onde o código e o preprint divergirem, **vale o preprint**, salvo decisão registrada na spec §0 (D1).
2. **Nada pago em funcionamento (R1).** Chamadas pagas (DeepSeek, Perplexity) só na construção do corpus, na máquina do autor.
3. **O app só lê `app_dados/`** (spec §3.4). Nunca lê `corpus/` nem as camadas brutas.
4. **`scribe_mediation_likely` só no PDF** (D1). Nunca na interface, nos filtros, no CSV nem no JSON.
5. **Biografias: só ALTA e MÉDIA** (ou `decisao_manual = aceitar`). Nunca REJEITADO nem BAIXA (D2).
6. **IDs públicos são `doc_uid` e `pessoa_uid`.** Nunca `P0001`, nem posição de linha.
7. **Pastas, coleções e tipologias vêm dos dados, nunca do código** (D6).
8. Nenhum segredo no código. Chaves ficam no `.env` local (via `ApiKeyManager`) e nunca são versionadas.
9. Nomes: **SRSP**, valor **inteiro**, nunca dividido por 10. A escala é um **dado da camada**, não
   uma constante do código: sai de `"escala_srsp": [min, max]` no `camada.json` da camada ativa
   (hoje `[1, 9]` em `sociolinguistica_deepseek_v3`). Uma reclassificação com outra escala é uma
   **camada nova**, que declara a sua. O prompt do §2 é registro histórico e não se edita.
10. Dependências fixadas com versão, em número mínimo (R2).
11. Arquivos grandes de dados são tratados em lotes e com retomada; nunca duplicar o catálogo inteiro em memória sem necessidade.

## Estrutura
```
ahu/          módulos do app, sem Streamlit exceto app.py
build/        scripts de construção (importar_base, importar_biografias_v2, construir_indice, construir_app)
avaliacao/    consultas de ouro, julgamento, métricas
corpus/       base + camadas (spec §3.1)
app_dados/    artefatos que o app lê (spec §3.4)
tests/        pytest
scripts/      gate.py
schemas/      JSON Schemas (ex.: eins-plano-coleta-1.json)
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
python build/construir_app.py     # WP 1.5 (artefatos de app_dados/; ~7 s)

streamlit run app.py              # WP 2.3: o app novo, em http://localhost:8501
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
| F2 | 2.4 Pessoas | próximo |
| F2–F4 | restantes | pendente |

Atualize esta tabela ao concluir cada pacote.

**Pendências para o autor:**
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
