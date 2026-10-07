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
9. Nomes: **SRSP**, escala inteira **1–9** (nunca dividida por 10).
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
```
Ambiente: Python 3.11 declarado em `runtime.txt` (desenvolvimento local em 3.12 também funciona).

## Estado atual
| Fase | WP | Estado |
|---|---|---|
| F0 | 0.1 Repositório e convenções | feito |
| F0 | 0.2 Licença e README | feito |
| F0 | 0.3 Busca v2 de referência | pendente |
| F1 | 1.1 Base e `doc_uid` | feito (7.051 docs, 7.051 uids únicos) |
| F1 | 1.2–1.5 | pendente |
| F2–F4 | — | pendente |

Atualize esta tabela ao concluir cada pacote.

## Convenção de trabalho
- Um pacote de trabalho por vez; terminar com testes passando (`gate.py` = PASS) e um commit.
- Não adicionar lógica de negócio em `app.py`; ela vai para `ahu/`.
