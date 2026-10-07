# Classificador AHU-Sul v3

Ferramenta de **triagem documental** sobre o catálogo do Arquivo Histórico Ultramarino (coleção AHU-Sul, 7.051 documentos).
Leva o pesquisador de uma pergunta a um conjunto de cotas que ele possa justificar, citar e levar ao arquivo.

> **Estado:** em construção (v3.0.0.dev0). O app em produção continua sendo a versão anterior (`main` do repositório `saulorrp/catalogo-ahu-cone-sul`). Veja o estado de cada pacote em [`AGENTS.md`](AGENTS.md).

## Base teórica

Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…* SciELO Preprints, 2026.
doi:[10.1590/SciELOPreprints.16461](https://doi.org/10.1590/SciELOPreprints.16461) (v3).

Nomes e escalas usados em todo o projeto (iguais aos do preprint):

| Termo | Significado |
|---|---|
| **SRSP** | Índice de registro sociolinguístico, **inteiro de 1 a 9** (nunca dividido por 10). Estimado pelo DeepSeek v3; auditoria do preprint §2.1: MAE 1,13, r = 0,48, viés +0,59, N = 705, avaliador único |
| **`doc_uid`** | Identificador estável de documento (ex.: `ahu-sul:d7f3a9c21e`) |
| **`pessoa_uid`** | Identificador permanente de pessoa (ex.: `pes:4b81c02d`) |

## Documentação

- [Especificação técnica](../spec_catalogo_ahu_v3.md)
- [Plano de implementação](../plano_implementacao_v3.md)
- [Guia para agentes](AGENTS.md)

## Estrutura

```
ahu/          módulos do app
build/        scripts de construção do corpus e dos artefatos
avaliacao/    avaliação da busca
corpus/       base e camadas (dados)
app_dados/    artefatos lidos pelo app
tests/        testes automáticos
scripts/      gate.py (ruff + pytest)
```

## Desenvolvimento

```
pip install -r requirements.txt
python scripts/gate.py        # ruff + pytest; deve imprimir PASS
python build/importar_base.py # importa a base e cria os doc_uid (WP 1.1)
```

Python 3.11 (declarado em `runtime.txt`); 3.12 funciona localmente.
Custo zero em funcionamento: nenhuma API paga é usada pelo app. As chamadas pagas (DeepSeek, Perplexity) ocorrem só na construção do corpus.

## Licença

- Código: **MIT** ([`LICENSE`](LICENSE)).
- Dados derivados: **CC BY 4.0**, provisório (ver [`LICENSE-DATA.md`](LICENSE-DATA.md)).
- Resumos e cotas do catálogo: Arquivo Histórico Ultramarino e Projeto Resgate "Barão do Rio Branco".

## Como citar

Rocha, S. R. P. *Classificador AHU-Sul v3* (software e dados), 2026. A citação final com o DOI do Zenodo será registrada na versão 3.0.0.
