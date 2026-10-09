# Classificador AHU-Sul v3

Ferramenta de **triagem documental** sobre o catálogo do Arquivo Histórico Ultramarino (coleção AHU-Sul, 7.051 documentos).
Leva o pesquisador de uma pergunta a um conjunto de cotas que ele possa justificar, citar e levar ao arquivo.

> **Estado:** em construção (v3.0.0.dev0). O app em produção continua sendo a versão anterior (`main` do repositório `saulorrp/catalogo-ahu-cone-sul`); a publicação da v3 núcleo é o WP 2.5, descrita em [`docs/publicacao.md`](docs/publicacao.md). Veja o estado de cada pacote em [`AGENTS.md`](AGENTS.md).

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
- [Publicação, memória e troca de modelo (WP 2.5)](docs/publicacao.md)

## Estrutura

```
ahu/          módulos do app (inclui fontes/, a Noto Serif embutida no PDF)
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

# construção do corpus, nesta ordem
python build/importar_base.py           # base e doc_uid (WP 1.1)
python build/importar_camadas.py        # tipologia e sociolinguística (WP 1.2)
python build/importar_biografias_v2.py  # entidades e biografias v2 (WP 1.3)
python build/construir_indice.py        # índice semântico (WP 1.4); 97,6 min em CPU nos 7.051 docs
#   para o modelo que vai para o ar (WP 2.5), em lugar da linha acima:
#   python build/construir_indice.py --modelo intfloat/multilingual-e5-small --desativar-outras
python build/construir_app.py           # artefatos que o app lê (WP 1.5)
```

Python 3.11 (declarado em `runtime.txt`); 3.12 funciona localmente. No Streamlit Community Cloud o
`runtime.txt` não é lido: a versão do Python vem das *Advanced settings* do deploy.
O `torch` é fixado na variante só-CPU — ver [`docs/publicacao.md`](docs/publicacao.md).
Custo zero em funcionamento: nenhuma API paga é usada pelo app. As chamadas pagas (DeepSeek, Perplexity) ocorrem só na construção do corpus.

## `autoria_verificada` (spec §7.2)

A amostra diacrônica por pessoa só aceita documentos em que a autoria daquela pessoa é
**verificável no próprio resumo do catálogo**. A regra, implementada em
[`ahu/amostras.py`](ahu/amostras.py):

> Um documento tem `autoria_verificada` para uma pessoa quando **um nome, uma variante gráfica ou
> um título dela aparece na descrição do AHU**.

Três detalhes que fazem a regra funcionar no corpus:

1. a comparação usa a **`chave_nome`** de [`ahu/texto.py`](ahu/texto.py) nos dois lados — a mesma
   canonicalização que produz o `pessoa_uid` (§3.3). É ela que sabe que *Manoel de Mello e Castro*
   e *Manuel de Melo Castro* são o mesmo nome: tratamentos (`D.`, `frei`) e partículas (`de`, `e`)
   saem, grafias antigas (`ph→f`, `ll→l`, `y→i`) são neutralizadas e variantes conhecidas são
   unificadas (`manoel→manuel`, `souza→sousa`);
2. o casamento é por **sequência de tokens**, inteira e seguida, de modo que *Melo* não casa
   dentro de *Melosa*;
3. uma forma de **um só token curto** (até três letras, como *Sá*) é descartada: ela casaria por
   acidente em quase qualquer descrição.

Isto é uma mudança em relação à v2 e ao que o pipeline grava hoje: lá `autoria_verificada` é um
sinalizador **por documento**, copiado para todos os remetentes dele. A §7.2 pede que passe a ser
**por entidade**, e é assim que o app calcula — sobre `app_dados/`, sem reconstruir camada
nenhuma. Levar o cálculo para dentro da camada de entidades é possível depois, e aí será uma
camada nova.

## Licença

- Código: **MIT** ([`LICENSE`](LICENSE)).
- Fonte embutida no PDF: **Noto Serif** (SIL OFL), em [`ahu/fontes/LICENSE-NotoSerif-OFL.txt`](ahu/fontes/LICENSE-NotoSerif-OFL.txt).
- Dados derivados: **CC BY 4.0**, provisório (ver [`LICENSE-DATA.md`](LICENSE-DATA.md)).
- Resumos e cotas do catálogo: Arquivo Histórico Ultramarino e Projeto Resgate "Barão do Rio Branco".

## Como citar

Rocha, S. R. P. *Classificador AHU-Sul v3* (software e dados), 2026. A citação final com o DOI do Zenodo será registrada na versão 3.0.0.
