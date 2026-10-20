# Classificador AHU-Sul

Ferramenta de **triagem documental** sobre o catálogo do Arquivo Histórico Ultramarino (coleção
AHU-Sul, **7.051 documentos**). Leva o pesquisador de uma pergunta a um conjunto de cotas que ele
possa justificar, citar e levar ao arquivo.

**Ela não contém as imagens dos manuscritos.** O que ela tem são os resumos do catálogo, as cotas,
e camadas de análise sobre eles.

## O que o app faz

- **Busca híbrida**: rota de cotas (as três formas do catálogo levam ao mesmo documento), BM25 por
  campos e busca semântica com `multilingual-e5-small`, fundidas por posição.
- **Filtros**: período, pasta, tipologia, pessoa, destinatário, e os do DeepSeek (categoria social,
  vetor, SRSP), que começam desligados.
- **Pessoas**: lista de pessoas e lista separada de coletivos e instituições, página da pessoa com
  linha do tempo, tipologias e destinatários, e um painel diatópico por país de nascimento.
- **Seleção** que atravessa buscas e sobrevive a um recarregamento (ela viaja na URL).
- **Conversor de cotas**: código de referência (`PT/AHU/CU/…`), cota do Resgate (`AHU_ACL_CU_…`)
  e cota antiga (`AHU-<região>, cx., doc.`) convertidos entre si, também para cotas de fora do
  catálogo. A cota antiga só se converte pela base, salvo Castro e Almeida e Mendes Gouveia, que
  guardam a numeração atual.
- **Lista para consulta no AHU**, em PDF: a folha que se leva ao arquivo, com ano, cota, código de
  referência, resumo, link do DigitArq e uma caixa para marcar.
- **Amostras**: diacrônica por pessoa e estratificada com semente visível, com exportação no
  formato `eins-plano-coleta/1`, que o EINS Lite importa.

## Rodar no seu computador

Python 3.11 (3.12 também funciona).

```
pip install -r requirements.txt
python -m streamlit run app.py
```

O app abre em <http://localhost:8501>. Ele lê só a pasta `app_dados/`, que já vem no repositório —
não é preciso reconstruir nada para usá-lo.

Para reconstruir o corpus a partir das camadas, e para os detalhes de manutenção, veja
[`AGENTS.md`](AGENTS.md) e [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Base teórica

Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…* SciELO Preprints, 2026.
doi:[10.1590/SciELOPreprints.16461](https://doi.org/10.1590/SciELOPreprints.16461).

> **Versão provisória.** O preprint ainda não passou por revisão por pares, e os números de
> auditoria citados na interface vêm dele. Leia-os como provisórios.

Nomes e escalas usados em todo o projeto, iguais aos do preprint:

| Termo | Significado |
|---|---|
| **SRSP** | Índice de registro sociolinguístico, **inteiro de 1 a 9** (nunca dividido por 10). Estimado pelo DeepSeek v3; auditoria do preprint §2.1: MAE 1,13, r = 0,48, viés +0,59, N = 705, avaliador único |
| **`doc_uid`** | Identificador estável de documento (ex.: `ahu-sul:d7f3a9c21e`), interno: não aparece na interface |
| **`pessoa_uid`** | Identificador permanente de pessoa (ex.: `pes:4b81c02d`), também interno |

## Licença

- **Código: MIT** ([`LICENSE`](LICENSE)).
- Dados derivados: **CC BY 4.0**, provisório ([`LICENSE-DATA.md`](LICENSE-DATA.md)).
- Fonte embutida no PDF: **Noto Serif** (SIL OFL), em
  [`ahu/fontes/LICENSE-NotoSerif-OFL.txt`](ahu/fontes/LICENSE-NotoSerif-OFL.txt).
- Resumos e cotas do catálogo: Arquivo Histórico Ultramarino e Projeto Resgate
  "Barão do Rio Branco".

## `autoria_verificada` (spec §7.2)

A amostra diacrônica por pessoa só aceita documentos em que a autoria daquela pessoa é
**verificável no próprio resumo do catálogo**:

> Um documento tem `autoria_verificada` para uma pessoa quando **um nome, uma variante gráfica ou
> um título dela aparece na descrição do AHU**.

A comparação usa a `chave_nome` de [`ahu/texto.py`](ahu/texto.py) nos dois lados — a mesma
canonicalização que produz o `pessoa_uid`. É ela que sabe que *Manoel de Mello e Castro* e *Manuel
de Melo Castro* são o mesmo nome: tratamentos e partículas saem, grafias antigas são neutralizadas
(`ph→f`, `ll→l`, `y→i`) e variantes conhecidas são unificadas (`manoel→manuel`, `souza→sousa`). O
casamento é por sequência de tokens, de modo que *Melo* não casa dentro de *Melosa*, e uma forma de
um só token curto (até três letras) é descartada.

Na v2 este sinalizador era **por documento**, copiado para todos os remetentes dele. A §7.2 pede
que seja **por entidade**, e é assim que o app calcula.
