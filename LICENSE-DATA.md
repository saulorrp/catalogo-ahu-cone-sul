# Licenças por camada

Este repositório mistura material de origens e direitos diferentes. Cada camada tem a sua licença,
e só vale a licença da linha em que o arquivo está. Em caso de dúvida, a mais restritiva prevalece.

| Camada | Onde está | Licença |
|---|---|---|
| **Código** | `ahu/`, `web/`, `build/`, `coleta/scripts/`, `avaliacao/`, `scripts/`, `tests/`, `app.py`, `Dockerfile`, `deploy/` | MIT ([`LICENSE`](LICENSE)) |
| **Fatos e identificadores** | `doc_uid`, cotas (Resgate, referência, antiga), série, caixa, documento, ano, local, links (DigitArq, DocReader, ACTD), página de origem, marcas de QC, `duplicata_de` | CC0 1.0 (fatos não são protegidos; a CC0 só remove dúvidas) |
| **Camadas próprias** | `corpus/camadas/*` (tipologia, entidades, pessoas, biografias, SRSP, enriquecimento), `avaliacao/`, `coleta/qc/`, documentação em `docs/` | CC BY 4.0, com o crédito abaixo |
| **Resumos do catálogo do AHU** (via ACTD/IICT) | coluna `description`/resumo de `corpus/colecoes/*/documentos.parquet` e de `app_dados/documentos.parquet` | **Fora da licença aberta.** Termos do ACTD: uso livre só para fins pessoais, educativos e de investigação, com crédito; uso comercial exige autorização |
| **Textos do Projeto Resgate** (gov.br / Biblioteca Nacional) | `description_resgate` e textos copiados do portal | CC BY-ND 3.0 (licença do portal gov.br): cópia literal com crédito, sem alterar o texto |
| **Derivados diretos dos resumos** | `app_dados/lexico.pkl`, `app_dados/embeddings*`, vetores em `corpus/camadas/embeddings_*` | Seguem os termos dos resumos de que derivam (uso de pesquisa e educação) até haver autorização do AHU |

Os resumos nunca são reescritos: erros da fonte ficam como estão e ganham a marca `[sic]` só na
exibição. Correções, quando houver, vão para colunas separadas.

## Depósito aberto sem os resumos

Não há autorização do AHU nem da BN para redistribuir os resumos sob licença aberta. Por isso o
depósito aberto dos dados (Zenodo) sai **sem os resumos**, com a coluna `sha1_resumo` e um script
que os recoloca a partir dos PDFs públicos do ACTD; quem precisa do texto o obtém da fonte, nos
termos da fonte. O pacote completo, se for depositado, fica em acesso restrito para pesquisa.

## Crédito exigido

> Dados derivados: Rocha, S. R. P. *Catálogo AHU-Brasil* (2026), doi:10.5281/zenodo.18772666, CC BY 4.0.
> Catálogo de origem: Arquivo Histórico Ultramarino (Lisboa), catálogos publicados no Arquivo
> Científico Tropical Digital (ACTD/IICT), e Projeto Resgate "Barão do Rio Branco"
> (Biblioteca Nacional / Ministério da Cultura).

Quem hospedar o catálogo deve manter este crédito no rodapé do site.

Base teórica: Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…*
SciELO Preprints, 2026, doi:10.1590/SciELOPreprints.16461.

## Histórico

- 12/11/2026: licença por camada, depois do parecer de direitos de uso. Substitui a CC BY 4.0
  "provisória" para todos os dados derivados, que relicenciava os resumos sem base para isso.
- Antes: a CC BY-NC-ND 4.0 do app anterior, que proibia obras derivadas.
