# Licenças por Camada de Dados

| Camada | Onde está | Licença |
|---|---|---|
| **Código** | `ahu/`, `web/`, `build/`, `coleta/scripts/`, `avaliacao/`, `scripts/`, `tests/`, `app.py`, `Dockerfile`, `deploy/` | MIT ([`LICENSE`](LICENSE)) |
| **Fatos e identificadores** | `doc_uid`, cotas (Resgate, referência, antiga), série, caixa, documento, ano, local, links (DigitArq, DocReader, ACTD), página de origem, marcas de QC, `duplicata_de` | CC0 1.0 (fatos não são protegidos; a CC0 só remove dúvidas) |
| **Camadas próprias** | `corpus/camadas/*` (tipologia, entidades, pessoas, biografias, SRSP, enriquecimento), `avaliacao/`, `coleta/qc/`, documentação em `docs/` | CC BY 4.0, com o crédito abaixo |
| **Resumos do catálogo do AHU** (via ACTD/IICT) | coluna `description`/resumo de `corpus/colecoes/*/documentos.parquet` e de `app_dados/documentos.parquet` | **Fora da licença aberta.** Termos do ACTD: uso livre só para fins pessoais, educativos e de investigação, com crédito; uso comercial exige autorização |
| **Textos do Projeto Resgate** (gov.br / Biblioteca Nacional) | `description_resgate` e textos copiados do portal | CC BY-ND 3.0 (licença do portal gov.br): cópia literal com crédito, sem alterar o texto |
| **Derivados diretos dos resumos** | `app_dados/lexico.pkl`, `app_dados/embeddings*`, vetores em `corpus/camadas/embeddings_*` | Seguem os termos dos resumos de que derivam (uso de pesquisa e educação) até haver autorização do AHU |

## Crédito exigido

> Dados derivados: Rocha, S. R. P. *Catálogo AHU-Brasil* (2026), doi:10.5281/zenodo.18772666, CC BY 4.0.
> Catálogo de origem: Arquivo Histórico Ultramarino (Lisboa), catálogos publicados no Arquivo
> Científico Tropical Digital (ACTD/IICT), e Projeto Resgate "Barão do Rio Branco"
> (Biblioteca Nacional / Ministério da Cultura).

Quem hospedar o catálogo deve manter este crédito no rodapé do site.

Base teórica: Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…*
SciELO Preprints, 2026, doi:10.1590/SciELOPreprints.16461.

## Licenças dos arquivos deste Registro B

| Arquivo / Diretório | Licença | Descrição |
|---|---|---|
| `documentos_sem_resumo.parquet` (fatos e índices) | CC0 1.0 | Identificadores, cotas, anos e referências arquivísticas |
| `documentos_sem_resumo.parquet` (camadas analíticas) | CC BY 4.0 | Tipologias normalizadas e enriquecimentos do projeto |
| `camadas/` | CC BY 4.0 | Camadas analíticas de tipologia, pessoas e SRSP |
| `reidratacao/*.py` | MIT | Scripts utilitários de reidratação de textos |
| `reidratacao/fontes_checksums.tsv` | CC0 1.0 | Checksums das fontes documentais externas |
| `reconstruir.sh` | MIT | Script de reconstrução dos artefatos da aplicação |
| `docker-compose.yml` | MIT | Configuração de implantação conteinerizada |
| `avaliacao/*` | CC BY 4.0 | Consultas de teste e pares avaliados de busca |
| `QC/*` | CC BY 4.0 | Relatórios de controle de qualidade e auditoria |
| `*.md`, `esquema.json` | CC BY 4.0 | Documentação técnica e esquemas de dados |
| `SHA256SUMS` | CC0 1.0 | Tabela de conferência de integridade criptográfica |
