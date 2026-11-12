# Catálogo AHU-Brasil — registro B (dados abertos, sem resumos)

Versão do catálogo: **3.1.0**  
Data de geração: **2026-11-12T09:30:00+00:00**

Este registro reúne os metadados abertos, identificadores, cotas e enriquecimentos do Catálogo AHU-Brasil. Os textos protegidos por direitos de terceiros (resumos do catálogo e do Projeto Resgate) foram retirados para viabilizar a distribuição pública irrestrita. O conjunto permite pesquisa historiográfica computacional imediata e possibilita a reidratação local dos resumos a partir das fontes públicas do ACTD.

## Contagens do acervo

- **Documentos catalogados**: 167.344
- **Pessoas e autoridades identificadas**: 1.359
- **Distribuição por coleção**:
- **ahu-brasil**: 160.293 documentos
- **ahu-sul**: 7.051 documentos

## Como citar

> Dados derivados: Rocha, S. R. P. *Catálogo AHU-Brasil* (2026), doi:10.5281/zenodo.18772666, CC BY 4.0.
> Catálogo de origem: Arquivo Histórico Ultramarino (Lisboa), catálogos publicados no Arquivo
> Científico Tropical Digital (ACTD/IICT), e Projeto Resgate "Barão do Rio Branco"
> (Biblioteca Nacional / Ministério da Cultura).

Quem hospedar o catálogo deve manter este crédito no rodapé do site.

Base teórica: Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…*
SciELO Preprints, 2026, doi:10.1590/SciELOPreprints.16461.

## Mapa dos arquivos

| Arquivo / Pasta | O que é | Licença |
|---|---|---|
| `documentos_sem_resumo.parquet` | Metadados tabulares de documentos sem os resumos protegidos | CC0 1.0 (fatos) / CC BY 4.0 (camadas) |
| `camadas/` | Camadas analíticas de tipologia, entidades, pessoas e SRSP | CC BY 4.0 |
| `reidratacao/` | Scripts e tabela de checksums para recomposição dos resumos | MIT / CC0 1.0 |
| `reconstruir.sh` | Script de automação do pipeline de reconstrução do app | MIT |
| `docker-compose.yml` | Orquestração local em contêiner Docker | MIT |
| `avaliacao/` | Consultas de ouro e pares julgados para testes de busca | CC BY 4.0 |
| `QC/` | Relatórios de controle de qualidade e resumo da auditoria | CC BY 4.0 |
| `DICIONARIO_DE_DADOS.md` | Dicionário detalhado de cada coluna do dataset | CC BY 4.0 |
| `esquema.json` | Esquema formal de tipos, regras e contagens de vazios | CC BY 4.0 |
| `PROVENIENCIA.md` | Proveniência de cada camada analítica e modelos empregados | CC BY 4.0 |
| `LIMITACOES.md` | Limitações conhecidas e métricas de auditoria calculadas | CC BY 4.0 |
| `AVALIACAO_BUSCA.md` | Metodologia e resultados da avaliação dos motores de busca | CC BY 4.0 |
| `SELFHOST.md` | Instruções de auto-hospedagem e perfis de execução | CC BY 4.0 |
| `CHANGELOG.md` | Histórico formal de versões deste depósito | CC BY 4.0 |
| `LICENCAS.md` | Termos de licenciamento por camada e atribuição | CC BY 4.0 |
| `SHA256SUMS` | Hashes SHA-256 para conferência de integridade dos arquivos | CC0 1.0 |

## Níveis de reprodução

1. **Nível 1 — Só dados**: análise direta das tabelas em DuckDB, Polars ou pandas sobre `documentos_sem_resumo.parquet` e arquivos sob `camadas/`.
2. **Nível 2 — Reidratação dos resumos**: recomposição da coluna `description` a partir dos PDFs públicos do ACTD com `python reidratacao/reidratar_resumos.py` conferindo os hashes de `fontes_checksums.tsv`.
3. **Nível 3 — Reconstrução integral**: execução de `reconstruir.sh` para reconstruir os artefatos de busca e inferência de `app_dados/`, validando os hashes finais gerados.

## Como conferir a integridade

Execute na raiz do diretório descompactado:

```bash
sha256sum -c SHA256SUMS
```
