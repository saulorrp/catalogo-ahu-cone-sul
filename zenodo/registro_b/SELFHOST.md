# Guia de Auto-hospedagem (Self-hosting)

O Catálogo AHU-Brasil pode ser executado em servidor próprio, estação local ou ambiente conteinerizado.

## Perfis de execução

| Perfil | Para quem | Recursos | Busca |
|---|---|---|---|
| **Só dados** | Quem quer consultar ou cruzar o banco | Nenhum servidor; DuckDB ou pandas sobre os parquet | SQL |
| **Leve** | Hospedagem gratuita ou máquina pequena | ~600 MiB de RAM, ~350 MB de disco | Léxico + Model2Vec (`AHU_SEM_SEMANTICA=1`) |
| **Completo** | Servidor institucional | ~3 GB de RAM, ~2 GB de disco, 2 núcleos | e5-small fp32 + reranker mMiniLM no top-30 (`AHU_RERANKER=1`); ~3 s por busca num i5 de 2014 |

### Exemplo de consulta analítica (Perfil Só dados com DuckDB)

```sql
SELECT doc_uid, ano_isolado, tipologia, reference_code
FROM read_parquet('documentos_sem_resumo.parquet')
WHERE ano_isolado BETWEEN 1750 AND 1777 LIMIT 10;
```

## Passos para implantação

1. **Baixar e verificar integridade**: faça o download do pacote e confira os arquivos com `sha256sum -c SHA256SUMS`.
2. **Reidratar resumos**: baixe os PDFs listados em `reidratacao/fontes_checksums.tsv` e rode `python reidratacao/reidratar_resumos.py ENTRADA.parquet PASTA_PDFS --checksums reidratacao/fontes_checksums.tsv` (ENTRADA = `doc_uid`, `url_actd`, `pagina_actd`, `sha1_resumo` deste parquet); o script diz quantos resumos bateram com `sha1_resumo`.
3. **Reconstruir artefatos**: execute `./reconstruir.sh` para gerar os índices e embeddings locais em `app_dados/`.
4. **Iniciar o contêiner**: suba a aplicação com `docker compose up -d` usando o arquivo `docker-compose.yml` desta pasta.
5. **Operação sem conexão à internet**: para executar em rede isolada, aponte a variável `AHU_ONNX_BASE_URL=file:///modelos/{repositorio}/resolve/{revisao}` com os pesos locais e hashes listados em `PROVENIENCIA.md`.
6. **Privacidade e logs**: o registro de buscas fica desativado por padrão (`AHU_LOG_DIR` vazio). Caso ativado, não registra endereços IP e aplica prazo estrito de retenção, em conformidade com LGPD e RGPD.
7. **Crédito obrigatório**: qualquer instância pública deve exibir o crédito de citação no rodapé da página:

> Dados derivados: Rocha, S. R. P. *Catálogo AHU-Brasil* (2026), doi:10.5281/zenodo.18772666, CC BY 4.0.
> Catálogo de origem: Arquivo Histórico Ultramarino (Lisboa), catálogos publicados no Arquivo
> Científico Tropical Digital (ACTD/IICT), e Projeto Resgate "Barão do Rio Branco"
> (Biblioteca Nacional / Ministério da Cultura).

Quem hospedar o catálogo deve manter este crédito no rodapé do site.

Base teórica: Rocha, S. R. P. *Desenvolvimento de um Classificador do Catálogo do AHU…*
SciELO Preprints, 2026, doi:10.1590/SciELOPreprints.16461.
