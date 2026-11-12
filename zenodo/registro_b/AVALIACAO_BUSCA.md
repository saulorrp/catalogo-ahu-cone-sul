# Avaliação dos Motores de Busca

Este documento descreve o estado atual da avaliação comparativa dos motores de busca desenvolvidos para o acervo.

## Conjunto de teste

- **Consultas de ouro**: 30 consultas catalogadas em `avaliacao/consultas_ouro.json`.
- **Distribuição por tipologia de consulta**:
- **cota**: 6 consultas
- **lugar**: 6 consultas
- **nome**: 6 consultas
- **tema**: 12 consultas
- **Pares julgados**: 960 pares documento-consulta avaliados às cegas por modelo de linguagem (Gemini) nas escalas de relevância 0, 1 e 2.

## Métricas por motor de busca

Tabela extraída de `docs/plano_busca_servidor.md`:

| motor | nDCG@10 | Recall@20 | MRR | tema | nome | lugar |
|---|---:|---:|---:|---:|---:|---:|
| léxico | 0,899 | 0,542 | 1,000 | 0,857 | 0,994 | 0,889 |
| híbrido (léxico + e5, RRF) | 0,863 | 0,518 | 0,946 | 0,803 | 0,926 | 0,920 |
| **híbrido + mMiniLM, top-30 (no ar)** | **0,929** | 0,551 | 1,000 | 0,909 | 0,960 | 0,936 |
| híbrido + mMiniLM, top-20 | 0,922 | 0,518 | 1,000 | 0,887 | 0,979 | 0,932 |
| híbrido + mMiniLM, top-50 | 0,945 | 0,557 | 1,000 | 0,933 | 0,953 | 0,961 |
| híbrido + bge-reranker-v2-m3 (int8), top-30 | 0,944 | 0,553 | 1,000 | 0,927 | 0,986 | 0,934 |

Nenhum julgamento humano foi publicado; avaliar_busca.py e julgar.py permitem refazê-lo.
