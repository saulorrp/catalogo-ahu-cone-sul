# Plano: busca no servidor próprio (registrado em 23/10/2026)

Decidido com o autor em 23/10/2026. A versão leve (Streamlit Community Cloud) usa o modelo
estático (`ahu/estatico.py`); a versão completa, no servidor próprio (Mac mini, ver
`deploy/SERVIDOR.md`), deve chegar ao estado da arte com o que segue. **O §1 está feito (12/11/2026); o §2 não.**

## 1. Híbrido e5 + reranker (prioridade)
- O léxico (BM25 com cobertura e proximidade, PR #12) e o e5-small trazem cada um o seu topo; o
  RRF junta os 50 primeiros, como hoje.
- Um **reranker** (cross-encoder) relê consulta e documento juntos e reordena esses 50.
  Candidato: `BAAI/bge-reranker-v2-m3` (multilíngue). Alternativa mais leve:
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Rodar por ONNX, como o e5 (`ahu/consulta_onnx.py`), para não trazer o `torch` ao app (R7).
  Exportar em fp32: o int8 do e5 reprovou (top-1 70%).
- Medir antes de ligar: tempo por consulta no Mac mini (meta: < 1,5 s para 50 pares) e RSS.
- Avaliar com `avaliacao/` (nDCG@10, Recall@20, MRR) contra o híbrido sem reranker.
- Liga por variável de ambiente (ex.: `AHU_RERANKER=1`), desligado na versão leve.

### Feito em 12/11/2026

`ahu/reranker.py` + `busca.reordenar`; liga com `AHU_RERANKER=1` (o `deploy/docker-compose.yml`
liga). Relê os **30** primeiros da fusão (`busca.LIMITE_RERANKER`), com o
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` em int8 (`onnx/model_quint8_avx2.onnx`, 119 MB),
pares cortados em 192 tokens. O modo Exata e a rota de cotas não passam por ele.

Avaliação com `avaliacao/comparar_motores.py` nas 24 consultas de ouro que não são cota. O top-20
de cada motor foi julgado às cegas (0/1/2) pelo Gemini pela API (chave de estudante; 960 pares em
`avaliacao/julgamentos_llm.json`). É um juiz de máquina, bom para comparar motores entre si; o
julgamento do autor (`julgar.py`) continua sendo o que vale para o artigo.

| motor | nDCG@10 | Recall@20 | MRR | tema | nome | lugar |
|---|---:|---:|---:|---:|---:|---:|
| léxico | 0,899 | 0,542 | 1,000 | 0,857 | 0,994 | 0,889 |
| híbrido (léxico + e5, RRF) | 0,863 | 0,518 | 0,946 | 0,803 | 0,926 | 0,920 |
| **híbrido + mMiniLM, top-30 (no ar)** | **0,929** | 0,551 | 1,000 | 0,909 | 0,960 | 0,936 |
| híbrido + mMiniLM, top-20 | 0,922 | 0,518 | 1,000 | 0,887 | 0,979 | 0,932 |
| híbrido + mMiniLM, top-50 | 0,945 | 0,557 | 1,000 | 0,933 | 0,953 | 0,961 |
| híbrido + bge-reranker-v2-m3 (int8), top-30 | 0,944 | 0,553 | 1,000 | 0,927 | 0,986 | 0,934 |

Tempo por consulta no agiulfo (i5-4278U de 2014, 2 núcleos), medido com a máquina parada: híbrido
**0,57 s** (mediana); com o reranker em 30, **3,1 s** (p90 3,8 s); em 50, 4,9 s. O bge-m3 levou
~24 s por consulta e foi descartado. A meta de 1,5 s não cabe neste processador: o top-30 é o meio
termo entre ganho (+0,066 no nDCG@10 sobre o híbrido) e espera. Com uma máquina mais rápida, subir
para 50 é só mudar `LIMITE_RERANKER`.

O RRF puro sai **abaixo** do léxico nesta avaliação (0,863 contra 0,899): o e5 traz documentos
de sentido próximo que o juiz considera menos relevantes que os de palavra exata. O reranker
recupera isso e passa os dois.

### Espera na tela (12/11/2026, PR do redesenho)

Sem trocar o modelo nem o número de candidatos, a espera caiu por quatro caminhos:

- **Busca em dois tempos** (`web/cache.py`, `busca.refinar`): a fusão responde e a tela mostra a
  lista; o reranker relê o topo num segundo pedido e a lista se reordena. O resultado final é o
  mesmo de `buscar(..., reranker=...)` (teste em `tests/test_web.py`).
- **Memória das respostas**: "Carregar mais", voltar e desmarcar um filtro não refazem a busca.
  Antes, cada página nova rodava a busca e o reranker de novo.
- **Memória dos pares** (`reranker.com_memoria`): a mesma consulta com outro filtro só relê os
  pares novos.
- **Lotes de 8 em ordem de tamanho**: a ordenação por tamanho que o código anunciava não
  acontecia (os pares já vinham preenchidos até o maior). Agora acontece.

Medido na nuvem (4 núcleos, 2 threads; os tempos do agiulfo são maiores, a proporção deve valer),
com os mesmos julgamentos, releitura de 30 pares:

| variante | nDCG@10 | Recall@20 | mediana |
|---|---:|---:|---:|
| como estava (lote 16, sem ordenar) | 0,926 | 0,548 | 1,55 s |
| lote 16, ordenado | 0,926 | 0,546 | 1,08 s |
| **lote 8, ordenado (no ar)** | **0,927** | 0,545 | **0,97 s** |
| lote 8, top-50 | 0,939 | 0,540 | 1,37 s |
| corte em 128 tokens (lote 16 antigo) | 0,918 | 0,539 | 0,91 s |
| corte em 96 tokens | 0,903 | 0,534 | 0,71 s |
| top-20 | 0,921 | 0,518 | 0,91 s |

O modelo int8 quantiza por lote, então a nota de um par oscila um pouco com a composição do
lote; nas métricas isso fica dentro do ruído (±0,003). Cortar tokens custa qualidade e não foi
adotado. `AHU_RERANKER_TOPO` e `AHU_RERANKER_TOKENS` mudam o número de candidatos e o corte sem
tocar no código: com o lote novo, top-50 custa menos que o top-30 custava.

## 2. Motor léxico com índice posicional (se a velocidade pedir)
- Tantivy (`tantivy-py`) ou SQLite FTS5: frase, NEAR e BM25 nativos, em milissegundos, do disco.
- Substituiria `ahu/lexico.py` e o bônus de proximidade feito à mão em `ahu/busca.py`.
- Só vale se o tempo do léxico (hoje 0,25 s em média, 0,7 s no pior caso) virar problema.

## Descartado por ora
ColBERT e SPLADE: um pouco melhores nos benchmarks, mas pesados de indexar e de guardar; para
167 mil verbetes curtos não compensam.
