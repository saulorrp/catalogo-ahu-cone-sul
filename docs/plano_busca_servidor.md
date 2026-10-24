# Plano: busca no servidor próprio (registrado em 23/10/2026)

Decidido com o autor em 23/10/2026. A versão leve (Streamlit Community Cloud) usa o modelo
estático (`ahu/estatico.py`); a versão completa, no servidor próprio (Mac mini, ver
`deploy/SERVIDOR.md`), deve chegar ao estado da arte com o que segue. **Nada disto está feito.**

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

## 2. Motor léxico com índice posicional (se a velocidade pedir)
- Tantivy (`tantivy-py`) ou SQLite FTS5: frase, NEAR e BM25 nativos, em milissegundos, do disco.
- Substituiria `ahu/lexico.py` e o bônus de proximidade feito à mão em `ahu/busca.py`.
- Só vale se o tempo do léxico (hoje 0,25 s em média, 0,7 s no pior caso) virar problema.

## Descartado por ora
ColBERT e SPLADE: um pouco melhores nos benchmarks, mas pesados de indexar e de guardar; para
167 mil verbetes curtos não compensam.
