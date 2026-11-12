"""reranker.py — releitura do topo da busca por um cross-encoder (plano da busca no servidor, §1).

O léxico e o e5 olham para a consulta e para o documento **separados**: o e5 compara dois
vetores feitos sem se verem, e o BM25 conta palavras. O cross-encoder lê os dois **juntos**, numa
só passada do transformer, e por isso pesa o que o resumo diz sobre o que se perguntou — ao custo
de uma inferência por par. Por isso ele só relê o topo da fusão (`busca.LIMITE_RERANKER`), nunca o
corpus.

Roda por ONNX, como o e5 (`ahu/consulta_onnx.py`): nada de `torch` no app (R7). É **desligado
por padrão** e liga com `AHU_RERANKER=1`, que é o que o servidor próprio faz; a versão leve não o
carrega.

O modelo e o arquivo vêm de variáveis de ambiente, para a troca não tocar no código:
`AHU_RERANKER_MODELO` (repositório no Hugging Face) e `AHU_RERANKER_ARQUIVO` (o `.onnx` dentro
dele). Os valores padrão e as medidas que os escolheram estão em `docs/plano_busca_servidor.md`.
"""
from __future__ import annotations

import os

import numpy as np

from ahu import consulta_onnx as onnx

MAXIMO_TOKENS = 192           # consulta + resumo: 99 % dos pares cabem (medido: p99 = 189)
LOTE = 16                     # pares por chamada ao onnxruntime

MODELO_PADRAO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ARQUIVO_PADRAO = "onnx/model_quint8_avx2.onnx"   # int8 para CPU com AVX2, como a do agiulfo


def ligado() -> bool:
    return os.environ.get("AHU_RERANKER", "").strip().lower() in ("1", "sim", "true")


def texto_do_documento(linha) -> str:
    """O que o cross-encoder lê de cada documento: tipologia, resumo e remetente/destinatário.

    A data entra porque consultas com século ou ano ("cartas de 1750") dependem dela.
    """
    partes = [str(linha.get("description") or "").strip()]
    data = str(linha.get("document_id_and_date") or "").strip()
    if data:
        partes.append(f"Data: {data}")
    return " ".join(p for p in partes if p)


def criar_reranker(modelo: str | None = None, arquivo: str | None = None, threads: int | None = None,
                   log=None):
    """Devolve `reranker(consulta, textos) -> notas`, uma nota (logit) por texto, maior = melhor."""
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as e:   # pragma: no cover - depende do ambiente
        raise onnx.ErroDeModelo(f"Falta uma dependência do reranker ({e}).") from e

    modelo = modelo or os.environ.get("AHU_RERANKER_MODELO", MODELO_PADRAO)
    arquivo = arquivo or os.environ.get("AHU_RERANKER_ARQUIVO", ARQUIVO_PADRAO)
    caminho_tok = onnx.baixar_arquivo(modelo, "tokenizer.json", log)
    caminho_modelo = onnx.baixar_arquivo(modelo, arquivo, log)
    # Modelos grandes guardam os pesos ao lado do grafo, em `<arquivo>_data`.
    if os.environ.get("AHU_RERANKER_DADOS_EXTERNOS", "").strip() in ("1", "sim", "true"):
        onnx.baixar_arquivo(modelo, arquivo + "_data", log)

    tokenizador = Tokenizer.from_file(str(caminho_tok))
    tokenizador.enable_truncation(max_length=MAXIMO_TOKENS)
    tokenizador.enable_padding()

    opcoes = ort.SessionOptions()
    # No servidor há núcleos de sobra para uma consulta de cada vez.
    opcoes.intra_op_num_threads = threads or int(os.environ.get("AHU_RERANKER_THREADS", "4"))
    opcoes.inter_op_num_threads = 1
    try:
        sessao = ort.InferenceSession(str(caminho_modelo), opcoes,
                                      providers=["CPUExecutionProvider"])
    except Exception as e:      # pragma: no cover - formato do arquivo
        raise onnx.ErroDeModelo(f"O reranker ONNX não pôde ser aberto ({e}).") from e
    esperados = {entrada.name for entrada in sessao.get_inputs()}

    def reranker(consulta: str, textos: list[str]) -> np.ndarray:
        # Os pares vão aos lotes em ordem de tamanho: cada lote é preenchido até o seu maior
        # par, e juntar curtos com curtos corta o enchimento que o modelo lê à toa.
        todos = tokenizador.encode_batch([(consulta, t) for t in textos])
        ordem = sorted(range(len(todos)), key=lambda i: len(todos[i].ids))
        notas = np.zeros(len(textos), dtype=np.float32)
        for inicio in range(0, len(ordem), LOTE):
            indices = ordem[inicio: inicio + LOTE]
            codificados = tokenizador.encode_batch([(consulta, textos[i]) for i in indices])
            ids = np.array([c.ids for c in codificados], dtype=np.int64)
            entradas = {
                "input_ids": ids,
                "attention_mask": np.array([c.attention_mask for c in codificados], dtype=np.int64),
                "token_type_ids": np.array([c.type_ids for c in codificados], dtype=np.int64),
            }
            saida = sessao.run(None, {k: v for k, v in entradas.items() if k in esperados})[0]
            notas[indices] = np.asarray(saida, dtype=np.float32).reshape(len(indices), -1)[:, 0]
        return notas

    return reranker
