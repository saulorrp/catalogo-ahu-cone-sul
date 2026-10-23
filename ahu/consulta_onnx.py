"""consulta_onnx.py — codificar a consulta **sem `torch`** (plano v3.1, R7).

Em execução o app não pode importar `torch` nem `sentence_transformers`: juntos eles custam
~312 MiB de RSS antes de qualquer peso de modelo, e o Streamlit Community Cloud garante 690 MiB.
Aqui a consulta é codificada com `onnxruntime` + `tokenizers`, que é tudo o que ela precisa.

**Os documentos não são reindexados.** Os vetores de `app_dados/embeddings/` continuam os
que o `sentence-transformers` produziu na máquina do autor; o que muda é só quem codifica a
*consulta*. Por isso a equivalência é cobrada por teste: cosseno ≥ 0,99 contra o
`sentence-transformers` em ~30 consultas reais, e o mesmo top-10 em ≥ 90 % delas
(`tests/test_consulta_onnx.py`).

Três coisas têm de ser idênticas ao caminho de construção, ou os vetores deixam de ser
comparáveis:

1. o prefixo `"query: "` do e5, lido do manifesto;
2. **mean pooling** com a máscara de atenção — média só sobre os tokens reais, não sobre o
   enchimento;
3. normalização **L2** no fim.

**O modelo não vem no repositório.** Nem o quantizado cabe: ~120 MB contra os 100 MB que o GitHub
aceita. Ele é baixado do Hugging Face na primeira consulta semântica e fica em cache no disco
(`AHU_ONNX_CACHE`, ou a pasta temporária do sistema). No Streamlit Cloud isso acontece uma vez por
ambiente.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

PREFIXO_PADRAO = "query: "
MAXIMO_TOKENS = 512

# De onde o modelo vem. `AHU_ONNX_BASE_URL` existe para um fork apontar para outro espelho.
BASE = os.environ.get("AHU_ONNX_BASE_URL", "https://huggingface.co/{repositorio}/resolve/main")

# O índice é construído com o modelo do `intfloat`, mas o ONNX **quantizado** não está lá — está
# no espelho do `Xenova`, que é o export para transformers.js. São os mesmos pesos, em int8.
#
# **Por que o quantizado, e o que se paga por ele** (medido em 09/10/2026, 30 consultas reais):
#
#   | caminho                     | pico de RSS | equivalência                                  |
#   |-----------------------------|------------:|-----------------------------------------------|
#   | torch + sentence-transformers |   948 MiB | é a referência                     |
#   | ONNX fp32                     |   960 MiB | cosseno ≥ 0,99 e top-10 idêntico   |
#   | **ONNX int8 (este)**          | **630 MiB** | cosseno ≥ 0,9940; 90 % dos mesmos |
#                                                #  documentos no top-10; mesmo primeiro
#                                                #  colocado em 26 de 30 consultas
#
# O fp32 **piora** o que veio consertar: 960 MiB contra os 948 do `torch`. O int8 é a única
# configuração que cabe nos 690 MiB que o Streamlit Community Cloud garante, e é por isso que
# ele é o padrão. O preço é a ordem exata do top-10, que muda — os cossenos do e5 são
# apertados, e 0,994 de semelhança no vetor da consulta já basta para trocar posições numa
# lista densa. Para um instrumento de **triagem** isso é aceitável; para um teste de ranqueamento
# não seria.
#
# `AHU_ONNX_ARQUIVO=onnx/model.onnx` volta ao fp32 sem tocar no código.
ESPELHOS = {
    "intfloat/multilingual-e5-small": "Xenova/multilingual-e5-small",
    "intfloat/multilingual-e5-base": "Xenova/multilingual-e5-base",
    "intfloat/multilingual-e5-large": "Xenova/multilingual-e5-large",
}
ARQUIVO_MODELO = os.environ.get("AHU_ONNX_ARQUIVO", "onnx/model_quantized.onnx")
ARQUIVOS = {"modelo": ARQUIVO_MODELO, "tokenizador": "tokenizer.json"}


def repositorio_de(modelo: str) -> str:
    """Onde buscar o ONNX de um modelo do índice. Sem espelho conhecido, o próprio repositório."""
    return ESPELHOS.get(modelo, modelo)


class ErroDeModelo(RuntimeError):
    """O modelo de consulta não pôde ser preparado (rede, disco ou formato)."""


def pasta_de_cache() -> Path:
    """Onde o ONNX fica guardado entre execuções.

    `AHU_ONNX_CACHE` manda; senão, a pasta temporária do sistema, que no Streamlit Cloud dura o
    que durar o contêiner — o bastante para o download não se repetir a cada consulta.
    """
    bruto = os.environ.get("AHU_ONNX_CACHE")
    pasta = Path(bruto) if bruto else Path(tempfile.gettempdir()) / "ahu-sul-onnx"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def baixar(modelo: str, qual: str, log=None) -> Path:
    """Baixa um arquivo do modelo, se ainda não estiver em cache. Devolve o caminho local."""
    import urllib.error
    import urllib.request

    repositorio = repositorio_de(modelo)
    destino = pasta_de_cache() / repositorio.replace("/", "__") / Path(ARQUIVOS[qual]).name
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(repositorio=repositorio) + "/" + ARQUIVOS[qual]
    if log:
        log(f"baixando {qual} de {url}")
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    try:
        with urllib.request.urlopen(url, timeout=120) as resposta, open(parcial, "wb") as saida:
            while bloco := resposta.read(1 << 20):
                saida.write(bloco)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        parcial.unlink(missing_ok=True)
        raise ErroDeModelo(
            f"Não foi possível baixar {qual} do modelo {modelo} ({e}). A busca conceitual "
            f"precisa dele; a busca exata e a por cota continuam funcionando."
        ) from e
    # Renomeia só no fim: um download interrompido não pode virar um cache quebrado.
    parcial.replace(destino)
    return destino


def _mean_pooling(estados: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """A média dos tokens reais, que é o que o `sentence-transformers` faz para o e5."""
    pesos = mascara[..., None].astype(np.float32)
    soma = (estados * pesos).sum(axis=1)
    quantos = np.clip(pesos.sum(axis=1), 1e-9, None)
    return soma / quantos


def _normalizar(vetores: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    return vetores / np.clip(normas, 1e-12, None)


def criar_encoder(modelo: str, prefixo: str = PREFIXO_PADRAO, log=None):
    """Devolve `encoder(textos) -> matriz (n, dim)`, pronta para o cosseno contra os documentos.

    O carregamento é feito aqui, uma vez; o `app.py` o embrulha num cache de recurso para que
    isto aconteça uma vez por processo, e só quando a primeira consulta conceitual chega.
    """
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as e:   # pragma: no cover - depende do ambiente
        raise ErroDeModelo(
            f"Falta uma dependência do caminho de consulta ({e}). Instale `onnxruntime` e "
            f"`tokenizers` — veja requirements.txt."
        ) from e

    caminho_tok = baixar(modelo, "tokenizador", log)
    caminho_modelo = baixar(modelo, "modelo", log)

    tokenizador = Tokenizer.from_file(str(caminho_tok))
    tokenizador.enable_truncation(max_length=MAXIMO_TOKENS)
    tokenizador.enable_padding()

    opcoes = ort.SessionOptions()
    # Uma linha de execução: no Community Cloud há um oitavo de núcleo, e cada linha extra
    # custa memória sem devolver tempo.
    opcoes.intra_op_num_threads = 1
    opcoes.inter_op_num_threads = 1
    # Sem o arena do onnxruntime. Ele pré-aloca um bloco grande para reaproveitar entre
    # inferências; aqui as consultas são uma de cada vez, e o bloco só ocupa memória que o
    # Community Cloud não tem de sobra. Medido: tira centenas de MiB do pico.
    opcoes.enable_cpu_mem_arena = False
    opcoes.enable_mem_pattern = False
    try:
        sessao = ort.InferenceSession(str(caminho_modelo), opcoes,
                                      providers=["CPUExecutionProvider"])
    except Exception as e:      # pragma: no cover - formato do arquivo
        raise ErroDeModelo(f"O modelo ONNX não pôde ser aberto ({e}).") from e

    esperados = {entrada.name for entrada in sessao.get_inputs()}

    def encoder(textos: list[str]) -> np.ndarray:
        codificados = tokenizador.encode_batch([prefixo + t for t in textos])
        ids = np.array([c.ids for c in codificados], dtype=np.int64)
        mascara = np.array([c.attention_mask for c in codificados], dtype=np.int64)
        entradas = {"input_ids": ids, "attention_mask": mascara}
        if "token_type_ids" in esperados:
            entradas["token_type_ids"] = np.zeros_like(ids)
        entradas = {nome: v for nome, v in entradas.items() if nome in esperados}
        estados = sessao.run(None, entradas)[0]
        return _normalizar(_mean_pooling(estados, mascara)).astype(np.float32)

    return encoder


def criar_encoder_de_dados(dados, log=None):
    """O codificador descrito pelo manifesto de `app_dados/` — modelo e prefixo vêm de lá."""
    info = dados.modelo_embeddings
    nome = info.get("nome")
    if not nome:
        raise ErroDeModelo("O manifesto não diz qual é o modelo de embeddings.")
    return criar_encoder(nome, info.get("prefixo_consulta") or PREFIXO_PADRAO, log)
