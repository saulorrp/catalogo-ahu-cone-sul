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

O download é fixado por revisão git e conferido por SHA-256 para garantir reprodutibilidade e
evitar alterações silenciosas em modelos remotos. Para usar um espelho local ou offline, baixe
os arquivos para `<pasta>/{repositorio}/resolve/{revisao}/<arquivo>` e aponte
`AHU_ONNX_BASE_URL=file:///<pasta>/{repositorio}/resolve/{revisao}`.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

PREFIXO_PADRAO = "query: "
MAXIMO_TOKENS = 512

BASE_PADRAO = "https://huggingface.co/{repositorio}/resolve/{revisao}"
# De onde o modelo vem. `AHU_ONNX_BASE_URL` existe para um fork apontar para outro espelho.
BASE = os.environ.get("AHU_ONNX_BASE_URL", BASE_PADRAO)

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
# Com os 167 mil documentos da coleção ahu-brasil (23/10/2026) o app já não cabe nos 690 MiB
# (956 MiB medidos com o int8) e vai para uma hospedagem com mais memória. A lista ficou tão
# densa que o int8 dava o mesmo primeiro colocado em só 70 % das consultas; o fp32 passa nos
# limites do teste de equivalência. O padrão voltou a ser o fp32.
#
# `AHU_ONNX_ARQUIVO=onnx/model_quantized.onnx` volta ao int8 sem tocar no código.
ESPELHOS = {
    "intfloat/multilingual-e5-small": "Xenova/multilingual-e5-small",
    "intfloat/multilingual-e5-base": "Xenova/multilingual-e5-base",
    "intfloat/multilingual-e5-large": "Xenova/multilingual-e5-large",
}
ARQUIVO_MODELO = os.environ.get("AHU_ONNX_ARQUIVO", "onnx/model.onnx")
ARQUIVOS = {"modelo": ARQUIVO_MODELO, "tokenizador": "tokenizer.json"}

# Revisões git e hashes SHA-256 fixados para garantir que execuções futuras baixem exatamente
# os mesmos artefatos testados e aprovados.
MODELOS_FIXADOS: dict[str, dict] = {
    "Xenova/multilingual-e5-small": {
        "revisao": "761b726dd34fb83930e26aab4e9ac3899aa1fa78",
        "arquivos": {
            "onnx/model.onnx": "4aa845c27760e06e9a686b9d8b5d440eae4b6612cd09e5b522b716d3941f77ff",
            "onnx/model_quantized.onnx": (
                "f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193"
            ),
            "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
        },
    },
    "Xenova/multilingual-e5-base": {
        "revisao": "1ec9243030a27d1a115d5c340572074c125b58b2",
        "arquivos": {
            "onnx/model.onnx": "84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa",
            "onnx/model_quantized.onnx": (
                "df7a9a29309e3ad491e1783adf8baee710262cc06079c7cbab63c630277fac94"
            ),
            "tokenizer.json": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        },
    },
    "Xenova/multilingual-e5-large": {
        "revisao": "00fc3aeb3dbb95842de2ac1961d33c6319acf57b",
        "arquivos": {
            "onnx/model.onnx": "8e17fbd06c54c2a53b467f5b127230f798d4fb69c5344ef9a9d6fa463862c453",
            "onnx/model.onnx_data": (
                "0cf1883fee81c63819a44e2ba0efa51d4043d9759685a4ebebbde97e0623d15c"
            ),
            "onnx/model_quantized.onnx": (
                "0a8d65db9a36f810ba5da15249f13145fcdc7890e6656f1fd38cd8b7c4db1fca"
            ),
            "tokenizer.json": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        },
    },
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": {
        "revisao": "1427fd652930e4ba29e8149678df786c240d8825",
        "arquivos": {
            "onnx/model.onnx": "3e9a03ed1e966f7c5288dd4230e3d6a9bf5e3a170a06f1f4241c5bca12c6487c",
            "onnx/model_quint8_avx2.onnx": (
                "6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b"
            ),
            "tokenizer.json": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        },
    },
}

# Memoriza no processo os caminhos locais de arquivos cuja integridade já foi conferida.
_CONFERIDOS: set[Path] = set()


def repositorio_de(modelo: str) -> str:
    """Onde buscar o ONNX de um modelo do índice. Sem espelho conhecido, o próprio repositório."""
    return ESPELHOS.get(modelo, modelo)


def revisao_de(repositorio: str) -> str:
    """A revisão git fixada para o repositório, ou 'main' se desconhecido."""
    info = MODELOS_FIXADOS.get(repositorio)
    if info and "revisao" in info:
        return info["revisao"]
    return "main"


def sha256_esperado(repositorio: str, arquivo: str) -> str | None:
    """O hash SHA-256 esperado para o arquivo do repositório, se fixado."""
    info = MODELOS_FIXADOS.get(repositorio)
    if not info:
        return None
    return info.get("arquivos", {}).get(arquivo)


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


def _sha256_arquivo(caminho: Path) -> str:
    """Calcula o SHA-256 de um arquivo em blocos de 1 MiB para não esgotar a memória."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while bloco := f.read(1 << 20):
            h.update(bloco)
    return h.hexdigest()


def baixar(modelo: str, qual: str, log=None) -> Path:
    """Baixa um arquivo do modelo, se ainda não estiver em cache. Devolve o caminho local."""
    try:
        return baixar_arquivo(repositorio_de(modelo), ARQUIVOS[qual], log)
    except ErroDeModelo as e:
        raise ErroDeModelo(
            f"Não foi possível baixar {qual} do modelo {modelo} ({e}). A busca conceitual "
            f"precisa dele; a busca exata e a por cota continuam funcionando."
        ) from e


def baixar_arquivo(repositorio: str, arquivo: str, log=None) -> Path:
    """Baixa `arquivo` do Hugging Face para o cache com conferência de revisão e SHA-256."""
    import urllib.error
    import urllib.request

    destino = pasta_de_cache() / repositorio.replace("/", "__") / Path(arquivo).name
    esperado = sha256_esperado(repositorio, arquivo)

    if destino.exists() and destino.stat().st_size > 0:
        if esperado is not None:
            if destino in _CONFERIDOS:
                return destino
            obtido = _sha256_arquivo(destino)
            if obtido == esperado:
                _CONFERIDOS.add(destino)
                return destino
            if log:
                log(
                    f"cache divergente para {arquivo} em {repositorio} "
                    f"({obtido} != {esperado}); baixando novamente"
                )
            destino.unlink(missing_ok=True)
        else:
            if log and destino not in _CONFERIDOS:
                log(f"aviso: arquivo {arquivo} de {repositorio} sem sha256 fixado")
                _CONFERIDOS.add(destino)
            return destino

    revisao = revisao_de(repositorio)
    destino.parent.mkdir(parents=True, exist_ok=True)
    url = BASE.format(repositorio=repositorio, revisao=revisao) + "/" + arquivo
    if log:
        log(f"baixando {url}")
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    try:
        with urllib.request.urlopen(url, timeout=120) as resposta, open(parcial, "wb") as saida:
            while bloco := resposta.read(1 << 20):
                saida.write(bloco)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        parcial.unlink(missing_ok=True)
        raise ErroDeModelo(str(e)) from e

    if esperado is not None:
        obtido = _sha256_arquivo(parcial)
        if obtido != esperado:
            parcial.unlink(missing_ok=True)
            raise ErroDeModelo(
                f"Integridade violada para {arquivo} de {repositorio}: "
                f"esperado {esperado}, obtido {obtido}."
            )
        _CONFERIDOS.add(destino)
    elif log:
        log(f"aviso: arquivo {arquivo} de {repositorio} sem sha256 fixado")

    # Renomeia só no fim: um download interrompido não pode virar um cache quebrado.
    parcial.replace(destino)
    return destino


def manifesto_modelos() -> dict:
    """Metadados dos modelos de consulta e reranker para o manifesto do app."""
    return {
        "fonte": BASE_PADRAO,
        "repositorios": {repo: dict(dados) for repo, dados in MODELOS_FIXADOS.items()},
        "espelhos": dict(ESPELHOS),
        "arquivo_consulta": ARQUIVO_MODELO,
    }


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
