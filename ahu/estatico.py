"""estatico.py — busca por sentido sem rede neural, para a versão leve do app.

O e5 não cabe nos ~1 GB do Streamlit Community Cloud. Um modelo **estático** (Model2Vec,
`minishlab/potion-multilingual-128M`) guarda um vetor por pedaço de palavra: a consulta é
tokenizada, os vetores dos pedaços são somados e normalizados, e mais nada. Não há `torch`, nem
ONNX, nem modelo baixado em execução.

O modelo original tem 500 mil pedaços (512 MB em float32, e o tokenizador sozinho ocupa 560 MiB
de RAM). O `build/construir_estatico.py` corta o vocabulário aos pedaços que o corpus usa, mais
os caracteres soltos, e grava aqui só isso:

    app_dados/estatico/
        meta.json         modelo de origem, revisão, dimensão, calibração do cosseno
        tokenizer.json    o tokenizador com o vocabulário cortado
        matriz.f16.npy    float16[V, d]: o vetor de cada pedaço, na ordem do tokenizador
        vetores.i8.npy    int8[N, d]:    linha i = documentos.parquet linha i (L2-normalizados,
                                         vezes 127 / escala; a escala está no meta.json)

A qualidade fica abaixo do e5 (é uma média de palavras, sem contexto); é um motor de apoio ao
léxico, não um substituto do e5 no servidor próprio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PASTA = "estatico"
VERSAO = "1"
ARQUIVO_VETORES = "vetores.i8.npy"


def montar_texto(tipologia, remetente, pasta, descricao) -> str:
    """O mesmo texto do e5 (`build/construir_indice.py`), sem o prefixo 'passage: '."""
    partes = []
    for valor in (tipologia, remetente, pasta, descricao):
        if valor is None or (isinstance(valor, float) and valor != valor):
            continue
        texto = " ".join(str(valor).split())
        if texto:
            partes.append(texto)
    return ". ".join(partes)


def existe(pasta_app: Path) -> bool:
    return (Path(pasta_app) / PASTA / "meta.json").exists()


class EncoderEstatico:
    """Codificador da consulta e vetores dos documentos do modelo estático.

    Serve de `encoder` para `ahu.busca.buscar`: além de ser chamável, traz os `vetores` em que o
    cosseno é calculado e o `cosseno_minimo` calibrado para a escala dele (outra que a do e5).
    """

    def __init__(self, pasta_app: Path):
        from tokenizers import Tokenizer  # import tardio: só a versão leve precisa

        pasta = Path(pasta_app) / PASTA
        self.meta = json.loads((pasta / "meta.json").read_text("utf-8"))
        self.tokenizer = Tokenizer.from_file(str(pasta / "tokenizer.json"))
        self.matriz = np.load(pasta / "matriz.f16.npy", mmap_mode="r")
        # O build abre o codificador antes de os vetores existirem: é ele que os grava.
        vetores = pasta / ARQUIVO_VETORES
        self.vetores = np.load(vetores, mmap_mode="r") if vetores.exists() else None
        self.unk = self.meta.get("unk_id")
        self.escala = float(self.meta.get("escala", 127.0)) / 127.0
        # Calibração própria (ver `build/construir_estatico.py`): o cosseno do modelo estático
        # se espalha mais que o do e5, e ele é mais fraco, por isso pesa menos no RRF.
        self.cosseno_minimo = float(self.meta["cosseno_minimo"])
        self.margens_rigor = dict(self.meta["margens_rigor"])
        self.pesos_rrf = {k: dict(v) for k, v in self.meta["pesos_rrf"].items()}

    def __call__(self, textos: list[str]) -> np.ndarray:
        saida = np.zeros((len(textos), self.matriz.shape[1]), dtype=np.float32)
        for i, codificado in enumerate(self.tokenizer.encode_batch(list(textos),
                                                                   add_special_tokens=False)):
            ids = [t for t in codificado.ids if t != self.unk]
            if ids:
                saida[i] = np.asarray(self.matriz[ids], dtype=np.float32).mean(axis=0)
        normas = np.linalg.norm(saida, axis=1, keepdims=True)
        return saida / np.where(normas > 0, normas, 1.0)

    def produto(self, vetor: np.ndarray, bloco: int = 10_000) -> np.ndarray:
        """Cosseno de `vetor` com cada documento, em blocos (nada vai inteiro para float32)."""
        saida = np.empty(len(self.vetores), dtype=np.float32)
        for a in range(0, len(self.vetores), bloco):
            saida[a:a + bloco] = np.asarray(self.vetores[a:a + bloco], dtype=np.float32) @ vetor
        return saida * self.escala


def quantizar(vetores: np.ndarray) -> tuple[np.ndarray, float]:
    """int8 com uma escala só: metade do float16, e o cosseno muda na terceira casa."""
    escala = float(np.abs(vetores).max()) or 1.0
    return np.round(vetores * (127.0 / escala)).astype(np.int8), escala
