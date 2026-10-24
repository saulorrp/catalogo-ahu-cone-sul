"""construir_estatico.py — o modelo estático da versão leve (ahu/estatico.py).

Entrada : app_dados/documentos.parquet (depois do `construir_app.py`)
Saída   : app_dados/estatico/ (meta.json, tokenizer.json, matriz.f16.npy, vetores.i8.npy)

    pip install model2vec==0.10.0
    python build/construir_estatico.py

Corta o vocabulário do modelo aos pedaços que o corpus usa, mais os caracteres soltos (para que
qualquer palavra da consulta ainda se decomponha em algo conhecido), e codifica os documentos
**com o tokenizador já cortado**, o mesmo que o app usa na consulta. Leva ~1 minuto em CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from ahu.estatico import (  # noqa: E402
    ARQUIVO_VETORES,
    PASTA,
    VERSAO,
    EncoderEstatico,
    montar_texto,
    quantizar,
)

MODELO_PADRAO = "minishlab/potion-multilingual-128M"

# Calibração, medida em 23/10/2026 com 167 mil documentos. O melhor documento de uma consulta com
# sentido fica entre 0,38 e 0,75; abaixo de 0,30 não há o que distinguir. O cosseno se espalha
# mais que o do e5: a margem 0,02 do e5 deixa 1 documento, e 0,10 deixa de 2 a 48 ("fuga de
# escravos", "deserção de soldados"), a faixa que a margem média do e5 dá (8 a 149).
COSSENO_MINIMO = 0.30
MARGENS_RIGOR = {"baixo": 0.15, "medio": 0.10, "alto": 0.06}
# Mais fraco que o e5 (média de palavras, sem contexto): é apoio ao léxico, nunca o manda.
PESOS_RRF = {"curta": {"lexico": 1.0, "semantico": 0.3},
             "longa": {"lexico": 1.0, "semantico": 0.6}}


def textos(df: pd.DataFrame) -> list[str]:
    return [montar_texto(t, s, f, d) for t, s, f, d in
            zip(df["tipologia"], df["sender_name"], df["folder"], df["description"])]


def cortar(tokenizer_json: dict, manter: set[int]) -> tuple[dict, np.ndarray]:
    """Tokenizador só com os pedaços de `manter`, e a ordem antiga de cada pedaço novo."""
    modelo = tokenizer_json["model"]
    if modelo["type"] != "Unigram":
        raise SystemExit(f"Só sei cortar Unigram; o modelo é {modelo['type']}.")
    vocab = modelo["vocab"]
    manter |= {i for i, (peca, _) in enumerate(vocab) if len(peca.lstrip("▁")) <= 1}
    manter |= {t["id"] for t in tokenizer_json.get("added_tokens", [])}
    manter.add(modelo["unk_id"])
    antigos = np.array(sorted(manter), dtype=np.int64)
    novo_de = {int(a): n for n, a in enumerate(antigos)}
    novo = dict(tokenizer_json)
    novo["model"] = dict(modelo, vocab=[vocab[a] for a in antigos], unk_id=novo_de[modelo["unk_id"]])
    novo["added_tokens"] = [dict(t, id=novo_de[t["id"]]) for t in tokenizer_json.get("added_tokens", [])]
    novo["post_processor"] = None  # a consulta é codificada sem tokens especiais
    return novo, antigos


def construir(pasta_app: Path, modelo: str = MODELO_PADRAO) -> dict:
    from huggingface_hub import snapshot_download
    from model2vec import StaticModel
    from tokenizers import Tokenizer

    caminho = Path(snapshot_download(modelo))
    original = StaticModel.from_pretrained(str(caminho))
    if original.weights is not None or original.token_mapping is not None:
        raise SystemExit("Modelo com pesos ou mapeamento de tokens: o corte não sabe tratar.")
    df = pd.read_parquet(pasta_app / "documentos.parquet",
                         columns=["tipologia", "sender_name", "folder", "description"])
    corpus = textos(df)

    completo = Tokenizer.from_file(str(caminho / "tokenizer.json"))
    usados: set[int] = set()
    for a in range(0, len(corpus), 5000):
        for cod in completo.encode_batch(corpus[a:a + 5000], add_special_tokens=False):
            usados.update(cod.ids)
    del completo

    tok_json = json.loads((caminho / "tokenizer.json").read_text("utf-8"))
    novo, antigos = cortar(tok_json, usados)
    saida = pasta_app / PASTA
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "tokenizer.json").write_text(json.dumps(novo, ensure_ascii=False), "utf-8")
    np.save(saida / "matriz.f16.npy", np.asarray(original.embedding)[antigos].astype(np.float16))
    del original

    meta = {"versao": VERSAO, "modelo": modelo, "revisao": caminho.name,
            "dimensao": int(np.load(saida / "matriz.f16.npy", mmap_mode="r").shape[1]),
            "pecas": len(antigos), "pecas_original": len(tok_json["model"]["vocab"]),
            "unk_id": novo["model"]["unk_id"], "documentos": len(corpus),
            "cosseno_minimo": COSSENO_MINIMO, "margens_rigor": MARGENS_RIGOR,
            "pesos_rrf": PESOS_RRF,
            "receita": "<tipologia>. <sender_name>. <folder>. <description> (vazias omitidas)"}
    (saida / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    encoder = EncoderEstatico(pasta_app)
    vetores = np.concatenate([encoder(corpus[a:a + 5000]) for a in range(0, len(corpus), 5000)])
    inteiros, meta["escala"] = quantizar(vetores)
    np.save(saida / ARQUIVO_VETORES, inteiros)
    (saida / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app-dados", type=Path, default=RAIZ / "app_dados")
    ap.add_argument("--modelo", default=MODELO_PADRAO)
    args = ap.parse_args()
    print(json.dumps(construir(args.app_dados, args.modelo), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
