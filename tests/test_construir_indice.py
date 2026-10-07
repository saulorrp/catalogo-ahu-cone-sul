"""Testes do WP 1.4: índice semântico reprodutível (com codificador falso, sem baixar o modelo)."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from build import construir_indice as ci
from build.importar_base import importar


class Falso:
    """Codificador determinístico (saco de palavras com hash). Ignora os prefixos passage:/query:."""

    DIM = 256

    def __init__(self, falhar_na_chamada=None):
        self.chamadas = 0
        self.textos_vistos = 0
        self.falhar_na_chamada = falhar_na_chamada

    def __call__(self, textos):
        self.chamadas += 1
        if self.falhar_na_chamada and self.chamadas >= self.falhar_na_chamada:
            raise KeyboardInterrupt("interrompido")
        self.textos_vistos += len(textos)
        saida = np.zeros((len(textos), self.DIM), dtype=np.float32)
        for i, t in enumerate(textos):
            base = t.split(": ", 1)[1] if t.startswith(("passage: ", "query: ")) else t
            for palavra in base.lower().replace(".", " ").split():
                h = int.from_bytes(hashlib.sha256(palavra.encode("utf-8")).digest()[:4], "big")
                saida[i, h % self.DIM] += 1.0
        return saida


def doc(n):
    return {
        "folder": "Pasta", "description": f"Descrição única do documento número {n} sobre o assunto {n * 7}",
        "old_code": None, "new_code": f"AHU_ACL_CU_001, Cx. 1, D. {n}.",
        "reference_code": f"PT/AHU/CU/001/0001/{n:05d}",
        "document_id_and_date": f"{n}. 1700, Janeiro, 1, Lisboa", "ano_isolado": 1700,
        "extracted_typology": "CARTA", "sender_name": f"Remetente {n}", "recipient_name": "Rei",
    }


@pytest.fixture()
def corpus(tmp_path):
    entrada = tmp_path / "e.json"
    entrada.write_text(json.dumps([doc(i) for i in range(1, 31)]), encoding="utf-8")
    importar(entrada, tmp_path / "corpus", "t", esperado=30)
    return tmp_path / "corpus"


def test_montar_texto_receita_do_preprint():
    t = ci.montar_texto("CARTA", "Conde de Sabugosa", "Bahia", "Pede mercê")
    assert t == "passage: CARTA. Conde de Sabugosa. Bahia. Pede mercê"


def test_montar_texto_omite_partes_vazias():
    assert ci.montar_texto(None, "", float("nan"), "texto  com   espaços") == "passage: texto com espaços"
    assert ci.montar_texto(pd.NA, "X", "Y", None) == "passage: X. Y"


def test_nome_da_camada_por_modelo():
    assert ci.nome_camada_para("intfloat/multilingual-e5-large") == "embeddings_e5_v1"
    assert ci.nome_camada_para("intfloat/multilingual-e5-base") == "embeddings_e5base_v1"
    assert ci.nome_camada_para("intfloat/multilingual-e5-small") == "embeddings_e5small_v1"


def test_mesmo_texto_mesmo_vetor(corpus):
    enc = Falso()
    ci.construir_indice(
        corpus, "t", "intfloat/multilingual-e5-large", enc, log=lambda *_: None, tamanho_chunk=7
    )
    a = np.load(corpus / "camadas" / "embeddings_e5_v1" / "embeddings.f16.npy")
    ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), forcar=True,
                        log=lambda *_: None, tamanho_chunk=11)
    b = np.load(corpus / "camadas" / "embeddings_e5_v1" / "embeddings.f16.npy")
    assert a.shape == (30, 256) and a.dtype == np.float16
    assert np.allclose(a.astype(np.float32), b.astype(np.float32), atol=1e-3)  # tolerância de float16


def test_vetores_normalizados(corpus):
    ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), log=lambda *_: None)
    emb = np.load(corpus / "camadas" / "embeddings_e5_v1" / "embeddings.f16.npy").astype(np.float32)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=2e-3)


def test_ordem_dos_uids_e_a_da_base(corpus):
    ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), log=lambda *_: None)
    base = pd.read_parquet(corpus / "colecoes" / "t" / "documentos.parquet")["doc_uid"].tolist()
    indice = pd.read_parquet(corpus / "camadas" / "embeddings_e5_v1" / "indice.parquet")["doc_uid"].tolist()
    assert indice == base


def test_sanidade_busca_pela_propria_descricao(corpus):
    r = ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), log=lambda *_: None)
    assert r["sanidade_aprovada"] is True
    camada = json.loads((corpus / "camadas" / "embeddings_e5_v1" / "camada.json").read_text(encoding="utf-8"))
    assert len(camada["sanidade"]) == 5 and all(s["primeiro"] for s in camada["sanidade"])


def test_sanidade_detecta_indice_trocado(corpus):
    df = pd.read_parquet(corpus / "colecoes" / "t" / "documentos.parquet")
    enc = Falso()
    emb = enc([ci.PREFIXO_DOCUMENTO + d for d in df["description"]])
    emb = ci._normalizar(emb).astype(np.float16)
    trocado = emb[::-1].copy()  # ordem invertida: o alinhamento está errado
    res = ci.verificar_sanidade(df, trocado, enc)
    assert not all(s["aceito"] for s in res)


def test_retomada_nao_recodifica_chunks_prontos(corpus):
    modelo = "intfloat/multilingual-e5-large"
    quebra = Falso(falhar_na_chamada=3)  # cai no 3º chunk
    with pytest.raises(KeyboardInterrupt):
        ci.construir_indice(corpus, "t", modelo, quebra, log=lambda *_: None, tamanho_chunk=5)
    parciais = list((corpus / "camadas" / "embeddings_e5_v1" / "parciais").glob("chunk_*.npz"))
    assert len(parciais) == 2

    retomada = Falso()
    r = ci.construir_indice(corpus, "t", modelo, retomada, log=lambda *_: None, tamanho_chunk=5)
    assert r["documentos"] == 30
    # 30 documentos - 10 já prontos = 20 + 5 consultas de sanidade
    assert retomada.textos_vistos == 20 + 5

    completa = Falso()
    ci.construir_indice(corpus, "t", modelo, completa, forcar=True, log=lambda *_: None, tamanho_chunk=5)
    a = np.load(corpus / "camadas" / "embeddings_e5_v1" / "embeddings.f16.npy")
    assert a.shape[0] == 30
    assert not (corpus / "camadas" / "embeddings_e5_v1" / "parciais").exists()


def test_chunk_corrompido_e_refeito(corpus):
    modelo = "intfloat/multilingual-e5-large"
    with pytest.raises(KeyboardInterrupt):
        ci.construir_indice(
            corpus, "t", modelo, Falso(falhar_na_chamada=3), log=lambda *_: None, tamanho_chunk=5
        )
    pasta = corpus / "camadas" / "embeddings_e5_v1" / "parciais"
    (pasta / "chunk_00000.npz").write_bytes(b"truncado")
    r = ci.construir_indice(corpus, "t", modelo, Falso(), log=lambda *_: None, tamanho_chunk=5)
    assert r["documentos"] == 30


def test_atualizado_pula_e_texto_novo_refaz(corpus):
    modelo = "intfloat/multilingual-e5-large"
    ci.construir_indice(corpus, "t", modelo, Falso(), log=lambda *_: None)
    enc = Falso()
    r = ci.construir_indice(corpus, "t", modelo, enc, log=lambda *_: None)
    assert r["atualizado"] is False and enc.chamadas == 0

    arq = corpus / "colecoes" / "t" / "documentos.parquet"
    df = pd.read_parquet(arq)
    df.loc[0, "description"] = "descrição corrigida"
    df.to_parquet(arq, index=False)
    r = ci.construir_indice(corpus, "t", modelo, Falso(), log=lambda *_: None)
    assert r["atualizado"] is True


def test_limite_grava_camada_amostra_e_nao_ativa(corpus):
    ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), limite=8, log=lambda *_: None)
    manifesto = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    entrada = manifesto["camadas"]["embeddings_e5_v1_amostra"]
    assert entrada["documentos"] == 8 and entrada["ativa"] is False
    assert not (corpus / "camadas" / "embeddings_e5_v1").exists()


def test_manifesto_registra_hashes_e_camada_ativa(corpus):
    ci.construir_indice(corpus, "t", "intfloat/multilingual-e5-large", Falso(), log=lambda *_: None)
    manifesto = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    c = manifesto["camadas"]["embeddings_e5_v1"]
    assert c["ativa"] is True and c["modelo"] == "intfloat/multilingual-e5-large"
    assert all(len(h) == 64 for h in c["arquivos"].values())
    assert "t" in manifesto["colecoes"]  # o manifesto da base foi preservado


def test_base_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="importar_base"):
        ci.construir_indice(tmp_path / "corpus", "t", "m", Falso(), log=lambda *_: None)
