"""Testes do modelo estático da versão leve (ahu/estatico.py, build/construir_estatico.py)."""
import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from ahu import busca, estatico
from build.construir_estatico import cortar

PECAS = [("[PAD]", 0.0), ("[UNK]", 0.0), ("▁", -2.0), ("a", -3.0), ("b", -3.0),
         ("▁gado", -1.0), ("▁cavalo", -1.0), ("▁navio", -1.0), ("▁raro", -1.0)]


def _tokenizer_json() -> dict:
    tok = Tokenizer(models.Unigram(PECAS, unk_id=1))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    return json.loads(tok.to_str())


def test_cortar_mantem_os_usados_os_caracteres_e_o_unk():
    novo, antigos = cortar(_tokenizer_json(), {5})
    pecas = [p for p, _ in novo["model"]["vocab"]]
    assert "▁gado" in pecas and "▁raro" not in pecas and "▁navio" not in pecas
    assert {"[UNK]", "▁", "a", "b"} <= set(pecas)
    assert pecas[novo["model"]["unk_id"]] == "[UNK]"
    assert [PECAS[i][0] for i in antigos] == pecas


@pytest.fixture
def encoder(tmp_path):
    novo, antigos = cortar(_tokenizer_json(), {5, 6, 7})
    pasta = tmp_path / estatico.PASTA
    pasta.mkdir()
    (pasta / "tokenizer.json").write_text(json.dumps(novo), "utf-8")
    pecas = [p for p, _ in novo["model"]["vocab"]]
    matriz = np.zeros((len(pecas), 4), dtype=np.float16)
    for peca, vetor in (("▁gado", [1, 0, 0, 0]), ("▁cavalo", [1, 1, 0, 0]),
                        ("▁navio", [0, 0, 1, 0])):
        matriz[pecas.index(peca)] = vetor
    np.save(pasta / "matriz.f16.npy", matriz)
    meta = {"unk_id": novo["model"]["unk_id"], "cosseno_minimo": 0.1,
            "margens_rigor": {"baixo": 0.5, "medio": 0.4, "alto": 0.3},
            "pesos_rrf": {"curta": {"lexico": 1.0, "semantico": 0.3},
                          "longa": {"lexico": 1.0, "semantico": 0.6}}}
    (pasta / "meta.json").write_text(json.dumps(meta), "utf-8")
    sem_vetores = estatico.EncoderEstatico(tmp_path)
    vetores, meta["escala"] = estatico.quantizar(sem_vetores(["gado", "cavalo", "navio"]))
    np.save(pasta / estatico.ARQUIVO_VETORES, vetores)
    (pasta / "meta.json").write_text(json.dumps(meta), "utf-8")
    return estatico.EncoderEstatico(tmp_path)


def test_consulta_e_a_media_normalizada_dos_pedacos(encoder):
    v = encoder(["gado", "", "gado cavalo"])
    assert np.allclose(np.linalg.norm(v[[0, 2]], axis=1), 1.0)
    assert not v[1].any()  # texto vazio: vetor nulo, sem divisão por zero
    assert v[2][1] > 0 and v[2][0] > v[2][1]  # gado (1,0,0,0) + cavalo (1,1,0,0)


def test_produto_desfaz_a_quantizacao(encoder):
    cos = encoder.produto(encoder(["gado"])[0])
    assert cos.shape == (3,) and abs(cos[0] - 1.0) < 0.02 and int(cos.argmax()) == 0


def test_buscar_usa_os_vetores_e_a_calibracao_do_encoder(encoder):
    from types import SimpleNamespace
    dados = SimpleNamespace(embeddings=None)
    cos = busca.cossenos(dados, "cavalo", encoder)
    assert int(cos.argmax()) == 1
    assert busca.pesos_do_rrf(["gado"], encoder.pesos_rrf) == {"lexico": 1.0, "semantico": 0.3}
