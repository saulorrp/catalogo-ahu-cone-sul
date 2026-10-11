"""test_busca_v2.py — Testes do WP 0.3: Reprodução da busca v2 como referência."""
import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

from avaliacao.busca_v2 import (
    MotorBuscaV2,
    _check_authorship,
    _extract_year,
    carregar_dados_legado,
    localizar_arquivos_legado,
)


def test_extracao_ano_e_autoria():
    assert _extract_year("12. 1775 Julho 14.") == 1775
    assert _extract_year("Sem data") is pd.NA

    row_valida = pd.Series({"sender_name": "Lavradio", "description": "Carta do Marquês do Lavradio ao rei."})
    assert _check_authorship(row_valida) is True

    row_invalida = pd.Series({"sender_name": "Lavradio", "description": "Carta de terceiro."})
    assert _check_authorship(row_invalida) is False


def test_motor_v2_logica_de_pesos_e_booster():
    # Mini-catálogo com 3 documentos
    data = [
        {
            "description": "conflito de terras e posse de fazenda",
            "sender_name": "Manoel",
            "folder": "Santa Catarina",
            "reference_code": "PT/AHU/CU/001/0001/00001",
            "new_code": "AHU_ACL_CU_001, Cx. 1, D. 1",
            "old_code": "Doc. 1",
            "vernacular_score": 0.8,
        },
        {
            "description": "deserção de soldados do regimento",
            "sender_name": "Gomes Freire",
            "folder": "Rio de Janeiro",
            "reference_code": "PT/AHU/CU/001/0001/00002",
            "new_code": "AHU_ACL_CU_001, Cx. 1, D. 2",
            "old_code": "Doc. 2",
            "vernacular_score": 0.3,
        },
        {
            "description": "mapa e relação de despesas militares",
            "sender_name": "Lavradio",
            "folder": "Colônia",
            "reference_code": "PT/AHU/CU/001/0001/00003",
            "new_code": "AHU_ACL_CU_001, Cx. 1, D. 3",
            "old_code": "Doc. 3",
            "vernacular_score": 0.5,
        },
    ]
    df = pd.DataFrame(data)
    corpus = [
        f"{d['description']} {d['sender_name']} {d['folder']} "
        f"{d['reference_code']} {d['new_code']} {d['old_code']}".lower()
        for d in data
    ]
    bm25 = BM25Okapi([c.split() for c in corpus])
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ref_codes = [d["reference_code"] for d in data]

    def mock_encoder(q: str) -> np.ndarray:
        if "deserção" in q:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if "terras" in q:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    motor = MotorBuscaV2(
        df=df,
        bm25=bm25,
        embeddings=embeddings,
        reference_codes=ref_codes,
        encoder=mock_encoder,
    )

    # 1. Consulta curta (<= 2 termos)
    res_curta = motor.buscar("deserção soldados", limiar_semantico=0.20)
    assert not res_curta.empty
    assert res_curta.iloc[0]["reference_code"] == "PT/AHU/CU/001/0001/00002"

    # 2. Booster exato de cota
    res_cota = motor.buscar("PT/AHU/CU/001/0001/00003")
    assert not res_cota.empty
    assert res_cota.iloc[0]["reference_code"] == "PT/AHU/CU/001/0001/00003"
    assert res_cota.iloc[0]["semantic_score"] == 1.0

    # 3. Consulta vazia devolve ordenado por vernacular_score
    res_vazia = motor.buscar("")
    assert res_vazia.iloc[0]["vernacular_score"] == 0.8


def test_motor_v2_integracao_com_arquivos_legado():
    try:
        cat_p, idx_p = localizar_arquivos_legado()
    except FileNotFoundError:
        pytest.skip("Arquivos legados não disponíveis no ambiente de teste")

    assert cat_p.is_file()
    assert idx_p.is_file()

    # Testa carregamento do catálogo e criação do BM25
    df, bm25 = carregar_dados_legado(cat_p)
    assert len(df) == 7051

    # Motor com mock encoder para teste rápido sem rodar SentenceTransformer na suíte de testes
    def mock_encoder(q: str) -> np.ndarray:
        return np.ones(1024, dtype=np.float32)

    with open(idx_p, "rb") as f:
        import pickle

        idx_data = pickle.load(f)

    motor = MotorBuscaV2(
        df=df,
        bm25=bm25,
        embeddings=idx_data["embeddings"],
        reference_codes=idx_data["reference_codes"],
        encoder=mock_encoder,
    )

    # Consulta exata por cota encontra o documento pelo booster com escore 1.0
    cota = "PT/AHU/CU/021/0006/00390"
    res = motor.buscar(cota, limiar_semantico=0.50)
    assert not res.empty
    assert res.iloc[0]["reference_code"] == cota
    assert res.iloc[0]["semantic_score"] == 1.0
