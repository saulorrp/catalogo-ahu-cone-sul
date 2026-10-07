"""Testes do WP 1.1: importação da base e doc_uid estáveis."""
import copy
import json

import pandas as pd
import pytest

from build.importar_base import ErroImportacao, importar, normalizar, numero_item


def doc(n, ref=None, new=None, desc=None):
    return {
        "folder": "Pasta", "description": desc or f"Descrição do documento {n}",
        "old_code": None, "new_code": new if new is not None else f"AHU_ACL_CU_001, Cx. 1, D. {n}.",
        "reference_code": ref or f"PT/AHU/CU/001/0001/{n:05d}",
        "document_id_and_date": f"{n}. 1700, Janeiro, 1, Lisboa", "ano_isolado": 1700,
        "extracted_typology": "CARTA", "sender_name": "Fulano", "recipient_name": "Rei",
    }


def escrever(caminho, docs):
    caminho.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    return caminho


def uids(corpus):
    return list(pd.read_parquet(corpus / "colecoes" / "t" / "documentos.parquet")["doc_uid"])


def test_normalizar_e_numero_item():
    assert normalizar("  Ação  DE   Graças ") == "acao de gracas"
    assert normalizar(None) == ""
    assert numero_item("2673. 1686, Fevereiro, 12, Lisboa") == "2673"
    assert numero_item("sem número") == ""


def test_importa_e_uids_unicos(tmp_path):
    docs = [doc(i) for i in range(1, 51)]
    entrada = escrever(tmp_path / "e.json", docs)
    r = importar(entrada, tmp_path / "corpus", "t", esperado=50)
    assert r["documentos"] == 50 and r["uids_unicos"] == 50
    df = pd.read_parquet(tmp_path / "corpus" / "colecoes" / "t" / "documentos.parquet")
    assert df["doc_uid"].str.startswith("t:").all()
    assert (tmp_path / "corpus" / "manifest.json").exists()
    manifesto = json.loads((tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    assert all(len(h) == 64 for h in manifesto["colecoes"]["t"]["arquivos"].values())


def test_reimportar_duas_vezes_da_os_mesmos_uids(tmp_path):
    entrada = escrever(tmp_path / "e.json", [doc(i) for i in range(1, 21)])
    importar(entrada, tmp_path / "c", "t", esperado=20)
    primeira = uids(tmp_path / "c")
    importar(entrada, tmp_path / "c", "t", esperado=20)
    assert uids(tmp_path / "c") == primeira


def test_corrigir_virgula_na_descricao_mantem_uid(tmp_path):
    docs = [doc(i) for i in range(1, 11)]
    importar(escrever(tmp_path / "e.json", docs), tmp_path / "c", "t", esperado=10)
    antes = uids(tmp_path / "c")
    docs2 = copy.deepcopy(docs)
    docs2[4]["description"] = docs2[4]["description"] + ","
    importar(escrever(tmp_path / "e2.json", docs2), tmp_path / "c", "t", esperado=10)
    assert uids(tmp_path / "c") == antes


def test_ordem_diferente_mantem_uid_do_mesmo_documento(tmp_path):
    docs = [doc(i) for i in range(1, 11)]
    importar(escrever(tmp_path / "e.json", docs), tmp_path / "c", "t", esperado=10)
    antes = dict(zip([d["reference_code"] for d in docs], uids(tmp_path / "c")))
    inv = list(reversed(docs))
    importar(escrever(tmp_path / "e2.json", inv), tmp_path / "c", "t", esperado=10)
    depois = dict(zip([d["reference_code"] for d in inv], uids(tmp_path / "c")))
    assert depois == antes


def test_reference_code_duplicado_recebe_uids_diferentes(tmp_path):
    a, b = doc(1, ref="PT/AHU/CU/001/0001/00001"), doc(2, ref="PT/AHU/CU/001/0001/00001")
    importar(escrever(tmp_path / "e.json", [a, b]), tmp_path / "c", "t", esperado=2)
    u = uids(tmp_path / "c")
    assert len(set(u)) == 2


def test_documentos_totalmente_identicos_nao_colidem(tmp_path):
    a = doc(1)
    importar(escrever(tmp_path / "e.json", [a, copy.deepcopy(a)]), tmp_path / "c", "t", esperado=2)
    u = uids(tmp_path / "c")
    assert len(set(u)) == 2
    importar(escrever(tmp_path / "e.json", [a, copy.deepcopy(a)]), tmp_path / "c", "t", esperado=2)
    assert uids(tmp_path / "c") == u


def test_uid_de_documento_removido_nao_e_reaproveitado(tmp_path):
    docs = [doc(i) for i in range(1, 6)]
    importar(escrever(tmp_path / "e.json", docs), tmp_path / "c", "t", esperado=5)
    removido = uids(tmp_path / "c")[0]
    importar(escrever(tmp_path / "e2.json", docs[1:]), tmp_path / "c", "t", esperado=4)
    assert removido not in uids(tmp_path / "c")
    arq = tmp_path / "c" / "colecoes" / "t" / "registro_ids.json"
    registro = json.loads(arq.read_text(encoding="utf-8"))
    assert removido in {r["doc_uid"] for r in registro["documentos"]}


def test_nulos_e_tipos(tmp_path):
    d = doc(1)
    d["ano_isolado"] = None
    d["new_code"] = None
    importar(escrever(tmp_path / "e.json", [d]), tmp_path / "c", "t", esperado=1)
    df = pd.read_parquet(tmp_path / "c" / "colecoes" / "t" / "documentos.parquet")
    assert pd.isna(df.loc[0, "ano_isolado"]) and pd.isna(df.loc[0, "new_code"])


def test_contrato_numero_errado_de_documentos(tmp_path):
    entrada = escrever(tmp_path / "e.json", [doc(i) for i in range(1, 4)])
    with pytest.raises(ErroImportacao, match="Esperados 5"):
        importar(entrada, tmp_path / "c", "t", esperado=5)


def test_contrato_json_truncado(tmp_path):
    entrada = tmp_path / "e.json"
    entrada.write_text('[{"description": "x", "reference_code"', encoding="utf-8")
    with pytest.raises(ErroImportacao, match="JSON inválido"):
        importar(entrada, tmp_path / "c", "t", esperado=None)


def test_contrato_campo_do_pipeline_antigo(tmp_path):
    d = doc(1)
    d["perplexity_search_data"] = {"x": 1}
    with pytest.raises(ErroImportacao, match="perplexity_search_data"):
        importar(escrever(tmp_path / "e.json", [d]), tmp_path / "c", "t", esperado=1)


def test_contrato_campo_obrigatorio_ausente(tmp_path):
    d = doc(1)
    del d["reference_code"]
    with pytest.raises(ErroImportacao, match="reference_code"):
        importar(escrever(tmp_path / "e.json", [d]), tmp_path / "c", "t", esperado=1)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ErroImportacao, match="não encontrado"):
        importar(tmp_path / "nao.json", tmp_path / "c", "t", esperado=None)
