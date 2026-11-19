"""A Lista do eins zwei: a memória dos processos, só no endereço privado (ahu/lista_eins.py)."""
import json

import pytest
from fastapi.testclient import TestClient

from ahu.lista_eins import Lista, cota_curta, cota_valida
from web.main import criar_app

TOKEN = "segredo-de-teste"
PRIVADO = {"X-Eins-Lista": TOKEN}
HTMX = {**PRIVADO, "HX-Request": "true"}


def test_cota_curta_das_tres_formas():
    assert cota_curta("PT/AHU/CU/089/0010/00955") == "089-0010-00955"
    assert cota_curta("PT/AHU/CU/017-001/0057/13257") == "017.001-0057-13257"
    assert cota_curta("AHU_ACL_CU_021, Cx. 6, D. 390.") == "021-0006-00390"
    assert cota_curta("089-10-955 rafael") == "089-0010-00955"
    assert cota_curta("carta qualquer") is None


def test_cota_valida_recusa_caminhos():
    assert cota_valida("089-0010-00955") and cota_valida("017.001-0057-13257")
    for ruim in ("../x", "089-0010-00955/../a", "", "089/0010/00955"):
        assert not cota_valida(ruim)


def test_lista_junta_mudancas_e_guarda_lixeira(tmp_path):
    lista = Lista(tmp_path)
    ficha = {"doc_uid": "ahu:1", "titulo": "Carta", "ano": 1777, "cotas": {"Código de referência": "PT/AHU/CU/089/0010/00955"}}
    r = lista.atualizar("089-0010-00955", {"ficha": ficha})
    assert r["estado"] == "na lista" and r["cota_referencia"].endswith("00955") and r["doc_uid"] == "ahu:1"
    pasta = {"computador": "Hermerico", "nome": "089-0010-00955", "caminho": "C:\\x\\089-0010-00955"}
    lista.atualizar("089-0010-00955", {"estado": "triado", "pasta": pasta, "triagem": {"fotos": []},
                                       "triagem_sha256": "ab", "evento": {"tipo": "gerado"}})
    r = lista.atualizar("089-0010-00955", {"pasta": {**pasta, "caminho": "D:\\y"}})
    assert len(r["pastas"]) == 1 and r["pastas"][0]["caminho"] == "D:\\y"
    assert r["triagem"] == {"fotos": []} and r["estado"] == "triado"
    assert [h["tipo"] for h in r["historico"]] == ["adicionado", "estado", "gerado"]
    assert lista.estados_por_doc() == {"ahu:1": "triado"}
    with pytest.raises(ValueError):
        lista.atualizar("089-0010-00955", {"estado": "perdido"})
    assert lista.remover("089-0010-00955")
    assert lista.ler("089-0010-00955") is None and lista.listar() == []
    assert len(list((tmp_path / "lixeira").glob("089-0010-00955 - *.json"))) == 1
    assert lista.estados_por_doc() == {}


@pytest.fixture
def privado(corpus_busca, tmp_path, monkeypatch):
    monkeypatch.setenv("EINS_LISTA_DIR", str(tmp_path / "lista"))
    monkeypatch.setenv("EINS_LISTA_TOKEN", TOKEN)
    return TestClient(criar_app(dados_injetados=corpus_busca)), tmp_path / "lista"


def _uid_com_cota(corpus):
    docs = corpus.documentos
    return str(docs[docs["reference_code"] == "PT/AHU/CU/021/0006/00390"].iloc[0]["doc_uid"])


def test_lista_desligada_e_fechada(corpus_busca, monkeypatch):
    monkeypatch.delenv("EINS_LISTA_DIR", raising=False)
    monkeypatch.delenv("EINS_LISTA_TOKEN", raising=False)
    client = TestClient(criar_app(dados_injetados=corpus_busca))
    assert client.get("/api/lista", headers=PRIVADO).status_code == 404
    assert "Adicionar à lista" not in client.get(f"/doc/{_uid_com_cota(corpus_busca)}").text


def test_endereco_publico_nao_ve_a_lista(privado, corpus_busca):
    client, _ = privado
    assert client.get("/api/lista").status_code == 403
    assert client.get("/api/lista", headers={"X-Eins-Lista": "errado"}).status_code == 403
    assert client.put("/api/lista/021-0006-00390", json={}).status_code == 403
    uid = _uid_com_cota(corpus_busca)
    assert "Adicionar à lista" not in client.get(f"/doc/{uid}").text
    assert client.post(f"/lista/adicionar/{uid}", headers={"HX-Request": "true"}).status_code == 403


def test_botao_adiciona_e_tira(privado, corpus_busca):
    client, pasta = privado
    uid = _uid_com_cota(corpus_busca)
    assert "Adicionar à lista" in client.get(f"/doc/{uid}", headers=PRIVADO).text
    assert "Adicionar à lista" in client.get("/resultados?q=porto", headers=PRIVADO).text
    # Sem HX-Request (um formulário de outro site) não muda nada.
    assert client.post(f"/lista/adicionar/{uid}", headers=PRIVADO).status_code == 400
    resp = client.post(f"/lista/adicionar/{uid}", headers=HTMX)
    assert resp.status_code == 200 and "Na lista" in resp.text
    registro = json.loads((pasta / "021-0006-00390.json").read_text(encoding="utf-8"))
    assert registro["formato"] == "eins-lista/1" and registro["ficha"]["doc_uid"] == uid
    itens = client.get("/api/lista", headers=PRIVADO).json()["itens"]
    assert [i["cota"] for i in itens] == ["021-0006-00390"]
    assert "Na lista" in client.get(f"/doc/{uid}", headers=PRIVADO).text
    resp = client.post(f"/lista/remover/{uid}", headers=HTMX)
    assert "Adicionar à lista" in resp.text
    assert client.get("/api/lista/021-0006-00390", headers=PRIVADO).status_code == 404


def test_api_do_eins(privado):
    client, _ = privado
    cota = "021-0006-00390"
    mudanca = {"estado": "triado", "pasta": {"computador": "PC", "nome": cota, "caminho": None},
               "triagem": {"cota": cota}, "triagem_sha256": "00", "evento": {"tipo": "importado"}}
    assert client.put(f"/api/lista/{cota}", content=json.dumps(mudanca), headers=PRIVADO).status_code == 415
    resp = client.put(f"/api/lista/{cota}", json=mudanca, headers=PRIVADO)
    assert resp.status_code == 200
    r = resp.json()
    # Um processo importado de uma pasta, sem passar pelo botão, ganha a ficha do catálogo.
    assert r["ficha"] and r["cota_referencia"] == "PT/AHU/CU/021/0006/00390" and r["estado"] == "triado"
    assert client.get(f"/api/lista/{cota}", headers=PRIVADO).json()["triagem"] == {"cota": cota}
    assert client.put("/api/lista/..%2Fx", json={}, headers=PRIVADO).status_code in (400, 404)
    assert client.put(f"/api/lista/{cota}", json={"estado": "x"}, headers=PRIVADO).status_code == 400
    assert client.delete(f"/api/lista/{cota}", headers=PRIVADO).status_code == 204
    assert client.delete(f"/api/lista/{cota}", headers=PRIVADO).status_code == 404
