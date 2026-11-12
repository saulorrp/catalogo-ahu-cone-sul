import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.main import criar_app


@pytest.fixture
def client(corpus_busca):
    app = criar_app(dados_injetados=corpus_busca)
    return TestClient(app)


def test_rotas_basicas(client, corpus_busca):
    # /saude
    assert client.get("/saude").status_code == 200
    # /
    assert client.get("/").status_code == 200
    # /busca
    assert client.get("/busca?q=gado").status_code == 200
    # /resultados
    assert client.get("/resultados?q=gado").status_code == 200

    # /doc/{uid}
    uid = corpus_busca.documentos.iloc[0]["doc_uid"]
    assert client.get(f"/doc/{uid}").status_code == 200

    # /pessoas
    assert client.get("/pessoas").status_code == 200

    # /pessoa/{uid}
    pessoa_uid = None
    for _, p in corpus_busca.pessoas.iterrows():
        if p.get("pessoa_uid"):
            pessoa_uid = p["pessoa_uid"]
            break
    if pessoa_uid:
        assert client.get(f"/pessoa/{pessoa_uid}").status_code == 200
    else:
        assert client.get("/pessoa/P9999").status_code == 404

    # /conversor
    assert client.get("/conversor").status_code == 200
    assert client.get("/conversor?q=AHU_ACL_CU_019, Cx. 3, D. 246").status_code == 200

    # /selecao
    assert client.get("/selecao").status_code == 200

    # /selecao/adicionar e remover
    resp = client.post(f"/selecao/adicionar/{uid}")
    assert resp.status_code == 200
    assert "selecao" in resp.cookies

    resp = client.post(f"/selecao/remover/{uid}")
    assert resp.status_code == 200

    # /selecao/pdf
    client.post(f"/selecao/adicionar/{uid}")
    resp = client.get("/selecao/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"

    # /amostras
    assert client.get("/amostras").status_code == 200

    # /painel
    assert client.get("/painel").status_code == 200

    # /sobre
    assert client.get("/sobre").status_code == 200

    # /referencias
    assert client.get("/referencias").status_code == 200

    # /api/busca
    assert client.get("/api/busca?q=gado").status_code == 200

    # /api/doc/{uid}
    assert client.get(f"/api/doc/{uid}").status_code == 200

    # /api/sugerir
    assert client.get("/api/sugerir?campo=remetente&q=junta").status_code == 200


def test_busca_texto_e_cota(client):
    # Busca por texto
    resp = client.get("/api/busca?q=gado")
    assert resp.status_code == 200
    assert resp.json()["total"] > 0

    # Busca por cota
    resp = client.get("/api/busca?q=PT/AHU/CU/021/0006/00390")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_filtros_reduzem_total(client):
    resp_todos = client.get("/api/busca")
    total_todos = resp_todos.json()["total"]

    # Filtro por ano
    resp_ano = client.get("/api/busca?v=2&a1=1705")
    total_ano = resp_ano.json()["total"]
    assert total_ano < total_todos

    # Filtro por destinatário (no mini catálogo todos têm o mesmo remetente)
    resp_rem = client.get("/api/busca?v=2&fd=Rei")
    total_rem = resp_rem.json()["total"]
    assert total_rem < total_todos


def test_doc_mostra_sic_e_digitarq(client, corpus_busca):
    # Find a doc with [sic] if any
    uid_sic = None
    for i, row in corpus_busca.documentos.iterrows():
        resp = client.get(f"/doc/{row['doc_uid']}")
        if "[sic]" in resp.text:
            uid_sic = row["doc_uid"]
            break

    if uid_sic:
        resp = client.get(f"/doc/{uid_sic}")
        assert "[sic]" in resp.text

    # Find doc with DigitArq link
    uid_digitarq = None
    for i, row in corpus_busca.documentos.iterrows():
        if row.get("reference_code") == "PT/AHU/CU/021/0006/00390":
            uid_digitarq = row["doc_uid"]
            break

    assert uid_digitarq is not None
    resp = client.get(f"/doc/{uid_digitarq}")
    assert resp.status_code == 200
    assert "digitarq.arquivos.pt" in resp.text


def test_nenhum_doc_uid_visivel(client, corpus_busca):
    uid = corpus_busca.documentos.iloc[0]["doc_uid"]
    resp = client.get(f"/doc/{uid}")
    assert resp.status_code == 200

    # Remove tags and scripts
    html = resp.text
    html_no_script = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    visivel = re.sub(r"<[^>]*>", " ", html_no_script)

    assert uid not in visivel


def test_nenhuma_mencao_scribe_mediation_likely(client, corpus_busca):
    # Check in all pages
    urls = [
        "/",
        "/busca?q=gado",
        "/resultados?q=gado",
        f"/doc/{corpus_busca.documentos.iloc[0]['doc_uid']}",
        "/pessoas",
        "/conversor",
        "/amostras",
        "/painel",
        "/sobre",
        "/referencias"
    ]
    for url in urls:
        resp = client.get(url)
        assert "scribe_mediation_likely" not in resp.text

    # Also check the source code of web/
    web_dir = Path(__file__).parent.parent / "web"
    for root, _, files in os.walk(web_dir):
        for file in files:
            if file.endswith((".py", ".html", ".js", ".css")):
                content = (Path(root) / file).read_text(encoding="utf-8")
                assert "scribe_mediation_likely" not in content


def test_prefixo_ahu_root_path(monkeypatch, corpus_busca):
    monkeypatch.setenv("AHU_ROOT_PATH", "/ahu")
    app = criar_app(dados_injetados=corpus_busca)
    client = TestClient(app)

    # /ahu/saude
    resp = client.get("/ahu/saude")
    assert resp.status_code == 200

    # /ahu/
    resp = client.get("/ahu/")
    assert resp.status_code == 200
    assert "/ahu/busca" in resp.text or "/ahu/" in resp.text
    assert "/ahu/static/" in resp.text


def test_api_busca_formato(client):
    resp = client.get("/api/busca?q=gado")
    assert resp.status_code == 200
    dados = resp.json()
    assert "total" in dados
    assert "resultados" in dados
    if dados["resultados"]:
        res = dados["resultados"][0]
        assert "cota" in res
        assert "ano" in res


def test_amostras_plano_erro(client):
    # Sem pessoa_uid válido
    resp = client.get("/amostras/plano?tipo=diacronica&pessoa_uid=invalido")
    assert resp.status_code == 400
