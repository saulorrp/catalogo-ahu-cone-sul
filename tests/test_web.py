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


def test_campos_vazios_do_formulario_nao_zeram_a_consulta():
    """O formulário manda a1=&a2=&fu=…; vazio é "sem filtro", não um link inválido."""
    from starlette.requests import Request

    from web.core import parse_consulta

    qs = (b"v=2&q=domingos%20Brito%20Peixoto&m=automatico&r=medio&a1=&a2=&sd=1&fu=&fd="
          b"&s1=&s2=&o=relevancia")
    consulta = parse_consulta(Request({"type": "http", "query_string": qs, "headers": []}))
    assert consulta.texto == "domingos Brito Peixoto"
    assert consulta.filtros.ano_de is None


# ---------------------------------------------------------------------------
# Redesenho (abas, gaveta da seleção, linha do tempo no Sobre) e busca em dois tempos
# ---------------------------------------------------------------------------

def _reranker_invertido(consulta, textos):
    """Um reranker de mentira: prefere os textos mais curtos. Basta para mudar a ordem."""
    import numpy as np
    return np.array([-len(t) for t in textos], dtype=np.float32)


def test_abas_no_topo_e_aba_atual(client):
    html = client.get("/sobre").text
    assert 'class="abas"' in html
    assert re.search(r'class="aba atual"[^>]*aria-current="page"[^>]*>Sobre', html)
    # A linha do tempo saiu da busca e está no Sobre, com os números por década.
    assert 'id="linha-do-tempo"' in html
    assert "histograma-svg-container" not in client.get("/").text


def test_gaveta_da_selecao_na_busca(client, corpus_busca):
    uid = corpus_busca.documentos.iloc[0]["doc_uid"]
    html = client.get("/busca?q=soldados").text
    assert 'id="gaveta-selecao"' in html
    assert "Pôr na seleção" in html

    # Pôr: a resposta traz o botão marcado, a gaveta e o número da aba, de uma vez.
    resp = client.post(f"/selecao/adicionar/{uid}")
    assert "Na seleção" in resp.text
    assert 'id="gaveta-selecao" hx-swap-oob="true"' in resp.text
    assert re.search(r'id="aba-conta"[^>]*hx-swap-oob="true"[^>]*>1<', resp.text)

    # A busca seguinte já mostra o documento como marcado e a gaveta com ele.
    html = client.get("/busca?q=soldados").text
    assert "btn-sel marcado" in html
    assert "sel-lista" in html

    # Tirar pela gaveta devolve o botão do documento fora do lugar clicado.
    resp = client.post(f"/selecao/remover/{uid}?de=gaveta")
    assert 'hx-swap-oob="outerHTML"' in resp.text
    assert "btn-sel marcado" not in resp.text
    assert "btn-sel marcado" not in client.get("/busca?q=soldados").text


def test_selecao_cheia_nao_apaga_o_que_ja_estava(client, corpus_busca, monkeypatch):
    """Passado o teto do cookie, o documento novo não entra e os anteriores ficam."""
    import ahu.selecao as selecao

    uids = list(corpus_busca.documentos["doc_uid"][:3])
    client.post(f"/selecao/adicionar/{uids[0]}")
    client.post(f"/selecao/adicionar/{uids[1]}")
    monkeypatch.setattr(selecao, "LIMITE_URL", 15)
    resp = client.post(f"/selecao/adicionar/{uids[2]}")
    assert "A seleção está cheia" in resp.text
    assert client.get("/selecao").text.count("<tr>") == 3      # cabeçalho + os dois de antes


def test_busca_em_dois_tempos(corpus_busca, monkeypatch):
    import web.main as main_module
    from ahu import busca
    from web import cache

    cache.limpar()
    client = TestClient(criar_app(dados_injetados=corpus_busca))
    monkeypatch.setattr(main_module, "reranker_app", _reranker_invertido)

    # Primeiro tempo: a lista sai na hora, com o pedido da ordem fina embutido.
    html = client.get("/busca?q=soldados").text
    assert "fino=1" in html
    # Segundo tempo: a ordem é a mesma de `buscar` com o reranker.
    fino = client.get("/resultados?q=soldados&fino=1").text
    esperado = busca.buscar(corpus_busca, "soldados", reranker=_reranker_invertido)
    ordem = re.findall(r'/doc/([^"]+)" class="link-ficha"', fino)
    assert ordem == [r.doc_uid for r in esperado.resultados[:25]]
    # Daí em diante a resposta relida está guardada: a página já sai na ordem fina.
    assert "fino=1" not in client.get("/busca?q=soldados").text
    # No modo Exata e na ordenação por ano o reranker não entra.
    assert "fino=1" not in client.get("/busca?q=soldados&m=exata").text
    assert "fino=1" not in client.get("/busca?q=soldados&o=ano_asc").text
    cache.limpar()


def test_refinar_da_o_mesmo_que_buscar_com_reranker(corpus_busca):
    from ahu import busca

    for texto in ("soldados", "deserção da tropa", "AHU_ACL_CU_021, Cx. 6, D. 390 obras do porto"):
        direto = busca.buscar(corpus_busca, texto, reranker=_reranker_invertido)
        base = busca.buscar(corpus_busca, texto)
        antes = [r.doc_uid for r in base.resultados]
        fino = busca.refinar(corpus_busca, texto, base, _reranker_invertido)
        assert [r.doc_uid for r in fino.resultados] == [r.doc_uid for r in direto.resultados]
        assert [r.posicao for r in fino.resultados] == list(range(1, len(fino.resultados) + 1))
        # A resposta de origem fica como estava: ela está na memória de buscas.
        assert [r.doc_uid for r in base.resultados] == antes
        assert all("reranker" not in r.posicoes for r in base.resultados)


def test_reranker_com_memoria_so_le_os_pares_novos():
    from ahu.reranker import com_memoria

    lidos = []

    def falso(consulta, textos):
        lidos.extend(textos)
        return [float(len(t)) for t in textos]

    rr = com_memoria(falso)
    assert list(rr("gado", ["a", "bbb"])) == [1.0, 3.0]
    assert list(rr("gado", ["bbb", "cc", "a"])) == [3.0, 2.0, 1.0]
    assert lidos == ["a", "bbb", "cc"]
    rr("outra consulta", ["a"])
    assert lidos == ["a", "bbb", "cc", "a"]


def test_pagina_da_pessoa_com_campos_vazios(client, corpus_busca):
    """A página quebrava (erro 500) quando um documento da pessoa não tinha tipologia."""
    for uid in corpus_busca.pessoas["pessoa_uid"].dropna():
        resp = client.get(f"/pessoa/{uid}")
        assert resp.status_code == 200
        assert "/busca?v=2" in resp.text      # "Ver os documentos" leva à busca, não à própria página
