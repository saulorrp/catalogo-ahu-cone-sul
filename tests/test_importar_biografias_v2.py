"""Testes do WP 1.3: camadas entidades_v2 e biografias_v2."""
import csv
import json

import pandas as pd
import pytest

from build import importar_biografias_v2 as ib
from build.importar_base import ErroImportacao, importar

# Níveis do relatório do p4 do autor (2026-10-06), conferidos no fim deste arquivo.
POR_NIVEL_P4 = {"ALTA": 192, "MEDIA": 61, "BAIXA": 36, "REJEITADO": 8,
                "IDENTIFICADO_SEM_LOCAL": 274, "NAO_ENCONTRADO": 788}

PIPELINE_REAL = ib.PIPELINE_PADRAO


# ---------------------------------------------------------------------------
# Pipeline de mentira: seis arquivos pequenos, com os casos que interessam
# ---------------------------------------------------------------------------

def documento(n, sender, remetentes):
    return {
        "folder": "Pasta", "description": f"Descrição do documento {n}",
        "old_code": None, "new_code": f"AHU_ACL_CU_001, Cx. 1, D. {n}.",
        "reference_code": f"PT/AHU/CU/001/0001/{n:05d}",
        "document_id_and_date": f"{n}. 17{n:02d}, Janeiro, 1, Lisboa", "ano_isolado": 1700 + n,
        "extracted_typology": "CARTA", "typology_status": True,
        "sender_name": sender, "recipient_name": "Rei",
        "sender_category": "Local Elite", "vector": "Bottom-Up",
        "typology_impact": "Narrative", "scribe_mediation_likely": False, "vernacular_score": 3,
        "sociolinguistic_reasoning_by_deepseek_v3": "Narrativa local.",
        "autoria_verificada": True, "remetentes": remetentes,
    }


def pessoa(pid, nome, titulos=(), ano_min=1700, ano_max=1700, n_documentos=1):
    return {
        "id": pid, "tipo": "pessoa" if nome else "titulo", "nome_principal": nome,
        "variantes_nome": [], "titulos": list(titulos), "cargos": ["governador"],
        "ano_min": ano_min, "ano_max": ano_max, "n_documentos": n_documentos,
        "locais_escrita": ["Bahia"], "destinatarios": ["Rei"], "categorias": ["Local Elite"],
        "nome_curto": False, "trechos": [],
    }


def biografia(pid, nivel, **extra):
    base = {
        "id": pid, "nome_dossie": pid, "ano_min": 1700, "ano_max": 1700, "n_documentos": 1,
        "nome_curto": False, "checagens": {"V1_nome": "nome compatível"},
        "nome_na_fonte": "Nome Na Fonte", "ano_nascimento": 1660, "ano_morte": 1720,
        "cidade_nascimento": "Lisboa", "regiao_nascimento": None,
        "pais_nascimento_epoca": "Reino de Portugal", "pais_atual": "Portugal",
        "url_fonte": "https://exemplo.pt/x", "trecho_fonte": "nasceu em Lisboa",
        "justificativa_identidade": "Compatível.", "confianca_modelo": "alta",
        "confianca_calculada": nivel,
    }
    base.update(extra)
    return base


@pytest.fixture()
def pipeline(tmp_path):
    """Monta a pasta do pipeline e a base do WP 1.1, e devolve (pasta, corpus, dados)."""
    docs = [
        documento(1, "Conselho Ultramarino",
                  [{"nome": "Conselho Ultramarino", "tipo": "instituicao", "pessoa_id": None}]),
        documento(2, "D. Manoel de Mello",
                  [{"nome": "D. Manoel de Mello", "tipo": "pessoa", "pessoa_id": "P0001"}]),
        documento(3, "Conde de Basto e João Dias",
                  [{"nome": "Conde de Basto", "tipo": "titulo", "pessoa_id": "P0002"},
                   {"nome": "João Dias", "tipo": "pessoa", "pessoa_id": "P0003"}]),
    ]
    dados = {
        "ahu_sul_catalog_biografias.json": docs,
        "entidades_por_remetente.json": {
            "Conselho Ultramarino": [
                {"nome": "Conselho Ultramarino", "tipo": "instituicao", "titulos": [], "cargos": [],
                 "notas": []}],
            "D. Manoel de Mello": [
                {"nome": "D. Manoel de Mello", "tipo": "pessoa", "titulos": ["barão de Laguna"],
                 "cargos": ["capitão-general"], "notas": []}],
            "Conde de Basto e João Dias": [
                {"nome": "Conde de Basto", "tipo": "titulo", "titulos": ["conde de Basto"],
                 "cargos": [], "notas": []},
                {"nome": "João Dias", "tipo": "pessoa", "titulos": [], "cargos": [], "notas": []}],
        },
        "doc_pessoas.json": [[None], ["P0001"], ["P0002", "P0003"]],
        "pessoas.json": [
            pessoa("P0001", "D. Manoel de Mello", ["barão de Laguna"]),
            pessoa("P0002", None, ["conde de Basto"], ano_min=1701, ano_max=1701),
            pessoa("P0003", "João Dias", ano_min=1702, ano_max=1702),
        ],
        "biografias_validadas.json": [
            biografia("P0001", "ALTA", wikidata={"qid": "Q1", "nascimento": 1660, "morte": 1720,
                                                 "local": "Lisboa", "pais": "Portugal"}),
            biografia("P0002", "REJEITADO", motivo_rejeicao="título não aparece na fonte"),
            biografia("P0003", "NAO_ENCONTRADO", cidade_nascimento=None, pais_atual=None,
                      ano_nascimento=None, url_fonte=None, confianca_modelo="baixa"),
        ],
    }
    pasta = tmp_path / "pipeline"
    pasta.mkdir()
    for nome, conteudo in dados.items():
        (pasta / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    escrever_revisao(pasta, {"P0001": "", "P0002": "", "P0003": ""})

    corpus = tmp_path / "corpus"
    importar(pasta / "ahu_sul_catalog_biografias.json", corpus, "t", esperado=len(docs))
    return pasta, corpus, dados


def escrever_revisao(pasta, decisoes: dict[str, str]):
    with open(pasta / "revisao_biografias.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "decisao_manual", "confianca_calculada", "nome_dossie"])
        for pid, decisao in decisoes.items():
            w.writerow([pid, decisao, "", pid])


def reescrever(pasta, nome, conteudo):
    (pasta / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Contrato de entrada (spec §3.2)
# ---------------------------------------------------------------------------

def test_arquivo_truncado_para_a_importacao_com_mensagem_clara(pipeline):
    pasta, corpus, _ = pipeline
    bruto = (pasta / "pessoas.json").read_text(encoding="utf-8")
    (pasta / "pessoas.json").write_text(bruto[: len(bruto) // 2], encoding="utf-8")
    with pytest.raises(ErroImportacao, match="pessoas.json está truncado"):
        ib.importar(pasta, corpus, "t")


def test_arquivo_ausente_para_a_importacao(pipeline):
    pasta, corpus, _ = pipeline
    (pasta / "biografias_validadas.json").unlink()
    with pytest.raises(ErroImportacao, match="não encontrado"):
        ib.importar(pasta, corpus, "t")


def test_doc_pessoas_fora_de_ordem_para_a_importacao(pipeline):
    pasta, corpus, dados = pipeline
    reescrever(pasta, "doc_pessoas.json", [[None], ["P0001"], ["P0003", "P0002"]])
    with pytest.raises(ErroImportacao, match=r"\(2\) Documento 2"):
        ib.importar(pasta, corpus, "t")


def test_doc_pessoas_com_outro_numero_de_documentos_para_a_importacao(pipeline):
    pasta, corpus, _ = pipeline
    reescrever(pasta, "doc_pessoas.json", [[None], ["P0001"]])
    with pytest.raises(ErroImportacao, match=r"\(2\) doc_pessoas.json"):
        ib.importar(pasta, corpus, "t")


def test_pessoa_id_sem_biografia_para_a_importacao(pipeline):
    pasta, corpus, dados = pipeline
    bio = [b for b in dados["biografias_validadas.json"] if b["id"] != "P0003"]
    reescrever(pasta, "biografias_validadas.json", bio)
    with pytest.raises(ErroImportacao, match=r"\(3\).*biografias_validadas.json"):
        ib.importar(pasta, corpus, "t")


def test_pessoa_id_fora_de_pessoas_json_para_a_importacao(pipeline):
    pasta, corpus, dados = pipeline
    reescrever(pasta, "pessoas.json", dados["pessoas.json"][:2])
    with pytest.raises(ErroImportacao, match=r"\(3\).*pessoas.json"):
        ib.importar(pasta, corpus, "t")


def test_campo_do_pipeline_antigo_para_a_importacao(pipeline):
    pasta, corpus, dados = pipeline
    docs = [dict(d) for d in dados["ahu_sul_catalog_biografias.json"]]
    docs[1]["perplexity_search_data"] = {"x": 1}
    reescrever(pasta, "ahu_sul_catalog_biografias.json", docs)
    with pytest.raises(ErroImportacao, match=r"\(4\).*perplexity_search_data"):
        ib.importar(pasta, corpus, "t")


def test_decisao_manual_desconhecida_para_a_importacao(pipeline):
    pasta, corpus, _ = pipeline
    escrever_revisao(pasta, {"P0001": "talvez", "P0002": "", "P0003": ""})
    with pytest.raises(ErroImportacao, match="decisao_manual"):
        ib.importar(pasta, corpus, "t")


def test_csv_de_revisao_sem_a_pessoa_para_a_importacao(pipeline):
    pasta, corpus, _ = pipeline
    escrever_revisao(pasta, {"P0001": "", "P0002": ""})
    with pytest.raises(ErroImportacao, match="CSV de revisão"):
        ib.importar(pasta, corpus, "t")


def test_nivel_desconhecido_para_a_importacao(pipeline):
    pasta, corpus, dados = pipeline
    bio = [dict(b) for b in dados["biografias_validadas.json"]]
    bio[0]["confianca_calculada"] = "MUITO_ALTA"
    reescrever(pasta, "biografias_validadas.json", bio)
    with pytest.raises(ErroImportacao, match="confianca_calculada"):
        ib.importar(pasta, corpus, "t")


# ---------------------------------------------------------------------------
# pessoa_uid (spec §3.3)
# ---------------------------------------------------------------------------

def test_assinatura_ignora_variante_grafica_e_tratamento():
    a = ib.assinatura_pessoa(pessoa("P1", "D. Manoel de Mello e Castro"))
    b = ib.assinatura_pessoa(pessoa("P9", "Manuel de Melo Castro"))
    assert a == b


def test_assinatura_separa_pessoas_de_periodos_diferentes():
    a = ib.assinatura_pessoa(pessoa("P1", "João Dias", ano_min=1700, ano_max=1700))
    b = ib.assinatura_pessoa(pessoa("P1", "João Dias", ano_min=1780, ano_max=1790))
    assert a != b


def test_a_mesma_entrada_duas_vezes_da_os_mesmos_pessoa_uid(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    primeira = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "pessoas.parquet")
    ib.importar(pasta, corpus, "t")
    segunda = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "pessoas.parquet")
    pd.testing.assert_frame_equal(primeira, segunda)


def test_pessoa_uid_sobrevive_a_renumeracao_do_p2(pipeline):
    """O p2 renumera P0001... a cada execução; o pessoa_uid não pode mudar com isso."""
    pasta, corpus, dados = pipeline
    ib.importar(pasta, corpus, "t")
    antes = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "pessoas.parquet")
    uid_antes = dict(zip(antes["nome_exibicao"], antes["pessoa_uid"]))

    # Mesmas pessoas, ids trocados (P0001 -> P0007 etc.), na ordem inversa.
    novo = {"P0001": "P0007", "P0002": "P0008", "P0003": "P0009"}
    pessoas = [{**p, "id": novo[p["id"]]} for p in reversed(dados["pessoas.json"])]
    bios = [{**b, "id": novo[b["id"]]} for b in dados["biografias_validadas.json"]]
    docs = [{**d, "remetentes": [{**r, "pessoa_id": novo.get(r["pessoa_id"]) if r["pessoa_id"] else None}
                                 for r in d["remetentes"]]}
            for d in dados["ahu_sul_catalog_biografias.json"]]
    reescrever(pasta, "pessoas.json", pessoas)
    reescrever(pasta, "biografias_validadas.json", bios)
    reescrever(pasta, "ahu_sul_catalog_biografias.json", docs)
    reescrever(pasta, "doc_pessoas.json", [[None], ["P0007"], ["P0008", "P0009"]])
    escrever_revisao(pasta, {"P0007": "", "P0008": "", "P0009": ""})

    ib.importar(pasta, corpus, "t")
    depois = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "pessoas.parquet")
    assert dict(zip(depois["nome_exibicao"], depois["pessoa_uid"])) == uid_antes
    assert set(depois["pessoa_id_pipeline"]) == {"P0007", "P0008", "P0009"}


def test_pessoa_que_saiu_nao_devolve_o_uid(pipeline):
    pasta, corpus, dados = pipeline
    ib.importar(pasta, corpus, "t")
    registro = json.loads(
        (corpus / "camadas" / ib.CAMADA_ENTIDADES / "registro_pessoas.json").read_text("utf-8"))
    uid_de_joao = next(r["pessoa_uid"] for r in registro["pessoas"] if r["nome"] == "João Dias")

    # João Dias sai do corpus; uma pessoa nova entra no seu lugar.
    docs = [dict(d) for d in dados["ahu_sul_catalog_biografias.json"]]
    docs[2]["remetentes"] = [{"nome": "Conde de Basto", "tipo": "titulo", "pessoa_id": "P0002"}]
    reescrever(pasta, "ahu_sul_catalog_biografias.json", docs)
    reescrever(pasta, "doc_pessoas.json", [[None], ["P0001"], ["P0002"]])
    reescrever(pasta, "pessoas.json", dados["pessoas.json"][:2])
    reescrever(pasta, "biografias_validadas.json", dados["biografias_validadas.json"][:2])
    escrever_revisao(pasta, {"P0001": "", "P0002": ""})
    ib.importar(pasta, corpus, "t")

    registro = json.loads(
        (corpus / "camadas" / ib.CAMADA_ENTIDADES / "registro_pessoas.json").read_text("utf-8"))
    aposentada = next(r for r in registro["pessoas"] if r["pessoa_uid"] == uid_de_joao)
    assert aposentada["pessoa_id_atual"] is None
    assert sum(1 for r in registro["pessoas"] if r["pessoa_id_atual"]) == 2


def test_registro_de_outra_versao_de_assinatura_para_a_importacao(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    arq = corpus / "camadas" / ib.CAMADA_ENTIDADES / "registro_pessoas.json"
    registro = json.loads(arq.read_text("utf-8"))
    registro["assinatura_versao"] = ib.ASSINATURA_VERSAO + 1
    arq.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ErroImportacao, match="assinatura versão"):
        ib.importar(pasta, corpus, "t")


# ---------------------------------------------------------------------------
# Camada entidades_v2
# ---------------------------------------------------------------------------

def test_entidades_tem_doc_uid_posicao_nome_tipo_e_pessoa_uid(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    e = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "entidades.parquet")
    base = pd.read_parquet(corpus / "colecoes" / "t" / "documentos.parquet", columns=["doc_uid"])

    assert list(e.columns) == ["doc_uid", "posicao", "nome", "tipo", "pessoa_uid", "titulos",
                               "cargos", "autoria_verificada"]
    assert len(e) == 4  # 1 + 1 + 2 entidades
    assert e["doc_uid"].isin(base["doc_uid"]).all()
    assert e.loc[e["tipo"] == "instituicao", "pessoa_uid"].isna().all()
    assert e.loc[e["nome"] == "D. Manoel de Mello", "titulos"].iloc[0].tolist() == ["barão de Laguna"]
    terceiro = e[e["doc_uid"] == e["doc_uid"].iloc[-1]]
    assert terceiro["posicao"].tolist() == [0, 1]


def test_pessoas_parquet_usa_pessoa_uid_e_guarda_o_id_do_pipeline(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    p = pd.read_parquet(corpus / "camadas" / ib.CAMADA_ENTIDADES / "pessoas.parquet")
    assert p["pessoa_uid"].is_unique and len(p) == 3
    assert p["pessoa_uid"].str.startswith("pes:").all()
    # Pessoa do tipo título não tem nome principal, mas tem nome de exibição.
    titulo = p[p["tipo"] == "titulo"].iloc[0]
    assert pd.isna(titulo["nome_principal"]) and titulo["nome_exibicao"] == "conde de Basto"
    assert set(p["pessoa_id_pipeline"]) == {"P0001", "P0002", "P0003"}


# ---------------------------------------------------------------------------
# Camada biografias_v2 (D2)
# ---------------------------------------------------------------------------

def test_rejeitado_fica_na_camada_mas_nao_e_exibivel(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    b = pd.read_parquet(corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "biografias.parquet")

    rejeitado = b[b["nivel"] == "REJEITADO"]
    assert len(rejeitado) == 1
    assert not bool(rejeitado["exibivel"].iloc[0])
    assert rejeitado["motivo_rejeicao"].iloc[0] == "título não aparece na fonte"
    assert bool(b.loc[b["nivel"] == "ALTA", "exibivel"].iloc[0])


def test_decisao_manual_manda_nos_dois_sentidos(pipeline):
    pasta, corpus, _ = pipeline
    escrever_revisao(pasta, {"P0001": "rejeitar", "P0002": "aceitar", "P0003": ""})
    ib.importar(pasta, corpus, "t")
    b = pd.read_parquet(corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "biografias.parquet")
    por_id = b.set_index("pessoa_id_pipeline")
    assert not bool(por_id.loc["P0001", "exibivel"])  # ALTA, mas rejeitada à mão
    assert bool(por_id.loc["P0002", "exibivel"])      # REJEITADO, mas aceita à mão
    assert por_id.loc["P0002", "decisao_manual"] == "aceitar"


def test_rotulos_nunca_dizem_desconhecida(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    b = pd.read_parquet(corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "biografias.parquet")
    rotulos = set(b["rotulo"])
    assert "Não identificado" in rotulos
    assert not any("esconhecida" in r for r in ib.ROTULOS.values())
    # Decisão do autor, registrada na spec §6.3.
    assert ib.ROTULOS["IDENTIFICADO_SEM_LOCAL"] == "Identificado, local não encontrado"
    assert ib.ROTULOS["NAO_ENCONTRADO"] == ib.ROTULOS["REJEITADO"] == "Não identificado"
    assert ib.ROTULOS["BAIXA"] == "Nascimento não confirmado"


def test_biografia_traz_fonte_origem_e_wikidata(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")
    b = pd.read_parquet(corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "biografias.parquet")
    alta = b[b["nivel"] == "ALTA"].iloc[0]
    assert alta["cidade_nascimento"] == "Lisboa"
    assert alta["pais_nascimento_epoca"] == "Reino de Portugal" and alta["pais_atual"] == "Portugal"
    assert alta["ano_nascimento"] == 1660 and str(b["ano_nascimento"].dtype) == "Int64"
    assert alta["url_fonte"] == "https://exemplo.pt/x"
    assert alta["origem_local"] == "fonte"
    assert alta["wikidata_qid"] == "Q1" and alta["wikidata_local"] == "Lisboa"

    nao_encontrado = b[b["nivel"] == "NAO_ENCONTRADO"].iloc[0]
    assert pd.isna(nao_encontrado["cidade_nascimento"]) and nao_encontrado["origem_local"] is None


def test_origem_local_da_wikidata_e_preservada(pipeline):
    pasta, corpus, dados = pipeline
    bios = [dict(b) for b in dados["biografias_validadas.json"]]
    bios[0]["origem_local"] = "wikidata"
    reescrever(pasta, "biografias_validadas.json", bios)
    ib.importar(pasta, corpus, "t")
    b = pd.read_parquet(corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "biografias.parquet")
    assert b.set_index("pessoa_id_pipeline").loc["P0001", "origem_local"] == "wikidata"


# ---------------------------------------------------------------------------
# Proveniência
# ---------------------------------------------------------------------------

def test_camadas_e_manifesto_com_proveniencia_e_hash(pipeline):
    pasta, corpus, _ = pipeline
    ib.importar(pasta, corpus, "t")

    ent = json.loads((corpus / "camadas" / ib.CAMADA_ENTIDADES / "camada.json").read_text("utf-8"))
    assert ent["versao"] == "2" and ent["pessoas"] == 3 and ent["entidades"] == 4
    assert ent["assinatura_versao"] == ib.ASSINATURA_VERSAO
    assert ent["contagem_por_tipo"] == {"instituicao": 1, "pessoa": 2, "titulo": 1}
    assert len(ent["entradas"]) == 6 and all(len(h) == 64 for h in ent["entradas"].values())

    bio = json.loads((corpus / "camadas" / ib.CAMADA_BIOGRAFIAS / "camada.json").read_text("utf-8"))
    assert bio["contagem_por_nivel"] == {"ALTA": 1, "NAO_ENCONTRADO": 1, "REJEITADO": 1}
    assert bio["exibiveis"] == 1 and "D2" in bio["regra_exibicao"]

    manifesto = json.loads((corpus / "manifest.json").read_text("utf-8"))
    assert set(manifesto["camadas"][ib.CAMADA_ENTIDADES]["arquivos"]) == {
        f"camadas/{ib.CAMADA_ENTIDADES}/{n}" for n in
        ("entidades.parquet", "pessoas.parquet", "registro_pessoas.json", "camada.json")}
    assert manifesto["camadas"][ib.CAMADA_BIOGRAFIAS]["ativa"] is True


# ---------------------------------------------------------------------------
# Contra os dados reais do autor (pulado se a pasta do pipeline não estiver aqui)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (PIPELINE_REAL / "biografias_validadas.json").exists(),
                    reason="pasta do pipeline de biografias v2 ausente")
def test_contagens_por_nivel_batem_com_o_relatorio_do_p4():
    bios = json.loads((PIPELINE_REAL / "biografias_validadas.json").read_text(encoding="utf-8"))
    contagem = {}
    for b in bios:
        contagem[b["confianca_calculada"]] = contagem.get(b["confianca_calculada"], 0) + 1
    assert contagem == POR_NIVEL_P4
    assert sum(POR_NIVEL_P4.values()) == len(bios) == 1359


CAMADA_NO_REPO = ib.RAIZ / "corpus" / "camadas" / ib.CAMADA_BIOGRAFIAS


@pytest.mark.skipif(not (CAMADA_NO_REPO / "biografias.parquet").exists(),
                    reason="camada biografias_v2 ainda não construída")
def test_camada_construida_bate_com_o_relatorio_do_p4():
    b = pd.read_parquet(CAMADA_NO_REPO / "biografias.parquet")
    assert b["nivel"].value_counts().to_dict() == POR_NIVEL_P4
    assert int(b["exibivel"].sum()) == POR_NIVEL_P4["ALTA"] + POR_NIVEL_P4["MEDIA"]
    assert b["pessoa_uid"].is_unique
