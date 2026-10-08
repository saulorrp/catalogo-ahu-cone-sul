"""Testes do WP 1.5: artefatos do app (§3.4), validação do carregamento (A4) e coleção nova (A11)."""
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ahu import dados as mod_dados
from ahu.dados import ErroDados, carregar_sem_cache
from build import construir_app as ca
from build import construir_indice as ci
from build import importar_biografias_v2 as ib
from build import importar_camadas as ic
from build.importar_base import ErroImportacao, importar
from tests.test_construir_indice import Falso
from tests.test_importar_biografias_v2 import biografia, documento, escrever_revisao, pessoa

FIXTURE_MINI = ib.RAIZ / "tests" / "fixtures" / "teste_mini_catalogo.json"


# ---------------------------------------------------------------------------
# Um corpus completo de mentira: base, quatro camadas e embeddings
# ---------------------------------------------------------------------------

def montar_corpus(tmp_path, colecao="t"):
    """Base + tipologia + sociolinguística + entidades + biografias + embeddings (codificador falso)."""
    docs = [
        documento(1, "Conselho Ultramarino",
                  [{"nome": "Conselho Ultramarino", "tipo": "instituicao", "pessoa_id": None}]),
        documento(2, "D. Manoel de Mello",
                  [{"nome": "D. Manoel de Mello", "tipo": "pessoa", "pessoa_id": "P0001"}]),
        documento(3, "Conde de Basto e João Dias",
                  [{"nome": "Conde de Basto", "tipo": "titulo", "pessoa_id": "P0002"},
                   {"nome": "João Dias", "tipo": "pessoa", "pessoa_id": "P0003"}]),
    ]
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir(exist_ok=True)
    arquivos = {
        "ahu_sul_catalog_biografias.json": docs,
        "entidades_por_remetente.json": {
            "Conselho Ultramarino": [{"nome": "Conselho Ultramarino", "tipo": "instituicao",
                                      "titulos": [], "cargos": [], "notas": []}],
            "D. Manoel de Mello": [{"nome": "D. Manoel de Mello", "tipo": "pessoa",
                                    "titulos": ["barão de Laguna"], "cargos": [], "notas": []}],
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
        # P0001 exibível (ALTA); P0002 BAIXA e P0003 REJEITADO não podem chegar ao app (A1, D2).
        "biografias_validadas.json": [
            biografia("P0001", "ALTA", wikidata={"qid": "Q1", "nascimento": 1660, "morte": 1720,
                                                 "local": "Lisboa", "pais": "Portugal"}),
            biografia("P0002", "BAIXA", cidade_nascimento="Porto", pais_atual="Portugal"),
            biografia("P0003", "REJEITADO", motivo_rejeicao="morto antes do documento"),
        ],
    }
    for nome, conteudo in arquivos.items():
        (pipeline / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    escrever_revisao(pipeline, {"P0001": "", "P0002": "", "P0003": ""})

    corpus = tmp_path / "corpus"
    entrada = pipeline / "ahu_sul_catalog_biografias.json"
    importar(entrada, corpus, colecao, esperado=len(docs))
    ic.importar_camadas(entrada, corpus, colecao)
    ib.importar(pipeline, corpus, colecao)
    ci.construir_indice(corpus, None, "modelo-falso", Falso(), log=lambda *_: None)
    return corpus, pipeline


@pytest.fixture()
def corpus(tmp_path):
    return montar_corpus(tmp_path)[0]


@pytest.fixture()
def artefatos(tmp_path, corpus):
    saida = tmp_path / "app_dados"
    ca.construir(corpus, saida)
    return saida


# ---------------------------------------------------------------------------
# Os artefatos do §3.4
# ---------------------------------------------------------------------------

def test_constroi_os_sete_artefatos(artefatos):
    esperados = ("manifest.json", "documentos.parquet", "pessoas.parquet", "doc_pessoa.parquet",
                 "pdf_extra.parquet", "embeddings.f16.npy", "lexico.pkl")
    assert sorted(p.name for p in artefatos.iterdir()) == sorted(esperados)


def test_documentos_junta_base_tipologia_e_sociolinguistica(artefatos):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    assert len(docs) == 3 and docs["doc_uid"].is_unique
    for coluna in ("description", "reference_code", "tipologia", "typology_status",
                   "sender_category", "vernacular_score"):
        assert coluna in docs.columns, coluna


def test_d1_scribe_mediation_so_no_pdf_extra(artefatos):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    pdf = pd.read_parquet(artefatos / "pdf_extra.parquet")
    assert "scribe_mediation_likely" not in docs.columns
    assert list(pdf.columns) == ["doc_uid", "scribe_mediation_likely"]
    assert pdf["doc_uid"].tolist() == docs["doc_uid"].tolist()


def test_a1_biografia_nao_exibivel_nao_chega_ao_app(artefatos):
    """BAIXA e REJEITADO ficam sem dado nenhum: nem local, nem ano, nem fonte, nem nível."""
    pessoas = pd.read_parquet(artefatos / "pessoas.parquet")
    assert len(pessoas) == 3  # as três continuam na lista de pessoas
    assert int(pessoas["nascimento_exibivel"].sum()) == 1

    escondidas = pessoas[~pessoas["nascimento_exibivel"]]
    for coluna in ca.COLUNAS_NASCIMENTO:
        assert escondidas[coluna].isna().all(), coluna
    assert set(pessoas["nivel"].dropna()) == {"ALTA"}
    assert "Porto" not in set(pessoas["cidade_nascimento"].dropna())

    exibivel = pessoas[pessoas["nascimento_exibivel"]].iloc[0]
    assert exibivel["cidade_nascimento"] == "Lisboa" and exibivel["url_fonte"]


def test_rotulo_da_spec_6_3_chega_para_todas(artefatos):
    pessoas = pd.read_parquet(artefatos / "pessoas.parquet")
    assert pessoas["rotulo"].notna().all()
    assert set(pessoas["rotulo"]) == {"Nascimento identificado (confiança alta)",
                                      "Nascimento não confirmado", "Não identificado"}


def test_doc_pessoa_liga_documentos_e_pessoas(artefatos):
    doc_pessoa = pd.read_parquet(artefatos / "doc_pessoa.parquet")
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    pessoas = pd.read_parquet(artefatos / "pessoas.parquet")
    assert len(doc_pessoa) == 4
    assert doc_pessoa["doc_uid"].isin(docs["doc_uid"]).all()
    assert doc_pessoa["pessoa_uid"].dropna().isin(pessoas["pessoa_uid"]).all()
    assert doc_pessoa["posicao"].max() == 1


def test_embeddings_na_ordem_de_documentos(artefatos, corpus):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    emb = np.load(artefatos / "embeddings.f16.npy")
    camada = pd.read_parquet(corpus / "camadas" / "embeddings_modelo_falso_v1" / "indice.parquet")
    vetores = np.load(corpus / "camadas" / "embeddings_modelo_falso_v1" / "embeddings.f16.npy")

    assert emb.shape[0] == len(docs) and emb.dtype == np.float16
    posicao = {u: i for i, u in enumerate(camada["doc_uid"])}
    for i, uid in enumerate(docs["doc_uid"]):
        assert np.array_equal(emb[i], vetores[posicao[uid]])


def test_lexico_tem_os_campos_e_pesos_da_5_2(artefatos):
    import pickle

    with open(artefatos / "lexico.pkl", "rb") as f:
        lexico = pickle.load(f)
    assert lexico["campos"] == ["pessoas", "destinatario", "tipologia", "descricao", "pasta"]
    assert lexico["pesos"] == {"pessoas": 2.0, "destinatario": 1.5, "tipologia": 1.5,
                               "descricao": 1.0, "pasta": 0.5}
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    assert lexico["doc_uids"] == docs["doc_uid"].tolist()

    # O campo de pessoas traz nome da entidade, nome principal, variantes e títulos.
    pontos = lexico["indices"]["pessoas"].get_scores(["baran", "laguna"])
    assert pontos[1] > 0 and pontos[0] == 0


def test_manifesto_registra_camadas_usadas_e_hashes(artefatos):
    manifesto = json.loads((artefatos / "manifest.json").read_text("utf-8"))
    assert manifesto["corpus"]["camadas"]["tipologia"] == ic.CAMADA_TIPOLOGIA
    assert manifesto["corpus"]["camadas"]["biografias"] == ib.CAMADA_BIOGRAFIAS
    assert manifesto["contagens"]["documentos"] == 3
    assert manifesto["contagens"]["nascimento_exibivel"] == 1
    assert len(manifesto["artefatos"]) == 6
    for nome, hash_esperado in manifesto["artefatos"].items():
        obtido = hashlib.sha256((artefatos / nome).read_bytes()).hexdigest()
        assert obtido == hash_esperado, nome


def test_camada_ativa_ambigua_para_a_construcao(tmp_path, corpus):
    arq = corpus / "manifest.json"
    manifesto = json.loads(arq.read_text("utf-8"))
    manifesto["camadas"]["tipologia_v2_experimental"] = {"tipo": "tipologia", "versao": "2",
                                                         "ativa": True, "arquivos": {}}
    arq.write_text(json.dumps(manifesto, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ErroImportacao, match="camadas de tipo 'tipologia' ativas"):
        ca.construir(corpus, tmp_path / "saida")


def test_documento_sem_camada_para_a_construcao(tmp_path, corpus):
    arq = corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "tipologia.parquet"
    pd.read_parquet(arq).iloc[:1].to_parquet(arq, index=False)
    with pytest.raises(ErroImportacao, match="não estão na camada de tipologia"):
        ca.construir(corpus, tmp_path / "saida")


# ---------------------------------------------------------------------------
# A4: o app recusa iniciar com artefato desalinhado ou com hash diferente
# ---------------------------------------------------------------------------

def test_carrega_e_valida_os_artefatos(artefatos):
    d = carregar_sem_cache(artefatos)
    assert len(d.documentos) == 3
    assert d.embeddings.shape[0] == 3
    assert d.colecoes == ["t"]
    assert d.indice_de(d.documentos["doc_uid"].iloc[2]) == 2
    assert d.modelo_embeddings["nome"] == "modelo-falso"


def test_a4_hash_diferente_do_manifesto_recusa(artefatos):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    docs.loc[0, "description"] = "descrição trocada depois da construção"
    docs.to_parquet(artefatos / "documentos.parquet", index=False)
    with pytest.raises(ErroDados, match="mudou depois da construção"):
        carregar_sem_cache(artefatos)


def test_a4_numero_de_vetores_diferente_recusa(artefatos):
    emb = np.load(artefatos / "embeddings.f16.npy")
    np.save(artefatos / "embeddings.f16.npy", emb[:2])
    _reassinar(artefatos, "embeddings.f16.npy")
    with pytest.raises(ErroDados, match="2 vetores e 3 documentos"):
        carregar_sem_cache(artefatos)


def test_a4_ordem_de_uids_diferente_recusa(artefatos):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    docs.iloc[::-1].to_parquet(artefatos / "documentos.parquet", index=False)
    _reassinar(artefatos, "documentos.parquet")
    with pytest.raises(ErroDados, match="ordem dos uids"):
        carregar_sem_cache(artefatos)


def test_a4_artefato_ausente_recusa(artefatos):
    (artefatos / "pdf_extra.parquet").unlink()
    with pytest.raises(ErroDados, match="Artefato ausente"):
        carregar_sem_cache(artefatos)


def test_a4_manifesto_ausente_recusa(artefatos):
    (artefatos / "manifest.json").unlink()
    with pytest.raises(ErroDados, match="não existe"):
        carregar_sem_cache(artefatos)


def test_a4_vinculo_orfao_recusa(artefatos):
    doc_pessoa = pd.read_parquet(artefatos / "doc_pessoa.parquet")
    doc_pessoa.loc[0, "doc_uid"] = "t:inexistente"
    doc_pessoa.to_parquet(artefatos / "doc_pessoa.parquet", index=False)
    _reassinar(artefatos, "doc_pessoa.parquet")
    with pytest.raises(ErroDados, match="documentos que não existem"):
        carregar_sem_cache(artefatos)


def test_a4_d1_campo_do_pdf_em_documentos_recusa(artefatos):
    docs = pd.read_parquet(artefatos / "documentos.parquet")
    docs["scribe_mediation_likely"] = False
    docs.to_parquet(artefatos / "documentos.parquet", index=False)
    _reassinar(artefatos, "documentos.parquet")
    with pytest.raises(ErroDados, match="D1"):
        carregar_sem_cache(artefatos)


def test_a4_d2_nascimento_escondido_com_dado_recusa(artefatos):
    pessoas = pd.read_parquet(artefatos / "pessoas.parquet")
    escondida = (~pessoas["nascimento_exibivel"]).idxmax()
    pessoas.loc[escondida, "cidade_nascimento"] = "Porto"
    pessoas.to_parquet(artefatos / "pessoas.parquet", index=False)
    _reassinar(artefatos, "pessoas.parquet")
    with pytest.raises(ErroDados, match="D2"):
        carregar_sem_cache(artefatos)


def test_mensagem_de_erro_diz_o_que_rodar(artefatos):
    (artefatos / "lexico.pkl").unlink()
    with pytest.raises(ErroDados, match="construir_app.py"):
        carregar_sem_cache(artefatos)


def _reassinar(pasta, nome):
    """Atualiza só o hash de um artefato, para testar as verificações seguintes à do hash."""
    arq = pasta / "manifest.json"
    manifesto = json.loads(arq.read_text("utf-8"))
    manifesto["artefatos"][nome] = hashlib.sha256((pasta / nome).read_bytes()).hexdigest()
    arq.write_text(json.dumps(manifesto, ensure_ascii=False), encoding="utf-8")


def test_carregar_usa_cache_por_pasta(artefatos):
    mod_dados._carregar_cacheado.cache_clear()
    primeiro = mod_dados.carregar(artefatos)
    segundo = mod_dados.carregar(artefatos)
    assert primeiro is segundo  # as tabelas e os vetores não são recarregados (C8)


# ---------------------------------------------------------------------------
# A11: coleção nova entra sem mudar o código do app
# ---------------------------------------------------------------------------

def test_a11_colecao_teste_mini_entra_sem_mudar_o_codigo(tmp_path):
    corpus, _ = montar_corpus(tmp_path)
    antes = carregar_sem_cache(_construir(corpus, tmp_path / "antes"))
    assert antes.colecoes == ["t"]

    # Só isto: importar a coleção fictícia e reconstruir. Nenhuma linha de código muda.
    assert FIXTURE_MINI.exists(), FIXTURE_MINI
    importar(FIXTURE_MINI, corpus, "teste-mini", esperado=20)
    ic.importar_camadas(FIXTURE_MINI, corpus, "teste-mini")
    ci.construir_indice(corpus, None, "modelo-falso", Falso(), forcar=True, log=lambda *_: None)
    depois = carregar_sem_cache(_construir(corpus, tmp_path / "depois"))

    assert depois.colecoes == ["t", "teste-mini"]
    assert len(depois.documentos) == 23
    assert depois.embeddings.shape[0] == 23
    # As pastas e as tipologias também saem dos dados (D6).
    assert "Ilha Fictícia" in depois.pastas and "Costa Imaginária" in depois.pastas
    assert "BANDO" in depois.tipologias
    # A coleção antiga continua inteira, com as suas pessoas.
    assert int((depois.documentos["colecao"] == "t").sum()) == 3
    assert len(depois.pessoas) == 3
    assert depois.manifesto["contagens"]["colecoes"] == {"t": 3, "teste-mini": 20}


def test_a11_camadas_da_colecao_antiga_nao_sao_apagadas(tmp_path):
    corpus, _ = montar_corpus(tmp_path)
    importar(FIXTURE_MINI, corpus, "teste-mini", esperado=20)
    ic.importar_camadas(FIXTURE_MINI, corpus, "teste-mini")
    tipologia = pd.read_parquet(corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "tipologia.parquet")
    socio = pd.read_parquet(corpus / "camadas" / ic.CAMADA_SOCIO / "sociolinguistica.parquet")
    assert len(tipologia) == len(socio) == 23
    camada = json.loads(
        (corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "camada.json").read_text("utf-8"))
    assert camada["colecoes"] == ["t", "teste-mini"]


def test_pessoas_de_outra_colecao_nao_sao_sobrescritas_em_silencio(tmp_path):
    """Consolidar pessoas entre coleções é o WP 5.4; até lá a camada não é sobreposta calada."""
    corpus, _ = montar_corpus(tmp_path)
    importar(FIXTURE_MINI, corpus, "teste-mini", esperado=20)
    uids_mini = pd.read_parquet(corpus / "colecoes" / "teste-mini" / "documentos.parquet",
                                columns=["doc_uid"])["doc_uid"].tolist()
    with pytest.raises(ErroImportacao, match="WP 5.4"):
        ib.verificar_colecao_unica(corpus, "teste-mini", uids_mini)


def test_fixture_mini_tem_20_documentos_ficticios():
    docs = json.loads(FIXTURE_MINI.read_text(encoding="utf-8"))
    assert len(docs) == 20
    assert all(d["reference_code"].startswith("XX/TESTE/") for d in docs)
    assert all(r["pessoa_id"] is None for d in docs for r in d["remetentes"])


def _construir(corpus, saida):
    ca.construir(corpus, saida)
    return saida


# ---------------------------------------------------------------------------
# Os artefatos de verdade, se já estiverem construídos
# ---------------------------------------------------------------------------

APP_DADOS = ib.RAIZ / "app_dados"


@pytest.mark.skipif(not (APP_DADOS / "manifest.json").exists(),
                    reason="app_dados/ ainda não construído")
def test_artefatos_reais_carregam_e_respeitam_a1_e_d1():
    d = carregar_sem_cache(APP_DADOS)
    assert len(d.documentos) == d.embeddings.shape[0]
    assert "scribe_mediation_likely" not in d.documentos.columns
    assert set(d.pessoas["nivel"].dropna()) <= {"ALTA", "MEDIA"}
    assert int(d.pessoas["nascimento_exibivel"].sum()) == 253
    assert d.colecoes == ["ahu-sul"]


@pytest.mark.skipif(not (APP_DADOS / "manifest.json").exists(),
                    reason="app_dados/ ainda não construído")
def test_csv_de_revisao_real_continua_legivel():
    """Guarda-chuva do WP 1.3: o CSV que o autor edita à mão tem de continuar abrindo."""
    arq = ib.PIPELINE_PADRAO / "revisao_biografias.csv"
    if not arq.exists():
        pytest.skip("pipeline ausente")
    with open(arq, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    assert len(linhas) == 1359 and "decisao_manual" in linhas[0]
