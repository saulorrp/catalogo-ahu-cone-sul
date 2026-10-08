"""Testes do WP 1.2: camadas tipologia_v1 e sociolinguistica_deepseek_v3."""
import json

import pandas as pd
import pytest

from build import importar_camadas as ic
from build.importar_base import ErroImportacao, importar


def doc(n, **extra):
    base = {
        "folder": "Pasta", "description": f"Descrição do documento {n}",
        "old_code": None, "new_code": f"AHU_ACL_CU_001, Cx. 1, D. {n}.",
        "reference_code": f"PT/AHU/CU/001/0001/{n:05d}",
        "document_id_and_date": f"{n}. 1700, Janeiro, 1, Lisboa", "ano_isolado": 1700,
        "extracted_typology": "OFÍCIO", "typology_status": True,
        "sender_name": f"Remetente {n}", "recipient_name": "Rei",
        "sender_category": "Local Elite", "vector": "Bottom-Up",
        "typology_impact": "Highly Formulaic", "scribe_mediation_likely": False,
        "vernacular_score": 2,
        "sociolinguistic_reasoning_by_deepseek_v3": "Discurso administrativo formular.",
    }
    base.update(extra)
    return base


def escrever(tmp_path, docs):
    entrada = tmp_path / "entrada.json"
    entrada.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    return entrada


@pytest.fixture()
def corpus_com_base(tmp_path):
    """Base do WP 1.1 pronta, com casos de tipologia e de SRSP variados."""
    docs = [
        doc(1),
        doc(2, extracted_typology=" carta  ", typology_status=True),
        doc(3, extracted_typology="COSIJLTA", typology_status=False, vernacular_score=6),
        doc(4, extracted_typology="UNKNOWN", typology_status=False, scribe_mediation_likely=True),
        doc(5, extracted_typology="AUTO DE EXAME.", typology_status=False, vernacular_score=9),
    ]
    entrada = escrever(tmp_path, docs)
    importar(entrada, tmp_path / "corpus", "t", esperado=len(docs))
    return entrada, tmp_path / "corpus", docs


# ---------------------------------------------------------------------------
# Normalização da tipologia
# ---------------------------------------------------------------------------

def test_normalizar_tipologia_so_mexe_na_forma():
    assert ic.normalizar_tipologia(" carta  régia ") == "CARTA RÉGIA"
    assert ic.normalizar_tipologia("AUTO DE EXAME.") == "AUTO DE EXAME"
    assert ic.normalizar_tipologia('"OFÍCIO"') == "OFÍCIO"
    assert ic.normalizar_tipologia(None) == ""


def test_chave_tipologia_ignora_acento_e_caixa():
    assert ic.chave_tipologia("Ofício") == ic.chave_tipologia("OFICIO") == "OFICIO"


def test_correcao_casa_com_acento_para_poder_corrigir_acento_trocado(tmp_path):
    """'OFICÍO' é erro de extração; 'OFÍCIO' é a forma certa. A tabela precisa distinguir as duas."""
    correcoes = ic.carregar_correcoes(tmp_path / "correcoes_tipologia.json")
    assert correcoes["OFICÍO"] == "OFÍCIO"
    assert "OFÍCIO" not in correcoes


def test_tabela_de_correcoes_e_criada_e_editavel(tmp_path):
    arq = tmp_path / "correcoes_tipologia.json"
    correcoes = ic.carregar_correcoes(arq)
    assert arq.exists()
    assert correcoes["COSIJLTA"] == "CONSULTA"
    assert "_LEIA-ME" not in correcoes

    arq.write_text(json.dumps({"_LEIA-ME": "x", "PRIMEIRA": "VIA"}), encoding="utf-8")
    assert ic.carregar_correcoes(arq) == {"PRIMEIRA": "VIA"}


def test_correcao_mal_formada_para_a_importacao(tmp_path):
    arq = tmp_path / "correcoes_tipologia.json"
    arq.write_text(json.dumps({"PRIMEIRA": ["VIA"]}), encoding="utf-8")
    with pytest.raises(ErroImportacao, match="string"):
        ic.carregar_correcoes(arq)


# ---------------------------------------------------------------------------
# Camada de tipologia
# ---------------------------------------------------------------------------

def test_camada_tipologia_uma_linha_por_doc_uid_na_ordem_da_base(corpus_com_base):
    entrada, corpus, docs = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")

    base = pd.read_parquet(corpus / "colecoes" / "t" / "documentos.parquet")
    df = pd.read_parquet(corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "tipologia.parquet")
    assert len(df) == len(docs)
    assert df["doc_uid"].tolist() == base["doc_uid"].tolist()
    assert df["doc_uid"].is_unique


def test_camada_tipologia_normaliza_corrige_e_marca_desconhecida(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")
    df = pd.read_parquet(corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "tipologia.parquet")

    assert df.loc[1, "tipologia"] == "CARTA"
    assert df.loc[1, "tipologia_bruta"] == " carta  "

    # 'COSIJLTA' é erro de extração: a tabela de correções o conserta e a linha fica marcada.
    assert df.loc[2, "tipologia"] == "CONSULTA"
    assert bool(df.loc[2, "tipologia_corrigida"])

    # 'UNKNOWN' vira tipologia nula e nunca é canônica.
    assert pd.isna(df.loc[3, "tipologia"])
    assert not bool(df.loc[3, "typology_status"])

    # Tipologia fora do vocabulário canônico fica na camada, só com typology_status falso.
    assert df.loc[4, "tipologia"] == "AUTO DE EXAME"
    assert not bool(df.loc[4, "typology_status"])


def test_tipologia_corrigida_herda_o_status_canonico_dos_dados(tmp_path):
    """'OFICÍO' corrigido para 'OFÍCIO' deixa de ser 'a revisar', porque 'OFÍCIO' é canônica nos dados."""
    docs = [
        doc(1, extracted_typology="OFÍCIO", typology_status=True),
        doc(2, extracted_typology="OFICÍO", typology_status=False),
        doc(3, extracted_typology="AUTO DE EXAME", typology_status=False),
    ]
    entrada = escrever(tmp_path, docs)
    importar(entrada, tmp_path / "corpus", "t", esperado=len(docs))
    ic.importar_camadas(entrada, tmp_path / "corpus", "t", so="tipologia")
    df = pd.read_parquet(tmp_path / "corpus" / "camadas" / ic.CAMADA_TIPOLOGIA / "tipologia.parquet")

    assert df["tipologia"].tolist() == ["OFÍCIO", "OFÍCIO", "AUTO DE EXAME"]
    assert df["typology_status"].tolist() == [True, True, False]
    assert df["tipologia_corrigida"].tolist() == [False, True, False]


def test_typology_status_invalido_para_a_importacao(tmp_path):
    entrada = escrever(tmp_path, [doc(1, typology_status="sim")])
    importar(entrada, tmp_path / "corpus", "t", esperado=1)
    with pytest.raises(ErroImportacao, match="typology_status"):
        ic.importar_camadas(entrada, tmp_path / "corpus", "t")


# ---------------------------------------------------------------------------
# Camada sociolinguística
# ---------------------------------------------------------------------------

def test_camada_socio_tem_os_campos_do_preprint(corpus_com_base):
    entrada, corpus, docs = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")
    df = pd.read_parquet(corpus / "camadas" / ic.CAMADA_SOCIO / "sociolinguistica.parquet")

    assert list(df.columns) == ["doc_uid", "sender_category", "vector", "typology_impact",
                                "vernacular_score", "scribe_mediation_likely", "justificativa"]
    assert len(df) == len(docs)
    assert df["justificativa"].iloc[0] == "Discurso administrativo formular."
    assert bool(df["scribe_mediation_likely"].iloc[3])


def test_srsp_fica_inteiro_de_1_a_9(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")
    df = pd.read_parquet(corpus / "camadas" / ic.CAMADA_SOCIO / "sociolinguistica.parquet")
    assert str(df["vernacular_score"].dtype) == "Int64"
    assert df["vernacular_score"].tolist() == [2, 2, 6, 2, 9]


@pytest.mark.parametrize("valor", [0.2, 0, 10, None, "alto"])
def test_srsp_dividido_ou_fora_da_escala_para_a_importacao(tmp_path, valor):
    entrada = escrever(tmp_path, [doc(1, vernacular_score=valor)])
    importar(entrada, tmp_path / "corpus", "t", esperado=1)
    with pytest.raises(ErroImportacao, match="vernacular_score"):
        ic.importar_camadas(entrada, tmp_path / "corpus", "t")


def test_campo_do_deepseek_faltando_para_a_importacao(tmp_path):
    d = doc(1)
    del d["vector"]
    entrada = escrever(tmp_path, [d])
    importar(entrada, tmp_path / "corpus", "t", esperado=1)
    with pytest.raises(ErroImportacao, match="vector"):
        ic.importar_camadas(entrada, tmp_path / "corpus", "t")


# ---------------------------------------------------------------------------
# Proveniência: camada.json e manifesto
# ---------------------------------------------------------------------------

def test_cada_camada_tem_camada_json_com_proveniencia(corpus_com_base):
    entrada, corpus, docs = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")

    tip = json.loads((corpus / "camadas" / ic.CAMADA_TIPOLOGIA / "camada.json").read_text("utf-8"))
    assert tip["nome"] == ic.CAMADA_TIPOLOGIA and tip["versao"] == "1"
    assert tip["documentos"] == len(docs)
    assert tip["processo"]["script"] == "build/importar_camadas.py"
    assert tip["entrada"]["arquivo"] == entrada.name and len(tip["entrada"]["sha256"]) == 64
    assert tip["correcoes"]["aplicadas"] == 2  # COSIJLTA e UNKNOWN
    assert tip["contagem_por_tipologia"]["CARTA"] == 1

    soc = json.loads((corpus / "camadas" / ic.CAMADA_SOCIO / "camada.json").read_text("utf-8"))
    assert soc["modelo"] == ic.MODELO_SOCIO and soc["versao"] == "3"
    assert soc["auditoria"]["mae"] == 1.13 and soc["auditoria"]["n"] == 705
    assert soc["contagem_por_srsp"] == {"2": 3, "6": 1, "9": 1}
    assert "PDF" in soc["uso_restrito"]["scribe_mediation_likely"]


def test_prompt_do_preprint_e_guardado_e_nao_se_perde(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    arq = corpus / "camadas" / ic.CAMADA_SOCIO / "camada.json"

    r = ic.importar_camadas(entrada, corpus, "t")
    assert r["camadas"][ic.CAMADA_SOCIO]["prompt_pendente"]
    assert json.loads(arq.read_text("utf-8"))["prompt"] == ic.PROMPT_PENDENTE

    ic.importar_camadas(entrada, corpus, "t", prompt="Classify the sender of this document...")
    assert json.loads(arq.read_text("utf-8"))["prompt"].startswith("Classify the sender")

    # Uma reimportação sem --prompt mantém o prompt já gravado.
    r = ic.importar_camadas(entrada, corpus, "t")
    assert not r["camadas"][ic.CAMADA_SOCIO]["prompt_pendente"]
    assert json.loads(arq.read_text("utf-8"))["prompt"].startswith("Classify the sender")


def test_manifesto_registra_as_duas_camadas_com_hash(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")
    manifesto = json.loads((corpus / "manifest.json").read_text("utf-8"))

    for nome, arquivo in ((ic.CAMADA_TIPOLOGIA, "tipologia.parquet"),
                          (ic.CAMADA_SOCIO, "sociolinguistica.parquet")):
        entrada_manifesto = manifesto["camadas"][nome]
        assert entrada_manifesto["ativa"] is True
        hashes = entrada_manifesto["arquivos"]
        assert set(hashes) == {f"camadas/{nome}/{arquivo}", f"camadas/{nome}/camada.json"}
        assert all(len(h) == 64 for h in hashes.values())
    assert "t" in manifesto["colecoes"]  # a coleção do WP 1.1 continua registrada


def test_so_constroi_a_camada_pedida(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t", so="tipologia")
    assert (corpus / "camadas" / ic.CAMADA_TIPOLOGIA).is_dir()
    assert not (corpus / "camadas" / ic.CAMADA_SOCIO).exists()


# ---------------------------------------------------------------------------
# Ligação com a base
# ---------------------------------------------------------------------------

def test_sem_a_base_a_mensagem_manda_rodar_o_wp_1_1(tmp_path):
    entrada = escrever(tmp_path, [doc(1)])
    with pytest.raises(ErroImportacao, match="importar_base"):
        ic.importar_camadas(entrada, tmp_path / "corpus", "t")


def test_entrada_diferente_da_base_para_a_importacao(corpus_com_base):
    entrada, corpus, docs = corpus_com_base
    outra = escrever(corpus.parent, docs + [doc(99)])
    with pytest.raises(ErroImportacao, match="importar_base"):
        ic.importar_camadas(outra, corpus, "t")


def test_reimportacao_da_as_mesmas_linhas(corpus_com_base):
    entrada, corpus, _ = corpus_com_base
    ic.importar_camadas(entrada, corpus, "t")
    primeira = pd.read_parquet(corpus / "camadas" / ic.CAMADA_SOCIO / "sociolinguistica.parquet")
    ic.importar_camadas(entrada, corpus, "t")
    segunda = pd.read_parquet(corpus / "camadas" / ic.CAMADA_SOCIO / "sociolinguistica.parquet")
    pd.testing.assert_frame_equal(primeira, segunda)
