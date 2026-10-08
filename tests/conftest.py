"""conftest.py — o corpus de mentira que os testes de busca, consulta e filtros compartilham.

Dez documentos inventados, com os casos que a spec §5 pede: grafia antiga ('Melo'/'Mello'), as
três formas de cota, frase para a busca entre aspas e uma pessoa com título. O índice semântico é
feito com o codificador falso de `test_construir_indice.py`, para não baixar o e5 nos testes.
"""
import json

import pytest

from ahu.dados import carregar_sem_cache
from build import construir_app as ca
from build import construir_indice as ci
from build import importar_biografias_v2 as ib
from build import importar_camadas as ic
from build.importar_base import importar
from tests.test_construir_indice import Falso
from tests.test_importar_biografias_v2 import biografia, documento, escrever_revisao, pessoa

APP_DADOS = ib.RAIZ / "app_dados"
tem_artefatos = (APP_DADOS / "manifest.json").exists()


def doc_busca(n, descricao, sender="Junta de Teste", tipologia="CARTA", recipient="Rei",
              reference=None, new_code=None):
    d = documento(n, sender, [{"nome": sender, "tipo": "instituicao", "pessoa_id": None}])
    d["description"] = descricao
    d["extracted_typology"] = tipologia
    d["recipient_name"] = recipient
    if reference:
        d["reference_code"] = reference
    if new_code:
        d["new_code"] = new_code
    return d


@pytest.fixture(scope="module")
def corpus_busca(tmp_path_factory):
    """Dez documentos com casos de grafia, cota e frase, mais uma pessoa com variantes."""
    tmp_path = tmp_path_factory.mktemp("busca")
    docs = [
        doc_busca(1, "CARTA sobre a deserção de soldados da guarnição do presídio."),
        doc_busca(2, "OFÍCIO sobre desertores e a falta de mantimentos na fronteira.",
                  tipologia="OFÍCIO"),
        doc_busca(3, "REQUERIMENTO de Manuel de Melo, pedindo soldo atrasado.",
                  tipologia="REQUERIMENTO"),
        doc_busca(4, "CARTA de José Nunes sobre gado e cavalos da estância."),
        doc_busca(5, "CARTA sobre obras do porto e a ilha de Santa Catarina.",
                  reference="PT/AHU/CU/021/0006/00390",
                  new_code="AHU_ACL_CU_021, Cx. 6, D. 390."),
        doc_busca(6, "MAPA dos habitantes da vila, com soldados e moradores.", tipologia="MAPA"),
        doc_busca(7, "CARTA sobre a fuga de presos e a deserção da tropa paga."),
        doc_busca(8, "PARECER sobre o pedido de Manoel de Mello e Castro.", tipologia="PARECER"),
        doc_busca(9, "CONSULTA sobre a demarcação de limites no rio Uruguai.",
                  tipologia="CONSULTA"),
        doc_busca(10, "CARTA do governador ao vice-rei marquês do Lavradio sobre tropas.",
                  recipient="marquês do Lavradio"),
    ]
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    arquivos = {
        "ahu_sul_catalog_biografias.json": docs,
        "entidades_por_remetente.json": {
            "Junta de Teste": [{"nome": "Junta de Teste", "tipo": "instituicao", "titulos": [],
                                "cargos": [], "notas": []}]},
        "doc_pessoas.json": [[None]] * len(docs),
        "pessoas.json": [pessoa("P0001", "Luís de Almeida", ["marquês do Lavradio"])],
        "biografias_validadas.json": [biografia("P0001", "ALTA")],
    }
    for nome, conteudo in arquivos.items():
        (pipeline / nome).write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    escrever_revisao(pipeline, {"P0001": ""})

    corpus = tmp_path / "corpus"
    entrada = pipeline / "ahu_sul_catalog_biografias.json"
    importar(entrada, corpus, "t", esperado=len(docs))
    ic.importar_camadas(entrada, corpus, "t")
    ib.importar(pipeline, corpus, "t")
    ci.construir_indice(corpus, None, "modelo-falso", Falso(), log=lambda *_: None)
    saida = tmp_path / "app_dados"
    ca.construir(corpus, saida)
    return carregar_sem_cache(saida)


@pytest.fixture(scope="module")
def encoder():
    return Falso()
