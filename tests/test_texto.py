"""Testes de `ahu/texto.py`: as chaves de nome que sustentam o `pessoa_uid` (WP 1.3; WP 2.1 amplia)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ahu import texto

PIPELINE = Path(__file__).resolve().parent.parent.parent / "biografias_v2"


def test_dobrar_tira_acento_caixa_e_pontuacao():
    assert texto.dobrar("D. José de Mello-Castro") == "d jose de mello castro"
    assert texto.dobrar("") == ""
    assert texto.dobrar(None) == ""


def test_chave_nome_junta_variantes_do_mesmo_nome():
    iguais = ["D. Manoel de Mello e Castro", "Manuel de Melo Castro", "manoel de mello castro"]
    assert len({texto.chave_nome(n) for n in iguais}) == 1


def test_chave_nome_separa_nomes_diferentes():
    assert texto.chave_nome("João Dias") != texto.chave_nome("João Dias Pereira")


def test_chave_titulo_tira_ordinal_e_particula():
    assert texto.chave_titulo("9º conde da Feira") == "conde feira"
    assert texto.chave_titulo("marquês do Lavradio") == "marques lavradio"


@pytest.mark.skipif(not (PIPELINE / "comum.py").exists(), reason="pipeline de biografias ausente")
def test_mesma_saida_do_comum_py_do_pipeline():
    """Enquanto houver duas cópias (o WP 2.1 faz o pipeline importar daqui), elas têm de bater.

    É por estas funções que o `pessoa_uid` é calculado: divergir aqui troca identificadores públicos.
    """
    spec = importlib.util.spec_from_file_location("comum_pipeline", PIPELINE / "comum.py")
    comum = importlib.util.module_from_spec(spec)
    sys.modules["comum_pipeline"] = comum
    spec.loader.exec_module(comum)

    nomes, titulos = [], []
    arq_pessoas = PIPELINE / "pessoas.json"
    if arq_pessoas.exists():
        for p in json.loads(arq_pessoas.read_text(encoding="utf-8")):
            nomes += [p["nome_principal"] or ""] + list(p["variantes_nome"])
            titulos += list(p["titulos"])
    nomes += ["D. Manoel de Mello e Castro", "Manuel de Melo Castro", "Luiz Antonio de Souza"]
    titulos += ["9º conde da Feira", "marquês do Lavradio"]

    assert all(comum.chave_nome(n) == texto.chave_nome(n) for n in nomes)
    assert all(comum.chave_titulo(t) == texto.chave_titulo(t) for t in titulos)
    assert all(comum.dobrar(n) == texto.dobrar(n) for n in nomes)
