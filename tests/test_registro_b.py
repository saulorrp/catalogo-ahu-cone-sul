"""Testes unitários para o gerador do Registro B do Zenodo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_SCRIPT = RAIZ / "zenodo" / "registro_b" / "scripts" / "exportar_registro_b.py"

# Carregamento dinâmico do script por caminho sem poluir sys.path
spec = importlib.util.spec_from_file_location("exportar_registro_b", CAMINHO_SCRIPT)
assert spec is not None and spec.loader is not None
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)

montar_documentos_sem_resumo = modulo.montar_documentos_sem_resumo
dicionario = modulo.dicionario
sha256sums = modulo.sha256sums


def test_colunas_terceiros_removidas_e_sha1_mantido():
    """Confere que description e description_resgate saem e sha1_resumo fica."""
    df_sintetico = pd.DataFrame(
        {
            "doc_uid": ["doc_001", "doc_002"],
            "colecao": ["ahu-brasil", "ahu-sul"],
            "description": ["Resumo do catálogo histórico", "Outro resumo de documento"],
            "description_resgate": ["Texto transcrito pelo Resgate", None],
            "sha1_resumo": [
                "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12",
                "de9f2c7fd25e1b3af3415df40a6ad7e85be4f2ac",
            ],
        }
    )

    df_resultado = montar_documentos_sem_resumo(df_sintetico)

    assert "description" not in df_resultado.columns
    assert "description_resgate" not in df_resultado.columns
    assert "sha1_resumo" in df_resultado.columns
    assert "doc_uid" in df_resultado.columns
    assert "colecao" in df_resultado.columns
    assert len(df_resultado) == 2

    # Valida que description sem sha1_resumo correspondente falha com erro explícito
    df_invalido = pd.DataFrame(
        {
            "doc_uid": ["doc_003"],
            "description": ["Texto presente sem hash"],
            "sha1_resumo": [None],
        }
    )
    with pytest.raises(ValueError, match="sha1_resumo"):
        montar_documentos_sem_resumo(df_invalido)


def test_dicionario_falha_coluna_sem_descricao():
    """Verifica que qualquer coluna não descrita em DESCRICOES interrompe a execução."""
    df_valido = pd.DataFrame(
        {
            "doc_uid": ["doc_001"],
            "colecao": ["ahu-brasil"],
        }
    )
    meta = dicionario(df_valido)
    assert len(meta) == 2
    assert meta[0]["nome"] == "doc_uid"
    assert meta[1]["nome"] == "colecao"

    df_com_coluna_estranha = pd.DataFrame(
        {
            "doc_uid": ["doc_001"],
            "coluna_nao_documentada_xyz": [123],
        }
    )
    with pytest.raises(KeyError, match="coluna_nao_documentada_xyz"):
        dicionario(df_com_coluna_estranha)


def test_formato_sha256sums(tmp_path: Path):
    """Valida a formatação de SHA256SUMS com caminhos relativos e dois espaços."""
    pasta = tmp_path / "saida"
    pasta.mkdir()
    sub = pasta / "subpasta"
    sub.mkdir()

    (pasta / "b_primeiro.txt").write_text("conteudo b\n", encoding="utf-8")
    (sub / "a_segundo.txt").write_text("conteudo a\n", encoding="utf-8")

    resultado_texto = sha256sums(pasta)
    arquivo_sums = pasta / "SHA256SUMS"

    assert arquivo_sums.exists()
    assert arquivo_sums.read_text(encoding="utf-8") == resultado_texto

    linhas = [linha for linha in resultado_texto.splitlines() if linha.strip()]
    assert len(linhas) == 2

    # Garante a ordem alfabética posix e a presença de exatamente dois espaços
    assert linhas[0].endswith("  b_primeiro.txt")
    assert linhas[1].endswith("  subpasta/a_segundo.txt")

    for linha in linhas:
        partes = linha.split("  ", 1)
        assert len(partes) == 2
        digest, caminho_rel = partes
        assert len(digest) == 64
        assert int(digest, 16) >= 0

    # Confere que o próprio SHA256SUMS não é indexado
    assert "SHA256SUMS" not in resultado_texto
