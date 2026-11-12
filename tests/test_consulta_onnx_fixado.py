"""test_consulta_onnx_fixado.py — conferência e reprodutibilidade de modelos fixados."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from ahu import consulta_onnx


@pytest.fixture(autouse=True)
def limpar_conferidos():
    consulta_onnx._CONFERIDOS.clear()
    yield
    consulta_onnx._CONFERIDOS.clear()


@pytest.fixture
def ambiente_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    remoto = tmp_path / "remoto"
    cache = tmp_path / "cache"
    remoto.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    base_url = remoto.as_uri() + "/{repositorio}/resolve/{revisao}"
    monkeypatch.setenv("AHU_ONNX_BASE_URL", base_url)
    monkeypatch.setattr(consulta_onnx, "BASE", base_url)
    monkeypatch.setenv("AHU_ONNX_CACHE", str(cache))
    return remoto, cache


def test_download_hash_correto(ambiente_mock: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    remoto, _ = ambiente_mock
    repo = "teste/modelo-correto"
    rev = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
    conteudo = b"pesos onnx validos de teste"
    sha = hashlib.sha256(conteudo).hexdigest()

    monkeypatch.setitem(consulta_onnx.MODELOS_FIXADOS, repo, {
        "revisao": rev,
        "arquivos": {"model.onnx": sha},
    })

    arq_remoto = remoto / repo / "resolve" / rev / "model.onnx"
    arq_remoto.parent.mkdir(parents=True, exist_ok=True)
    arq_remoto.write_bytes(conteudo)

    caminho = consulta_onnx.baixar_arquivo(repo, "model.onnx")
    assert caminho.exists()
    assert caminho.read_bytes() == conteudo
    assert caminho in consulta_onnx._CONFERIDOS


def test_download_hash_errado(ambiente_mock: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    remoto, cache = ambiente_mock
    repo = "teste/modelo-errado"
    rev = "1111222233334444555566667777888899990000"
    conteudo_real = b"conteudo modificado sem autorizacao"
    sha_esperado = "0" * 64

    monkeypatch.setitem(consulta_onnx.MODELOS_FIXADOS, repo, {
        "revisao": rev,
        "arquivos": {"model.onnx": sha_esperado},
    })

    arq_remoto = remoto / repo / "resolve" / rev / "model.onnx"
    arq_remoto.parent.mkdir(parents=True, exist_ok=True)
    arq_remoto.write_bytes(conteudo_real)

    with pytest.raises(consulta_onnx.ErroDeModelo, match="Integridade violada"):
        consulta_onnx.baixar_arquivo(repo, "model.onnx")

    destino = cache / repo.replace("/", "__") / "model.onnx"
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    assert not destino.exists()
    assert not parcial.exists()


def test_cache_corrompido_rebaixado(ambiente_mock: tuple[Path, Path], monkeypatch) -> None:
    remoto, cache = ambiente_mock
    repo = "teste/modelo-cache-corrompido"
    rev = "abcdefabcdefabcdefabcdefabcdefabcdefabcd"
    conteudo_certo = b"pesos autenticos"
    sha = hashlib.sha256(conteudo_certo).hexdigest()

    monkeypatch.setitem(consulta_onnx.MODELOS_FIXADOS, repo, {
        "revisao": rev,
        "arquivos": {"model.onnx": sha},
    })

    arq_remoto = remoto / repo / "resolve" / rev / "model.onnx"
    arq_remoto.parent.mkdir(parents=True, exist_ok=True)
    arq_remoto.write_bytes(conteudo_certo)

    # Simula arquivo antigo ou corrompido no cache
    destino = cache / repo.replace("/", "__") / "model.onnx"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(b"lixo deixado no cache")

    logs: list[str] = []
    caminho = consulta_onnx.baixar_arquivo(repo, "model.onnx", log=logs.append)
    assert caminho == destino
    assert caminho.read_bytes() == conteudo_certo
    assert caminho in consulta_onnx._CONFERIDOS
    assert any("divergente" in msg for msg in logs)


def test_repositorio_nao_fixado_usa_main(ambiente_mock: tuple[Path, Path]) -> None:
    remoto, _ = ambiente_mock
    repo = "desconhecido/modelo-livre"
    conteudo = b"modelo fora da tabela fixa"

    arq_remoto = remoto / repo / "resolve" / "main" / "outro.onnx"
    arq_remoto.parent.mkdir(parents=True, exist_ok=True)
    arq_remoto.write_bytes(conteudo)

    logs: list[str] = []
    caminho = consulta_onnx.baixar_arquivo(repo, "outro.onnx", log=logs.append)
    assert caminho.exists()
    assert caminho.read_bytes() == conteudo
    assert any("aviso" in msg for msg in logs)


def test_hashes_e_revisoes_formato() -> None:
    padrao_sha256 = re.compile(r"^[0-9a-f]{64}$")
    padrao_revisao = re.compile(r"^[0-9a-f]{40}$")

    assert consulta_onnx.MODELOS_FIXADOS, "MODELOS_FIXADOS não pode estar vazio"
    for repo, info in consulta_onnx.MODELOS_FIXADOS.items():
        rev = info["revisao"]
        assert padrao_revisao.match(rev), f"Revisão inválida para {repo}: {rev}"
        arquivos = info["arquivos"]
        assert arquivos, f"Repositório {repo} não tem arquivos registrados"
        for arq, sha in arquivos.items():
            assert padrao_sha256.match(sha), f"SHA-256 inválido para {arq} em {repo}: {sha}"


def test_manifesto_modelos_inclui_espelho_e5_small() -> None:
    manifesto = consulta_onnx.manifesto_modelos()
    assert manifesto["fonte"] == consulta_onnx.BASE_PADRAO
    assert manifesto["arquivo_consulta"] == consulta_onnx.ARQUIVO_MODELO
    espelho = consulta_onnx.repositorio_de("intfloat/multilingual-e5-small")
    assert espelho == "Xenova/multilingual-e5-small"
    assert espelho in manifesto["repositorios"]
    info = manifesto["repositorios"][espelho]
    assert info["revisao"] == "761b726dd34fb83930e26aab4e9ac3899aa1fa78"
    assert "onnx/model.onnx" in info["arquivos"]
    assert "onnx/model_quantized.onnx" in info["arquivos"]
    assert "tokenizer.json" in info["arquivos"]
