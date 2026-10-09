"""proveniencia.py — o bloco de proveniência que acompanha o que sai do app (spec §8.2).

Sem Streamlit. Era parte de `ahu/exportar.py`, que a **D7** retirou junto com as exportações em
CSV e JSON e o dossiê. O que sobrou é o que a **D11** ainda precisa: o plano de coleta do EINS
(`eins-plano-coleta/1`, §7.4) tem de dizer de que corpus e de que consulta ele saiu, senão o
Processo criado no Acervo não é rastreável.

O bloco traz a `Consulta` completa, a data, a versão do app (e o commit, quando o git está por
perto), as camadas ativas com as suas versões lidas do manifesto, o SHA-256 do manifesto e o DOI.
"""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

CITACAO = ("Rocha, S. R. P. Classificador AHU-Sul v3 (software e dados), 2026.")
DOI_PREPRINT = "10.1590/SciELOPreprints.16461"
DOI_ZENODO = None       # entra na versão 3.0.0 (WP 4.4)
CREDITO = ('Resumos e cotas: Arquivo Histórico Ultramarino e Projeto Resgate '
           '"Barão do Rio Branco".')


def versao_do_app() -> dict:
    """Versão declarada no `pyproject.toml` e o commit, quando dá para saber.

    O commit sai do git, que existe no Streamlit Community Cloud (o app é um clone) mas pode não
    existir num zip baixado. Quando não dá, o campo vem `None` em vez de uma invenção.
    """
    versao = None
    arquivo = RAIZ / "pyproject.toml"
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith("version"):
                versao = linha.split("=", 1)[1].strip().strip('"').strip("'")
                break
    commit = None
    try:
        bruto = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                               capture_output=True, text=True, timeout=5, check=False)
        if bruto.returncode == 0:
            commit = bruto.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"versao": versao, "commit": commit}


def _sha256_do_manifesto(dados) -> str | None:
    arquivo = Path(dados.pasta) / "manifest.json"
    if not arquivo.exists():
        return None
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()


def camadas_ativas(dados) -> dict:
    """Papel -> {camada, versão}, lido do manifesto. Nunca escrito no código (D6)."""
    corpus = dados.manifesto.get("corpus", {})
    declaradas = corpus.get("manifesto", {}).get("camadas", {})
    saida = {}
    for papel, nome in (corpus.get("camadas") or {}).items():
        entrada = declaradas.get(nome, {})
        saida[papel] = {"camada": nome, "versao": entrada.get("versao"),
                        "documentos": entrada.get("documentos")}
        if entrada.get("modelo"):
            saida[papel]["modelo"] = entrada["modelo"]
    return saida


def proveniencia(dados, consulta=None, total: int | None = None, origem: str | None = None) -> dict:
    """O bloco que acompanha **todo** arquivo exportado (§8.2)."""
    return {
        "app": "classificador-ahu-sul",
        "app_versao": versao_do_app(),
        "gerado_em": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "origem": origem,
        "documentos": total,
        "consulta": consulta.para_dicionario() if consulta is not None else None,
        "corpus": {
            "manifesto_sha256": _sha256_do_manifesto(dados),
            "versao_artefatos": dados.manifesto.get("versao_artefatos"),
            "camadas_ativas": camadas_ativas(dados),
            "modelo_embeddings": dados.modelo_embeddings,
        },
        "citacao": CITACAO,
        "doi": {"preprint": DOI_PREPRINT, "zenodo": DOI_ZENODO},
        "credito": CREDITO,
    }
