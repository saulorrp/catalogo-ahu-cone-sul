"""exportar.py — WP 3.2: exportação em CSV e JSON, com proveniência (spec §8.2).

Sem Streamlit: as funções daqui devolvem **bytes**, e o `app.py` só os entrega ao
`st.download_button`. É o que permite ao teste reabrir o CSV e comparar linha por linha.

Duas regras governam este módulo:

* **D1 — `scribe_mediation_likely` não sai daqui.** Ele só existe no dossiê em PDF. As colunas
  exportadas são uma lista fechada (`COLUNAS`), e há um teste que varre o CSV e o JSON procurando
  o campo proibido. É de propósito que a lista seja fechada em vez de "tudo menos X": uma coluna
  nova em `documentos.parquet` não vaza para a exportação por acidente;
* **proveniência em todo arquivo (§8.2).** A `Consulta` completa, a data, a versão do app, as
  camadas ativas com as suas versões, o SHA-256 do manifesto do corpus e o DOI. Sem isso o CSV é
  um punhado de linhas sem origem, e o pesquisador não consegue refazer a busca que o gerou.

**O CSV é feito para abrir no Excel em português:** UTF-8 **com BOM** (senão o Excel lê os acentos
como lixo) e `;` como separador (o Excel em configuração pt-BR/pt-PT espera ponto-e-vírgula). A
proveniência vai em linhas de comentário `#` **antes** do cabeçalho, e `ler_csv` tira só as linhas
de comentário do início — um `#` dentro de um resumo não atrapalha.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ahu import links
from ahu.ficha import CAMPOS_PROIBIDOS

RAIZ = Path(__file__).resolve().parent.parent

SEPARADOR_CSV = ";"
BOM = "﻿"
COMENTARIO = "#"

CITACAO = ("Rocha, S. R. P. Classificador AHU-Sul v3 (software e dados), 2026.")
DOI_PREPRINT = "10.1590/SciELOPreprints.16461"
DOI_ZENODO = None       # entra na versão 3.0.0 (WP 4.4)
CREDITO = ('Resumos e cotas: Arquivo Histórico Ultramarino e Projeto Resgate '
           '"Barão do Rio Branco".')

# As colunas do §8.2, em ordem. `campo` é a coluna de `documentos.parquet`; None = calculada.
COLUNAS: tuple[tuple[str, str | None], ...] = (
    ("doc_uid", "doc_uid"),
    ("colecao", "colecao"),
    ("reference_code", "reference_code"),
    ("new_code", "new_code"),
    ("old_code", "old_code"),
    ("item_e_data", "document_id_and_date"),
    ("ano", "ano_isolado"),
    ("pasta", "folder"),
    ("tipologia", "tipologia"),
    ("tipologia_a_revisar", None),
    ("pessoas", None),
    ("pessoas_uid", None),
    ("destinatario", "recipient_name"),
    ("srsp", "vernacular_score"),
    ("categoria_social", "sender_category"),
    ("vetor", "vector"),
    ("impacto_tipologia", "typology_impact"),
    ("origem_remetente", None),
    ("origem_nivel", None),
    ("resumo_ahu", "description"),
    ("link_digitarq", None),
    ("link_resgate", None),
)
# Só no JSON (§8.2: "o mesmo + a justificativa do DeepSeek").
COLUNAS_JSON = (*COLUNAS, ("justificativa_ia", "justificativa"))


# ---------------------------------------------------------------------------
# Proveniência (§8.2)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# As linhas
# ---------------------------------------------------------------------------

def _valor(linha, campo):
    if campo is None or campo not in linha.index:
        return None
    v = linha[campo]
    return None if v is None or pd.isna(v) else v


def _pessoas(dados, doc_uid: str) -> tuple[str, str, str, str]:
    """Nomes, uids e a origem do primeiro remetente com nascimento exibível (D2)."""
    vinculos = dados.doc_pessoa
    recorte = vinculos[vinculos["doc_uid"] == doc_uid].sort_values("posicao")
    nomes = [str(n) for n in recorte["nome"] if n is not None and not pd.isna(n)]
    uids = [str(u) for u in recorte["pessoa_uid"] if u is not None and not pd.isna(u)]
    origem, nivel = "", ""
    if uids:
        pessoas = dados.pessoas.set_index("pessoa_uid")
        for uid in uids:
            if uid not in pessoas.index:
                continue
            p = pessoas.loc[uid]
            if not bool(p["nascimento_exibivel"]):
                continue
            partes = [str(p[c]) for c in ("cidade_nascimento", "pais_atual")
                      if c in p.index and not pd.isna(p[c])]
            origem = ", ".join(partes)
            nivel = "" if pd.isna(p.get("nivel")) else str(p["nivel"])
            break
    return "; ".join(nomes), "; ".join(uids), origem, nivel


def linhas(dados, doc_uids, colunas=COLUNAS) -> list[dict]:
    """Uma linha por documento, na ordem dada, com as colunas do §8.2."""
    docs = dados.documentos
    saida = []
    for uid in doc_uids:
        linha = docs.iloc[dados.indice_de(uid)]
        nomes, uids_pessoas, origem, nivel = _pessoas(dados, uid)
        calculados = {
            "tipologia_a_revisar": "sim" if not _valor(linha, "typology_status") else "nao",
            "pessoas": nomes,
            "pessoas_uid": uids_pessoas,
            "origem_remetente": origem,
            "origem_nivel": nivel,
            "link_digitarq": links.digitarq(_valor(linha, "reference_code")
                                            or _valor(linha, "new_code")) or "",
            "link_resgate": links.resgate(),
        }
        registro = {}
        for nome, campo in colunas:
            if campo is None:
                registro[nome] = calculados.get(nome, "")
            else:
                valor = _valor(linha, campo)
                if nome in ("ano", "srsp") and valor is not None:
                    valor = int(valor)
                registro[nome] = "" if valor is None else valor
        saida.append(registro)
    return saida


def tabela(dados, doc_uids) -> pd.DataFrame:
    return pd.DataFrame(linhas(dados, doc_uids), columns=[n for n, _ in COLUNAS])


# ---------------------------------------------------------------------------
# CSV (UTF-8 com BOM, `;`) — §8.2
# ---------------------------------------------------------------------------

def _linhas_de_proveniencia(prov: dict) -> list[str]:
    """A proveniência em linhas de comentário, legíveis sem abrir o JSON."""
    app = prov.get("app_versao") or {}
    camadas = "; ".join(
        f"{papel}={c['camada']}" + (f" v{c['versao']}" if c.get("versao") else "")
        for papel, c in (prov.get("corpus", {}).get("camadas_ativas") or {}).items())
    consulta = prov.get("consulta") or {}
    filtros = consulta.get("filtros_ativos") or {}
    return [
        f"{COMENTARIO} {prov.get('app')} {app.get('versao') or '?'}"
        + (f" (commit {app['commit']})" if app.get("commit") else ""),
        f"{COMENTARIO} gerado em: {prov.get('gerado_em')}"
        + (f" · origem: {prov['origem']}" if prov.get("origem") else ""),
        f"{COMENTARIO} documentos: {prov.get('documentos')}",
        f"{COMENTARIO} consulta: texto={consulta.get('texto', '')!r} "
        f"modo={consulta.get('modo')} rigor={consulta.get('rigor')} "
        f"lente={consulta.get('lente')} ordenacao={consulta.get('ordenacao')} "
        f"semente={consulta.get('semente')}",
        f"{COMENTARIO} filtros ativos: {filtros or 'nenhum'}",
        f"{COMENTARIO} camadas ativas: {camadas or '?'}",
        f"{COMENTARIO} modelo de embeddings: "
        f"{(prov.get('corpus', {}).get('modelo_embeddings') or {}).get('nome')}",
        f"{COMENTARIO} manifesto (sha256): "
        f"{prov.get('corpus', {}).get('manifesto_sha256')}",
        f"{COMENTARIO} citação: {prov.get('citacao')} "
        f"doi do preprint: {(prov.get('doi') or {}).get('preprint')}",
        f"{COMENTARIO} {prov.get('credito')}",
    ]


def para_csv(dados, doc_uids, prov: dict | None = None) -> bytes:
    """O CSV do §8.2: UTF-8 com BOM, `;`, proveniência em comentário antes do cabeçalho."""
    prov = prov if prov is not None else proveniencia(dados, total=len(list(doc_uids)))
    registros = linhas(dados, doc_uids)
    buffer = io.StringIO(newline="")
    buffer.write(BOM)
    for linha in _linhas_de_proveniencia(prov):
        buffer.write(linha.replace("\r", " ").replace("\n", " ") + "\r\n")
    escritor = csv.DictWriter(buffer, fieldnames=[n for n, _ in COLUNAS],
                              delimiter=SEPARADOR_CSV, lineterminator="\r\n",
                              quoting=csv.QUOTE_MINIMAL)
    escritor.writeheader()
    for registro in registros:
        escritor.writerow(registro)
    return buffer.getvalue().encode("utf-8")


def ler_csv(bruto: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Reabre o que `para_csv` escreveu. Devolve a tabela e as linhas de proveniência.

    Só as linhas de comentário **do início** são tiradas: um `#` dentro de um resumo fica onde
    está. É esta função que o teste usa para cobrar a ida e volta do §8.2.
    """
    texto = bruto.decode("utf-8-sig")
    linhas_todas = texto.splitlines()
    cabecalho = 0
    while cabecalho < len(linhas_todas) and linhas_todas[cabecalho].startswith(COMENTARIO):
        cabecalho += 1
    prov = linhas_todas[:cabecalho]
    corpo = "\n".join(linhas_todas[cabecalho:])
    tabela_lida = pd.read_csv(io.StringIO(corpo), sep=SEPARADOR_CSV, dtype=str,
                              keep_default_na=False)
    return tabela_lida, prov


# ---------------------------------------------------------------------------
# JSON — §8.2
# ---------------------------------------------------------------------------

def para_json(dados, doc_uids, prov: dict | None = None) -> bytes:
    """O JSON do §8.2: as mesmas colunas, mais a justificativa do DeepSeek, mais proveniência."""
    doc_uids = list(doc_uids)
    prov = prov if prov is not None else proveniencia(dados, total=len(doc_uids))
    conteudo = {
        "schema": "exportacao-ahu-sul/1",
        "proveniencia": prov,
        "documentos": linhas(dados, doc_uids, COLUNAS_JSON),
    }
    return json.dumps(conteudo, ensure_ascii=False, indent=2, default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# D1 conferida de novo, de graça
# ---------------------------------------------------------------------------

def campos_proibidos_presentes() -> list[str]:
    """Os nomes do §D1 que por acaso estejam na lista de colunas. Deve ser sempre vazio."""
    nomes = {n for n, _ in COLUNAS_JSON} | {c for _, c in COLUNAS_JSON if c}
    return sorted(nomes & set(CAMPOS_PROIBIDOS))
