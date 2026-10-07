"""importar_base.py — WP 1.1: importa a base de documentos do AHU e cria os `doc_uid`.

Entrada : ahu_sul_catalog_biografias.json (saída do p5 do pipeline de biografias v2)
Saídas  : corpus/colecoes/<colecao>/documentos.parquet   (só os campos do AHU)
          corpus/colecoes/<colecao>/registro_ids.json    (como cada documento recebeu o doc_uid)
          corpus/colecoes/<colecao>/camada.json          (proveniência da base)
          corpus/manifest.json                           (coleções, camadas e SHA-256 de cada arquivo)

Regras (spec §3.1, §3.3):
  * doc_uid = "<colecao>:" + 10 hex do SHA-1 de
        reference_code | new_code | document_id_and_date | descricao[:200]   (normalizados);
  * numa reimportação, um documento que já existe mantém o uid, mesmo que a descrição seja corrigida.
    O casamento é: (1) conteúdo idêntico; (2) reference_code + new_code + número do item;
  * colisões são resolvidas por sufixo determinístico e nunca reaproveitam um uid em uso.

Uso: python build/importar_base.py [--entrada CAMINHO] [--corpus corpus] [--colecao ahu-sul] [--esperado 7051]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA_PADRAO = RAIZ.parent / "biografias_v2" / "ahu_sul_catalog_biografias.json"
VERSAO_SCRIPT = "1.0.0"
ESPERADO_AHU_SUL = 7051

# Campos que vêm do catálogo do AHU e formam a base (plano WP 1.1, tarefa 1).
CAMPOS_BASE = [
    "folder", "description", "old_code", "new_code", "reference_code",
    "document_id_and_date", "ano_isolado", "extracted_typology", "sender_name", "recipient_name",
]
CAMPOS_TEXTO = [c for c in CAMPOS_BASE if c != "ano_isolado"]
CAMPOS_OBRIGATORIOS = ["description", "reference_code", "document_id_and_date"]
CAMPOS_PROIBIDOS = ["perplexity_search_data", "quarentena_search_data"]

RE_ITEM = re.compile(r"^\s*(\d+)")


class ErroImportacao(Exception):
    """Falha de contrato: a importação para com uma mensagem clara."""


# ---------------------------------------------------------------------------
# Normalização e identificadores
# ---------------------------------------------------------------------------

def normalizar(texto) -> str:
    """NFKD, sem acentos, minúsculas, espaços colapsados. None vira ''."""
    if texto is None:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", t).strip()


def numero_item(document_id_and_date) -> str:
    """'2673. 1686, Fevereiro, 12, Lisboa' -> '2673'."""
    m = RE_ITEM.match(document_id_and_date or "")
    return m.group(1) if m else ""


def hash_conteudo(doc: dict) -> str:
    partes = [
        normalizar(doc.get("reference_code")),
        normalizar(doc.get("new_code")),
        normalizar(doc.get("document_id_and_date")),
        normalizar(doc.get("description"))[:200],
    ]
    return "|".join(partes)


def sha1_hex(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def chave_cota(doc: dict) -> str:
    """reference_code + new_code + número do item (spec §3.3)."""
    return "|".join([
        normalizar(doc.get("reference_code")),
        normalizar(doc.get("new_code")),
        numero_item(doc.get("document_id_and_date")),
    ])


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def commit_atual() -> str:
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True, timeout=10,
        )
        return saida.stdout.strip() if saida.returncode == 0 and saida.stdout.strip() else "sem-commit"
    except (OSError, subprocess.SubprocessError):
        return "sem-commit"


# ---------------------------------------------------------------------------
# Validação do contrato de entrada
# ---------------------------------------------------------------------------

def validar_entrada(docs, esperado: int | None) -> None:
    if not isinstance(docs, list):
        raise ErroImportacao(
            "O catálogo de entrada deve ser uma lista de documentos (JSON truncado ou formato errado)."
        )
    if esperado is not None and len(docs) != esperado:
        raise ErroImportacao(f"Esperados {esperado} documentos, encontrados {len(docs)}.")
    for i, d in enumerate(docs):
        if not isinstance(d, dict):
            raise ErroImportacao(f"Documento {i} não é um objeto JSON.")
        faltam = [c for c in CAMPOS_OBRIGATORIOS if not d.get(c)]
        if faltam:
            raise ErroImportacao(
                f"Documento {i}: campos obrigatórios vazios ou ausentes: {', '.join(faltam)}."
            )
        proibidos = [c for c in CAMPOS_PROIBIDOS if c in d]
        if proibidos:
            raise ErroImportacao(
                f"Documento {i}: sobrou campo do pipeline antigo ({', '.join(proibidos)}). Rode o p5 de novo."
            )


# ---------------------------------------------------------------------------
# Atribuição de doc_uid
# ---------------------------------------------------------------------------

def atribuir_uids(docs: list[dict], colecao: str, registro_anterior: dict | None) -> list[dict]:
    """Devolve, para cada documento (mesma ordem), um registro com doc_uid, hash e chave de cota."""
    anterior = (registro_anterior or {}).get("documentos", [])
    por_hash: dict[str, list[dict]] = {}
    por_cota: dict[str, list[dict]] = {}
    for r in anterior:
        por_hash.setdefault(r["hash_conteudo"], []).append(r)
        por_cota.setdefault(r["chave_cota"], []).append(r)

    usados: set[str] = set()
    uids: list[str | None] = [None] * len(docs)
    hashes = [hash_conteudo(d) for d in docs]
    cotas = [chave_cota(d) for d in docs]

    def pegar(lista: list[dict] | None) -> str | None:
        for r in lista or []:
            if r["doc_uid"] not in usados:
                return r["doc_uid"]
        return None

    # Passo 1: conteúdo idêntico. Passo 2: cota + número do item (descrição corrigida).
    for passo, indice, chaves in ((1, por_hash, hashes), (2, por_cota, cotas)):
        for i in range(len(docs)):
            if uids[i] is None:
                uid = pegar(indice.get(chaves[i]))
                if uid:
                    uids[i] = uid
                    usados.add(uid)

    # Passo 3: uids novos. Nunca reaproveita um uid já registrado (mesmo de documento removido).
    reservados = {r["doc_uid"] for r in anterior}
    for i in range(len(docs)):
        if uids[i] is not None:
            continue
        n = 1
        while True:
            base = hashes[i] if n == 1 else f"{hashes[i]}|#{n}"
            uid = f"{colecao}:{sha1_hex(base)[:10]}"
            if uid not in usados and uid not in reservados:
                break
            n += 1
        uids[i] = uid
        usados.add(uid)

    if len(set(uids)) != len(uids):
        raise ErroImportacao("Colisão de doc_uid não resolvida (não deveria ocorrer).")

    return [
        {"doc_uid": uids[i], "hash_conteudo": hashes[i], "chave_cota": cotas[i],
         "reference_code": docs[i].get("reference_code"), "new_code": docs[i].get("new_code")}
        for i in range(len(docs))
    ]


# ---------------------------------------------------------------------------
# Construção da base
# ---------------------------------------------------------------------------

def montar_tabela(docs: list[dict], registros: list[dict], colecao: str) -> pd.DataFrame:
    linhas = []
    for d, r in zip(docs, registros):
        linha = {"doc_uid": r["doc_uid"], "colecao": colecao}
        for c in CAMPOS_TEXTO:
            v = d.get(c)
            linha[c] = None if v is None else str(v)
        linha["ano_isolado"] = d.get("ano_isolado")
        linhas.append(linha)
    df = pd.DataFrame(linhas, columns=["doc_uid", "colecao", *CAMPOS_BASE])
    df["ano_isolado"] = pd.to_numeric(df["ano_isolado"], errors="coerce").astype("Int64")
    return df


def importar(
    entrada: Path, corpus: Path, colecao: str = "ahu-sul", esperado: int | None = ESPERADO_AHU_SUL
) -> dict:
    entrada = Path(entrada)
    corpus = Path(corpus)
    if not entrada.exists():
        raise ErroImportacao(f"Arquivo de entrada não encontrado: {entrada}")
    try:
        with open(entrada, encoding="utf-8") as f:
            docs = json.load(f)
    except json.JSONDecodeError as e:
        raise ErroImportacao(f"JSON inválido ou truncado em {entrada}: {e}") from e

    validar_entrada(docs, esperado)

    pasta = corpus / "colecoes" / colecao
    pasta.mkdir(parents=True, exist_ok=True)
    arq_registro = pasta / "registro_ids.json"
    registro_anterior = None
    if arq_registro.exists():
        with open(arq_registro, encoding="utf-8") as f:
            registro_anterior = json.load(f)

    registros = atribuir_uids(docs, colecao, registro_anterior)
    df = montar_tabela(docs, registros, colecao)

    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # O registro também guarda os uids de documentos que saíram, para nunca serem reaproveitados.
    ativos = {r["doc_uid"] for r in registros}
    antigos = (registro_anterior or {}).get("documentos", [])
    aposentados = [r for r in antigos if r["doc_uid"] not in ativos]
    registro = {
        "colecao": colecao,
        "regra": "doc_uid = colecao:sha1(ref|new|document_id_and_date|descricao[:200] normalizados)[:10]",
        "gerado_em": agora,
        "documentos": registros + aposentados,
    }
    with open(arq_registro, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=1)

    arq_parquet = pasta / "documentos.parquet"
    df.to_parquet(arq_parquet, index=False)

    camada = {
        "nome": f"base_{colecao}",
        "tipo": "base",
        "versao": "1",
        "colecao": colecao,
        "processo": {
            "script": "build/importar_base.py", "versao_script": VERSAO_SCRIPT, "commit": commit_atual(),
        },
        "entrada": {"arquivo": entrada.name, "sha256": sha256_arquivo(entrada)},
        "data": agora,
        "documentos": int(len(df)),
        "esquema": {c: str(t) for c, t in df.dtypes.items()},
        "origem": "Arquivo Histórico Ultramarino (AHU) e Projeto Resgate 'Barão do Rio Branco'. "
                  "Resumos e cotas pertencem a essas instituições e são creditados como tal.",
    }
    with open(pasta / "camada.json", "w", encoding="utf-8") as f:
        json.dump(camada, f, ensure_ascii=False, indent=2)

    manifesto = atualizar_manifesto(corpus, colecao, agora)
    return {"documentos": len(df), "uids_unicos": int(df["doc_uid"].nunique()),
            "novos": sum(1 for r in registros if r["doc_uid"] not in
                         {x["doc_uid"] for x in (registro_anterior or {}).get("documentos", [])}),
            "manifesto": manifesto}


def atualizar_manifesto(corpus: Path, colecao: str, agora: str) -> dict:
    arq = corpus / "manifest.json"
    manifesto = {"versao_corpus": "0.1.0", "colecoes": {}, "camadas": {}}
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            manifesto = json.load(f)
    pasta = corpus / "colecoes" / colecao
    arquivos = {}
    for nome in ("documentos.parquet", "registro_ids.json", "camada.json"):
        arquivos[f"colecoes/{colecao}/{nome}"] = sha256_arquivo(pasta / nome)
    manifesto.setdefault("colecoes", {})[colecao] = {"atualizado_em": agora, "arquivos": arquivos}
    manifesto["gerado_em"] = agora
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)
    return manifesto


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(ENTRADA_PADRAO))
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    ap.add_argument("--colecao", default="ahu-sul")
    ap.add_argument("--esperado", type=int, default=ESPERADO_AHU_SUL,
                    help="número de documentos esperado (0 desliga a verificação)")
    args = ap.parse_args()
    try:
        r = importar(Path(args.entrada), Path(args.corpus), args.colecao, args.esperado or None)
    except ErroImportacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"{r['documentos']} documentos, {r['uids_unicos']} doc_uid únicos ({r['novos']} novos).")
    print(f"Manifesto: {Path(args.corpus) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
