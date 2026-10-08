"""construir_app.py — WP 1.5: junta as camadas ativas nos artefatos que o app lê (spec §3.4).

Entrada : corpus/ (base + camadas ativas, segundo o corpus/manifest.json)
Saídas  : app_dados/
              manifest.json        manifesto do corpus + camadas usadas + hash de cada artefato
              documentos.parquet   base + tipologia + sociolinguística, uma linha por doc_uid
              pessoas.parquet      pessoas com o nascimento exibível (D2) e o nível
              doc_pessoa.parquet   doc_uid x pessoa_uid x posicao x tipo
              pdf_extra.parquet    só o que vai para o PDF (D1): scribe_mediation_likely
              embeddings.f16.npy   linha i = documentos.parquet linha i
              lexico.pkl           índice BM25 por campos (§5.2), pronto para consulta

O que este script garante, para que o app não precise lembrar (A1, D1, D2):
  * `scribe_mediation_likely` sai de `documentos.parquet` e vive só em `pdf_extra.parquet`;
  * biografia não exibível (REJEITADO, BAIXA ou `decisao_manual = rejeitar`) **perde o dado**:
    nem local, nem ano, nem fonte, nem nível, nem Wikidata chegam ao app. A pessoa continua na
    lista, com o rótulo da §6.3;
  * as coleções, as pastas e as tipologias saem dos dados (D6): acrescentar uma coleção em
    `corpus/colecoes/` e reconstruir basta, sem tocar no código (A11).

Uso:
    python build/construir_app.py
    python build/construir_app.py --corpus corpus --saida app_dados
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ahu.texto import tokenizar  # noqa: E402
from build.importar_base import ErroImportacao, commit_atual, sha256_arquivo  # noqa: E402

VERSAO_SCRIPT = "1.0.0"
VERSAO_ARTEFATOS = "1"

# Pesos iniciais da §5.2. Calibrados no conjunto de teste (§5.6, WP 4.2).
PESOS_CAMPOS = {"pessoas": 2.0, "destinatario": 1.5, "tipologia": 1.5, "descricao": 1.0, "pasta": 0.5}

# Campos que só vão para o dossiê em PDF (D1): nunca para a interface, filtros, CSV ou JSON.
CAMPOS_PDF = ["scribe_mediation_likely"]

COLUNAS_BASE = ["doc_uid", "colecao", "folder", "description", "old_code", "new_code",
                "reference_code", "document_id_and_date", "ano_isolado", "recipient_name",
                "sender_name"]
COLUNAS_TIPOLOGIA = ["tipologia", "tipologia_chave", "typology_status"]
COLUNAS_SOCIO = ["sender_category", "vector", "typology_impact", "vernacular_score",
                 "justificativa"]
# Dados de nascimento: só chegam ao app quando a biografia é exibível (D2).
COLUNAS_NASCIMENTO = ["nivel", "cidade_nascimento", "regiao_nascimento", "pais_nascimento_epoca",
                      "pais_atual", "ano_nascimento", "ano_morte", "nome_na_fonte", "url_fonte",
                      "trecho_fonte", "origem_local", "wikidata_qid"]


# ---------------------------------------------------------------------------
# Leitura das camadas ativas
# ---------------------------------------------------------------------------

def ler_manifesto(corpus: Path) -> dict:
    arq = corpus / "manifest.json"
    if not arq.exists():
        raise ErroImportacao(f"{arq} não existe. Rode build/importar_base.py antes (WP 1.1).")
    with open(arq, encoding="utf-8") as f:
        return json.load(f)


def camada_ativa(manifesto: dict, tipo: str, obrigatoria: bool = True) -> str | None:
    """Nome da camada ativa de um tipo. O manifesto decide qual é, não o código (spec §3.1)."""
    ativas = [nome for nome, c in manifesto.get("camadas", {}).items()
              if c.get("tipo") == tipo and c.get("ativa")]
    if len(ativas) > 1:
        raise ErroImportacao(
            f"Há {len(ativas)} camadas de tipo '{tipo}' ativas ({', '.join(sorted(ativas))}). "
            "Deixe uma só ativa no corpus/manifest.json."
        )
    if not ativas:
        if obrigatoria:
            raise ErroImportacao(
                f"Nenhuma camada de tipo '{tipo}' está ativa no manifesto. "
                "Rode os scripts de construção do corpus (WP 1.1 a 1.4)."
            )
        return None
    return ativas[0]


def _ler_parquet(caminho: Path, o_que: str) -> pd.DataFrame:
    if not caminho.exists():
        raise ErroImportacao(f"{o_que} não encontrado: {caminho}. Reconstrua a camada.")
    return pd.read_parquet(caminho)


def ler_base(corpus: Path, manifesto: dict) -> pd.DataFrame:
    """Todas as coleções do manifesto, concatenadas na ordem em que aparecem (D6, A11)."""
    partes = []
    for colecao in manifesto.get("colecoes", {}):
        caminho = corpus / "colecoes" / colecao / "documentos.parquet"
        partes.append(_ler_parquet(caminho, f"base da coleção {colecao}"))
    if not partes:
        raise ErroImportacao("O manifesto não tem nenhuma coleção. Rode build/importar_base.py.")
    base = pd.concat(partes, ignore_index=True)
    if not base["doc_uid"].is_unique:
        repetidos = base.loc[base["doc_uid"].duplicated(), "doc_uid"].head(3).tolist()
        raise ErroImportacao(f"doc_uid repetido entre coleções (ex.: {', '.join(repetidos)}).")
    return base


# ---------------------------------------------------------------------------
# documentos.parquet e pdf_extra.parquet
# ---------------------------------------------------------------------------

def montar_documentos(
    base: pd.DataFrame, tipologia: pd.DataFrame, socio: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (documentos, pdf_extra). O `scribe_mediation_likely` só vai no segundo (D1)."""
    for nome, camada in (("tipologia", tipologia), ("sociolinguística", socio)):
        faltam = set(base["doc_uid"]) - set(camada["doc_uid"])
        if faltam:
            raise ErroImportacao(
                f"{len(faltam)} documentos da base não estão na camada de {nome} "
                f"(ex.: {', '.join(sorted(faltam)[:3])}). Reconstrua a camada."
            )

    docs = base[[c for c in COLUNAS_BASE if c in base.columns]].copy()
    docs = docs.merge(tipologia[["doc_uid", *COLUNAS_TIPOLOGIA]], on="doc_uid", how="left")
    docs = docs.merge(socio[["doc_uid", *COLUNAS_SOCIO]], on="doc_uid", how="left")

    vazamento = [c for c in CAMPOS_PDF if c in docs.columns]
    if vazamento:
        raise ErroImportacao(f"D1: {', '.join(vazamento)} não pode estar em documentos.parquet.")

    pdf_extra = socio[["doc_uid", *CAMPOS_PDF]].copy()
    pdf_extra = docs[["doc_uid"]].merge(pdf_extra, on="doc_uid", how="left")
    return docs, pdf_extra


# ---------------------------------------------------------------------------
# pessoas.parquet e doc_pessoa.parquet
# ---------------------------------------------------------------------------

def montar_pessoas(pessoas: pd.DataFrame, biografias: pd.DataFrame) -> pd.DataFrame:
    """Pessoas com o nascimento exibível (D2). O resto do nascimento não chega ao app."""
    colunas = ["pessoa_uid", "exibivel", "rotulo", *COLUNAS_NASCIMENTO]
    faltam = [c for c in colunas if c not in biografias.columns]
    if faltam:
        raise ErroImportacao(
            f"A camada de biografias não tem {', '.join(faltam)}. Reconstrua-a (WP 1.3)."
        )
    df = pessoas.merge(biografias[colunas], on="pessoa_uid", how="left")

    # A1/D2: quem não é exibível perde o dado aqui, no construtor, e não no app.
    nao_exibivel = df["exibivel"] != True  # noqa: E712  (NaN também conta como não exibível)
    df.loc[nao_exibivel, COLUNAS_NASCIMENTO] = None
    df["nascimento_exibivel"] = (~nao_exibivel).astype(bool)
    df["rotulo"] = df["rotulo"].fillna("Não identificado")
    return df.drop(columns=["exibivel"])


def montar_doc_pessoa(entidades: pd.DataFrame, pessoas: pd.DataFrame) -> pd.DataFrame:
    colunas = ["doc_uid", "pessoa_uid", "posicao", "tipo", "nome", "autoria_verificada"]
    df = entidades[[c for c in colunas if c in entidades.columns]].copy()
    desconhecidas = set(df["pessoa_uid"].dropna()) - set(pessoas["pessoa_uid"])
    if desconhecidas:
        raise ErroImportacao(
            f"{len(desconhecidas)} pessoa_uid das entidades não estão na camada de pessoas "
            f"(ex.: {', '.join(sorted(desconhecidas)[:3])}). Reconstrua as camadas (WP 1.3)."
        )
    return df


# ---------------------------------------------------------------------------
# lexico.pkl — BM25 por campos (§5.2)
# ---------------------------------------------------------------------------

def _lista(valor) -> list[str]:
    """Coluna de lista do Parquet volta como array do numpy; None e NaN voltam vazios."""
    if valor is None or isinstance(valor, float):
        return []
    return [str(v) for v in valor if v is not None]


def _nomes_por_pessoa(pessoas: pd.DataFrame) -> dict[str, list[str]]:
    """pessoa_uid -> nome principal, variantes e títulos (campo 'pessoas' da §5.2)."""
    partes_de = {}
    for uid, principal, variantes, titulos in zip(
            pessoas["pessoa_uid"], pessoas["nome_principal"], pessoas["variantes_nome"],
            pessoas["titulos"]):
        partes = [] if principal is None or pd.isna(principal) else [str(principal)]
        partes_de[uid] = partes + _lista(variantes) + _lista(titulos)
    return partes_de


def textos_por_campo(docs: pd.DataFrame, doc_pessoa: pd.DataFrame,
                     pessoas: pd.DataFrame) -> dict[str, list[str]]:
    """Um texto por documento e por campo da §5.2, na ordem de `docs`."""
    partes_de = _nomes_por_pessoa(pessoas)
    por_doc: dict[str, list[str]] = {}
    for doc_uid, pessoa_uid, nome in zip(doc_pessoa["doc_uid"], doc_pessoa["pessoa_uid"],
                                         doc_pessoa["nome"]):
        partes = [] if nome is None or pd.isna(nome) else [str(nome)]
        partes += partes_de.get(pessoa_uid, [])
        por_doc.setdefault(doc_uid, []).extend(p for p in partes if p)

    def coluna(nome: str) -> list[str]:
        if nome not in docs.columns:
            return [""] * len(docs)
        return ["" if v is None or v is pd.NA else str(v) for v in docs[nome].fillna("")]

    return {
        "pessoas": [" ".join(dict.fromkeys(por_doc.get(u, []))) for u in docs["doc_uid"]],
        "destinatario": coluna("recipient_name"),
        "tipologia": coluna("tipologia"),
        "descricao": coluna("description"),
        "pasta": coluna("folder"),
    }


def _corrigir_idf(bm25, n_documentos: int) -> None:
    """Troca o IDF do `rank_bm25` pelo do Lucene, que nunca é negativo.

    O `BM25Okapi` usa `log((N - df + 0,5) / (df + 0,5))`, que fica **negativo** para um termo que
    está em mais da metade do corpus, e então o substitui por `epsilon * idf_médio` — um valor que
    pode ser negativo e não depende do termo. Com isto, 'OFÍCIO' (47% dos documentos) e 'carta'
    teriam nota arbitrária, às vezes abaixo de zero. A fórmula do Lucene,
    `log(1 + (N - df + 0,5) / (df + 0,5))`, é sempre positiva e cai junto com a frequência: um
    termo em todos os documentos quase não pesa, mas também não atrapalha.
    """
    frequencia: dict[str, int] = {}
    for doc in bm25.doc_freqs:
        for termo in doc:
            frequencia[termo] = frequencia.get(termo, 0) + 1
    bm25.idf = {
        termo: math.log(1 + (n_documentos - df + 0.5) / (df + 0.5))
        for termo, df in frequencia.items()
    }
    bm25.average_idf = (sum(bm25.idf.values()) / len(bm25.idf)) if bm25.idf else 0.0


def construir_lexico(docs: pd.DataFrame, doc_pessoa: pd.DataFrame,
                     pessoas: pd.DataFrame) -> dict:
    """Índice BM25 por campo, pronto para o app (§3.4: o único pickle, gerado por nós)."""
    from importlib import metadata

    from rank_bm25 import BM25Okapi

    textos = textos_por_campo(docs, doc_pessoa, pessoas)
    indices, normalizados, vazios = {}, {}, {}
    for campo, valores in textos.items():
        tokens = [tokenizar(v) for v in valores]
        vazios[campo] = int(sum(1 for t in tokens if not t))
        # Os tokens já normalizados, juntos num texto: é neles que a busca por frase entre aspas
        # procura a sequência literal (§5.5) e de onde sai a evidência de termos (ahu/busca.py).
        normalizados[campo] = [" ".join(t) for t in tokens]
        # BM25Okapi divide por zero se o corpus for todo vazio; um token-sentinela evita isso.
        bm25 = BM25Okapi([t or ["\x00"] for t in tokens])
        _corrigir_idf(bm25, len(tokens))
        indices[campo] = bm25
    return {
        "versao": VERSAO_ARTEFATOS,
        "rank_bm25": metadata.version("rank-bm25"),
        "pesos": dict(PESOS_CAMPOS),
        "campos": list(textos),
        "doc_uids": docs["doc_uid"].astype(str).tolist(),
        "indices": indices,
        "normalizados": normalizados,
        "documentos_sem_token": vazios,
    }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def alinhar_embeddings(pasta_camada: Path, docs: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Reordena os vetores da camada para a ordem de `documentos.parquet` (§3.4)."""
    indice = _ler_parquet(pasta_camada / "indice.parquet", "índice de embeddings")
    arq = pasta_camada / "embeddings.f16.npy"
    if not arq.exists():
        raise ErroImportacao(f"Embeddings não encontrados: {arq}. Rode build/construir_indice.py.")
    emb = np.load(arq)
    if emb.shape[0] != len(indice):
        raise ErroImportacao(
            f"A camada de embeddings tem {emb.shape[0]} vetores e {len(indice)} uids. "
            "Reconstrua o índice (WP 1.4)."
        )
    posicao = {uid: i for i, uid in enumerate(indice["doc_uid"].astype(str))}
    faltam = [u for u in docs["doc_uid"] if u not in posicao]
    if faltam:
        raise ErroImportacao(
            f"{len(faltam)} documentos não têm vetor (ex.: {', '.join(faltam[:3])}). "
            "Rode build/construir_indice.py depois de reimportar a base."
        )
    ordem = [posicao[u] for u in docs["doc_uid"]]
    with open(pasta_camada / "camada.json", encoding="utf-8") as f:
        camada = json.load(f)
    return emb[ordem], camada


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

def construir(corpus: Path, saida: Path) -> dict:
    corpus, saida = Path(corpus), Path(saida)
    manifesto = ler_manifesto(corpus)
    usadas = {
        "tipologia": camada_ativa(manifesto, "tipologia"),
        "sociolinguistica": camada_ativa(manifesto, "sociolinguistica"),
        "entidades": camada_ativa(manifesto, "entidades"),
        "biografias": camada_ativa(manifesto, "biografias"),
        "embeddings": camada_ativa(manifesto, "embeddings"),
    }
    pasta = corpus / "camadas"

    base = ler_base(corpus, manifesto)
    tipologia = _ler_parquet(pasta / usadas["tipologia"] / "tipologia.parquet", "camada de tipologia")
    socio = _ler_parquet(pasta / usadas["sociolinguistica"] / "sociolinguistica.parquet",
                         "camada sociolinguística")
    entidades = _ler_parquet(pasta / usadas["entidades"] / "entidades.parquet", "camada de entidades")
    pessoas_camada = _ler_parquet(pasta / usadas["entidades"] / "pessoas.parquet", "camada de pessoas")
    biografias = _ler_parquet(pasta / usadas["biografias"] / "biografias.parquet",
                              "camada de biografias")

    docs, pdf_extra = montar_documentos(base, tipologia, socio)
    pessoas = montar_pessoas(pessoas_camada, biografias)
    doc_pessoa = montar_doc_pessoa(entidades, pessoas)
    doc_pessoa = doc_pessoa[doc_pessoa["doc_uid"].isin(set(docs["doc_uid"]))].reset_index(drop=True)
    embeddings, camada_emb = alinhar_embeddings(pasta / usadas["embeddings"], docs)
    lexico = construir_lexico(docs, doc_pessoa, pessoas)

    saida.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(saida / "documentos.parquet", index=False)
    pessoas.to_parquet(saida / "pessoas.parquet", index=False)
    doc_pessoa.to_parquet(saida / "doc_pessoa.parquet", index=False)
    pdf_extra.to_parquet(saida / "pdf_extra.parquet", index=False)
    np.save(saida / "embeddings.f16.npy", embeddings.astype(np.float16))
    with open(saida / "lexico.pkl", "wb") as f:
        pickle.dump(lexico, f, protocol=pickle.HIGHEST_PROTOCOL)

    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artefatos = ["documentos.parquet", "pessoas.parquet", "doc_pessoa.parquet", "pdf_extra.parquet",
                 "embeddings.f16.npy", "lexico.pkl"]
    app_manifesto = {
        "versao_artefatos": VERSAO_ARTEFATOS,
        "gerado_em": agora,
        "processo": {"script": "build/construir_app.py", "versao_script": VERSAO_SCRIPT,
                     "commit": commit_atual()},
        "corpus": {"versao_corpus": manifesto.get("versao_corpus"),
                   "gerado_em": manifesto.get("gerado_em"),
                   "colecoes": sorted(manifesto.get("colecoes", {})),
                   "camadas": {t: n for t, n in usadas.items() if n},
                   "manifesto": manifesto},
        "modelo_embeddings": {
            "nome": camada_emb.get("modelo"), "dimensao": camada_emb.get("dimensao"),
            "prefixo_consulta": camada_emb.get("prefixo_consulta"),
            "normalizado_l2": camada_emb.get("normalizado_l2"),
        },
        "lexico": {"rank_bm25": lexico["rank_bm25"], "pesos": lexico["pesos"],
                   "campos": lexico["campos"]},
        "contagens": {
            "documentos": int(len(docs)), "pessoas": int(len(pessoas)),
            "doc_pessoa": int(len(doc_pessoa)),
            "nascimento_exibivel": int(pessoas["nascimento_exibivel"].sum()),
            "colecoes": {str(k): int(v) for k, v in docs["colecao"].value_counts().items()},
        },
        # O app confere esta lista de uids, nesta ordem, contra documentos.parquet (§3.4, C7).
        "doc_uids_sha256": _sha256_texto("\n".join(docs["doc_uid"].astype(str))),
        "pessoa_uids_sha256": _sha256_texto("\n".join(pessoas["pessoa_uid"].astype(str))),
        "artefatos": {nome: sha256_arquivo(saida / nome) for nome in artefatos},
    }
    with open(saida / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(app_manifesto, f, ensure_ascii=False, indent=2)

    return {
        "documentos": len(docs), "pessoas": len(pessoas), "doc_pessoa": len(doc_pessoa),
        "nascimento_exibivel": int(pessoas["nascimento_exibivel"].sum()),
        "colecoes": app_manifesto["contagens"]["colecoes"],
        "camadas": {t: n for t, n in usadas.items() if n},
        "dimensao": int(embeddings.shape[1]),
    }


def _sha256_texto(texto: str) -> str:
    import hashlib

    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    ap.add_argument("--saida", default=str(RAIZ / "app_dados"))
    args = ap.parse_args()
    try:
        r = construir(Path(args.corpus), Path(args.saida))
    except ErroImportacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"{r['documentos']} documentos, {r['pessoas']} pessoas "
          f"({r['nascimento_exibivel']} com nascimento exibível), "
          f"{r['doc_pessoa']} vínculos, embeddings de {r['dimensao']} dimensões.")
    print("Coleções: " + ", ".join(f"{k}={v}" for k, v in r["colecoes"].items()))
    print("Camadas: " + ", ".join(f"{k}={v}" for k, v in r["camadas"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
