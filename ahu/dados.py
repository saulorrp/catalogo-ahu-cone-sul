"""dados.py — carrega e valida `app_dados/` (spec §3.4, C7, A4).

O app chama `carregar()` uma vez. Se qualquer coisa estiver fora do lugar — hash diferente do
manifesto, número de vetores diferente do número de documentos, ordem de `doc_uid` diferente da
registrada — o carregamento **para** com uma mensagem que diz o que reconstruir. É isso que o
critério A4 cobra: um artefato desalinhado não pode produzir uma ficha com a biografia de outra
pessoa (C7).

O app só lê `app_dados/` (AGENTS.md 3). Nada aqui abre `corpus/`.

Uso no app:
    from ahu.dados import carregar
    dados = carregar()          # cacheado por st.cache_resource; nunca copia as tabelas
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
APP_DADOS = RAIZ / "app_dados"

ARTEFATOS_PARQUET = ("documentos.parquet", "pessoas.parquet", "doc_pessoa.parquet",
                     "pdf_extra.parquet")
ARTEFATOS = (*ARTEFATOS_PARQUET, "embeddings.f16.npy", "lexico.pkl")

# D1: nunca na interface, nos filtros, no CSV nem no JSON. Só no dossiê em PDF.
CAMPOS_SO_PDF = ("scribe_mediation_likely",)

RECONSTRUIR = "Rode `python build/construir_app.py` para reconstruir os artefatos."


class ErroDados(RuntimeError):
    """Artefatos ausentes, desalinhados ou com hash diferente do manifesto (A4)."""


@dataclass(frozen=True, eq=False)
class Dados:
    """Os artefatos do §3.4, já validados. As tabelas não são copiadas."""

    documentos: pd.DataFrame
    pessoas: pd.DataFrame
    doc_pessoa: pd.DataFrame
    pdf_extra: pd.DataFrame
    embeddings: np.ndarray
    lexico: dict
    manifesto: dict
    pasta: Path
    posicao: dict = field(repr=False, default_factory=dict)

    @property
    def colecoes(self) -> list[str]:
        """Vêm dos dados, nunca do código (D6, A11)."""
        return sorted(self.documentos["colecao"].dropna().unique().tolist())

    @property
    def pastas(self) -> list[str]:
        return sorted(self.documentos["folder"].dropna().unique().tolist())

    @property
    def tipologias(self) -> list[str]:
        return sorted(self.documentos["tipologia"].dropna().unique().tolist())

    @property
    def modelo_embeddings(self) -> dict:
        return self.manifesto.get("modelo_embeddings", {})

    def indice_de(self, doc_uid: str) -> int:
        """Linha de `documentos` (e do vetor) de um doc_uid."""
        try:
            return self.posicao[doc_uid]
        except KeyError:
            raise KeyError(f"doc_uid desconhecido: {doc_uid}") from None


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------

def _sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ler_manifesto(pasta: Path) -> dict:
    arq = pasta / "manifest.json"
    if not arq.exists():
        raise ErroDados(f"{arq} não existe. {RECONSTRUIR}")
    try:
        with open(arq, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ErroDados(f"{arq} está corrompido ({e}). {RECONSTRUIR}") from e


def verificar_hashes(pasta: Path, manifesto: dict) -> None:
    esperados = manifesto.get("artefatos") or {}
    if not esperados:
        raise ErroDados(f"O manifesto não lista os artefatos. {RECONSTRUIR}")
    for nome in ARTEFATOS:
        caminho = pasta / nome
        if not caminho.exists():
            raise ErroDados(f"Artefato ausente: {caminho}. {RECONSTRUIR}")
        if nome not in esperados:
            raise ErroDados(f"O manifesto não tem o hash de {nome}. {RECONSTRUIR}")
        obtido = _sha256_arquivo(caminho)
        if obtido != esperados[nome]:
            raise ErroDados(
                f"{nome} mudou depois da construção: o manifesto espera "
                f"{esperados[nome][:12]}... e o arquivo tem {obtido[:12]}.... {RECONSTRUIR}"
            )


def verificar_alinhamento(dados: dict, manifesto: dict) -> None:
    docs, pessoas = dados["documentos"], dados["pessoas"]
    emb, lexico = dados["embeddings"], dados["lexico"]

    if docs.empty:
        raise ErroDados(f"documentos.parquet está vazio. {RECONSTRUIR}")
    if not docs["doc_uid"].is_unique:
        raise ErroDados(f"documentos.parquet tem doc_uid repetido. {RECONSTRUIR}")
    if not pessoas["pessoa_uid"].is_unique:
        raise ErroDados(f"pessoas.parquet tem pessoa_uid repetido. {RECONSTRUIR}")

    if emb.shape[0] != len(docs):
        raise ErroDados(
            f"São {emb.shape[0]} vetores e {len(docs)} documentos. {RECONSTRUIR}"
        )
    dimensao = manifesto.get("modelo_embeddings", {}).get("dimensao")
    if dimensao and emb.ndim == 2 and emb.shape[1] != dimensao:
        raise ErroDados(
            f"Os vetores têm {emb.shape[1]} dimensões e o manifesto diz {dimensao}. {RECONSTRUIR}"
        )

    uids = docs["doc_uid"].astype(str).tolist()
    for nome, chave, obtidos in (("documentos.parquet", "doc_uids_sha256", uids),
                                 ("pessoas.parquet", "pessoa_uids_sha256",
                                  pessoas["pessoa_uid"].astype(str).tolist())):
        esperado = manifesto.get(chave)
        if esperado and _sha256_texto("\n".join(obtidos)) != esperado:
            raise ErroDados(
                f"A ordem dos uids de {nome} não é a registrada no manifesto. {RECONSTRUIR}"
            )

    if lexico.get("doc_uids") != uids:
        raise ErroDados(f"O léxico BM25 não está na ordem de documentos.parquet. {RECONSTRUIR}")
    esperado = (manifesto.get("lexico") or {}).get("rank_bm25")
    if esperado and lexico.get("rank_bm25") != esperado:
        raise ErroDados(
            f"O léxico foi gerado com rank-bm25 {lexico.get('rank_bm25')} e o manifesto diz "
            f"{esperado}. {RECONSTRUIR}"
        )

    orfaos = set(dados["doc_pessoa"]["doc_uid"]) - set(uids)
    if orfaos:
        raise ErroDados(
            f"{len(orfaos)} vínculos de doc_pessoa.parquet apontam para documentos que não existem "
            f"(ex.: {', '.join(sorted(orfaos)[:3])}). {RECONSTRUIR}"
        )
    sem_pessoa = set(dados["doc_pessoa"]["pessoa_uid"].dropna()) - set(pessoas["pessoa_uid"])
    if sem_pessoa:
        raise ErroDados(
            f"{len(sem_pessoa)} vínculos apontam para pessoas que não existem "
            f"(ex.: {', '.join(sorted(sem_pessoa)[:3])}). {RECONSTRUIR}"
        )
    if dados["pdf_extra"]["doc_uid"].astype(str).tolist() != uids:
        raise ErroDados(f"pdf_extra.parquet não está na ordem de documentos.parquet. {RECONSTRUIR}")


def verificar_regras(dados: dict) -> None:
    """D1 e D2 conferidos no carregamento: o app não pode exibir o que não deve."""
    vazou = [c for c in CAMPOS_SO_PDF if c in dados["documentos"].columns]
    if vazou:
        raise ErroDados(
            f"D1: {', '.join(vazou)} está em documentos.parquet e só pode estar em "
            f"pdf_extra.parquet. {RECONSTRUIR}"
        )
    pessoas = dados["pessoas"]
    if "nascimento_exibivel" not in pessoas.columns:
        raise ErroDados(f"pessoas.parquet não tem a coluna nascimento_exibivel. {RECONSTRUIR}")
    escondidas = pessoas.loc[~pessoas["nascimento_exibivel"].astype(bool)]
    with_dado = [c for c in ("nivel", "cidade_nascimento", "url_fonte", "ano_nascimento")
                 if c in escondidas.columns and escondidas[c].notna().any()]
    if with_dado:
        raise ErroDados(
            f"D2: pessoas com nascimento não exibível ainda trazem {', '.join(with_dado)}. "
            f"{RECONSTRUIR}"
        )


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_sem_cache(pasta: Path | str = APP_DADOS) -> Dados:
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise ErroDados(f"A pasta de artefatos não existe: {pasta}. {RECONSTRUIR}")
    manifesto = ler_manifesto(pasta)
    verificar_hashes(pasta, manifesto)

    dados = {nome.split(".")[0]: pd.read_parquet(pasta / nome) for nome in ARTEFATOS_PARQUET}
    dados["embeddings"] = np.load(pasta / "embeddings.f16.npy", mmap_mode="r")
    with open(pasta / "lexico.pkl", "rb") as f:
        dados["lexico"] = pickle.load(f)

    verificar_alinhamento(dados, manifesto)
    verificar_regras(dados)

    return Dados(
        documentos=dados["documentos"], pessoas=dados["pessoas"], doc_pessoa=dados["doc_pessoa"],
        pdf_extra=dados["pdf_extra"], embeddings=dados["embeddings"], lexico=dados["lexico"],
        manifesto=manifesto, pasta=pasta,
        posicao={uid: i for i, uid in enumerate(dados["documentos"]["doc_uid"].astype(str))},
    )


@lru_cache(maxsize=4)
def _carregar_cacheado(pasta: str) -> Dados:
    return carregar_sem_cache(Path(pasta))


_cacheado_streamlit = None


def carregar(pasta: Path | str = APP_DADOS) -> Dados:
    """Carrega os artefatos uma vez por processo.

    Dentro do Streamlit usa `st.cache_resource`, para que as tabelas e os vetores sejam
    carregados uma vez e nunca copiados (spec §4, C8). Fora dele, um cache simples.
    """
    global _cacheado_streamlit
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return _carregar_cacheado(str(pasta))
    if _cacheado_streamlit is None:
        # Embrulhado uma única vez: embrulhar a cada chamada criaria um cache novo a cada vez.
        _cacheado_streamlit = st.cache_resource(
            show_spinner="Carregando o corpus..."
        )(_carregar_cacheado)
    return _cacheado_streamlit(str(pasta))
