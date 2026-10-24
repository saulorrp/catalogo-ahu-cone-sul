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
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ahu.lexico import Lexico

RAIZ = Path(__file__).resolve().parent.parent
APP_DADOS = RAIZ / "app_dados"

ARTEFATOS_PARQUET = ("documentos.parquet", "pessoas.parquet", "doc_pessoa.parquet",
                     "pdf_extra.parquet")
# Os vetores vêm em partes (`embeddings/parte_000.f16.npy`, ...) e o léxico numa pasta: o
# manifesto lista cada arquivo. Estes são os que não podem faltar.
ARTEFATOS = (*ARTEFATOS_PARQUET, "lexico/meta.json")
PASTA_EMBEDDINGS = "embeddings"

# D1: nunca na interface, nos filtros, no CSV nem no JSON. Só no dossiê em PDF.
CAMPOS_SO_PDF = ("scribe_mediation_likely",)

RECONSTRUIR = "Rode `python build/construir_app.py` para reconstruir os artefatos."


class Vetores:
    """Os vetores dos documentos, em partes lidas por `mmap`: nada vai inteiro para a memória.

    Uma parte só tem menos de 100 MB (o limite de arquivo do GitHub). O cosseno é calculado
    parte por parte, em blocos, para não converter a matriz inteira para float32 de uma vez.
    """

    BLOCO = 20_000

    def __init__(self, partes: list[np.ndarray]):
        self.partes = partes
        dimensao = partes[0].shape[1] if partes and partes[0].ndim == 2 else 0
        self.shape = (sum(p.shape[0] for p in partes), dimensao)
        self.ndim = 2

    def __len__(self) -> int:
        return self.shape[0]

    def produto(self, vetor: np.ndarray) -> np.ndarray:
        saida = np.empty(len(self), dtype=np.float32)
        inicio = 0
        for parte in self.partes:
            for a in range(0, parte.shape[0], self.BLOCO):
                bloco = np.asarray(parte[a:a + self.BLOCO], dtype=np.float32)
                saida[inicio + a: inicio + a + len(bloco)] = bloco @ vetor
            inicio += parte.shape[0]
        return saida


class ErroDados(RuntimeError):
    """Artefatos ausentes, desalinhados ou com hash diferente do manifesto (A4)."""


@dataclass(frozen=True, eq=False)
class Dados:
    """Os artefatos do §3.4, já validados. As tabelas não são copiadas."""

    documentos: pd.DataFrame
    pessoas: pd.DataFrame
    doc_pessoa: pd.DataFrame
    pdf_extra: pd.DataFrame
    embeddings: Vetores
    lexico: Lexico
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
        if nome not in esperados:
            raise ErroDados(f"O manifesto não registra o artefato {nome}. {RECONSTRUIR}")
    for nome, esperado in esperados.items():
        caminho = pasta / nome
        if not caminho.exists():
            raise ErroDados(f"Artefato ausente: {caminho}. {RECONSTRUIR}")
        obtido = _sha256_arquivo(caminho)
        if obtido != esperado:
            raise ErroDados(
                f"{nome} mudou depois da construção: o manifesto espera "
                f"{esperado[:12]}... e o arquivo tem {obtido[:12]}.... {RECONSTRUIR}"
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

    if len(lexico) != len(uids):
        raise ErroDados(f"O léxico tem {len(lexico)} linhas e são {len(uids)} documentos. "
                        f"{RECONSTRUIR}")

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
    partes = sorted(n for n in manifesto["artefatos"] if n.startswith(f"{PASTA_EMBEDDINGS}/"))
    dados["embeddings"] = Vetores([np.load(pasta / n, mmap_mode="r") for n in partes])
    dados["lexico"] = Lexico(pasta, dados["documentos"]["doc_uid"].astype(str).tolist())

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


def _carregar_streamlit(pasta: str, versao_codigo: str) -> Dados:
    # `versao_codigo` só entra na chave do cache: ver `_versao_do_codigo`.
    return carregar_sem_cache(Path(pasta))


# Os módulos cujas classes ficam dentro do objeto em cache (o `Lexico` vive em `Dados`).
_MODULOS_EM_CACHE = ("dados.py", "lexico.py")


def _versao_do_codigo() -> str:
    """Hash do código das classes guardadas no cache.

    O Streamlit recarrega os módulos alterados a cada atualização do repositório, mas o
    `st.cache_resource` devolve o objeto antigo, que ainda é da classe antiga: depois do PR #13
    o `Lexico` em cache não tinha `idf_maximo` e a busca quebrou (24/10/2026). Com o hash do
    código na chave, código novo carrega os dados de novo. É lido a cada chamada, e não na
    importação, porque mudar só o `lexico.py` não recarrega este módulo.
    """
    pasta = Path(__file__).parent
    h = hashlib.sha1()
    for nome in _MODULOS_EM_CACHE:
        h.update((pasta / nome).read_bytes())
    return h.hexdigest()


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
        # max_entries=1: a versão antiga sai da memória quando a nova entra (teto de 690 MiB).
        _cacheado_streamlit = st.cache_resource(
            show_spinner="Carregando o corpus...", max_entries=1
        )(_carregar_streamlit)
    return _cacheado_streamlit(str(pasta), _versao_do_codigo())
