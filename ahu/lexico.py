"""lexico.py — o índice léxico da §5.2 em disco: BM25 por campo sem carregar o corpus na memória.

Com 167 mil documentos o `lexico.pkl` (um `BM25Okapi` por campo, com um dicionário de
frequências por documento) não cabe na memória do Streamlit Community Cloud. Este módulo guarda
o mesmo índice em arquivos lidos por `mmap`, e a nota de cada documento é **a mesma** que o
`rank_bm25` daria (k1 = 1,5, b = 0,75, IDF do Lucene, token-sentinela nos documentos vazios):

    app_dados/lexico/
        meta.json                 versão, campos, pesos, k1, b, avgdl por campo
        lexico.sqlite             vocabulário (termo -> coluna, idf) e os textos normalizados
        <campo>.ptr.npy           int64[V+1]: onde começa a lista de cada termo
        <campo>.doc.npy           int32[P]:   linhas que têm o termo, em ordem
        <campo>.tf.npy            uint16[P]:  quantas vezes o termo aparece na linha
        <campo>.len.npy           float32[N]: tamanho de cada linha, em tokens

Uma consulta lê só as listas dos termos dela. A frase entre aspas (§5.5) é procurada nos textos
normalizados do SQLite, que ficam no disco.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

K1 = 1.5
B = 0.75
SENTINELA = "\x00"
PASTA = "lexico"
VERSAO = "2"


def _sufixos(campo: str) -> list[str]:
    return [f"{campo}.{s}.npy" for s in ("ptr", "doc", "tf", "len")]


def arquivos(campos: Iterable[str]) -> list[str]:
    """Os arquivos do índice, relativos a `app_dados/` (o manifesto guarda o hash de cada um)."""
    nomes = ["meta.json", "lexico.sqlite"]
    for campo in campos:
        nomes += _sufixos(campo)
    return [f"{PASTA}/{n}" for n in nomes]


# ---------------------------------------------------------------------------
# Construção (só no build)
# ---------------------------------------------------------------------------

def gravar(pasta: Path, tokens_por_campo: dict[str, list[list[str]]], pesos: dict[str, float],
           documentos_sem_token: dict[str, int]) -> dict:
    """Grava o índice de `tokens_por_campo` (uma lista de tokens por linha, em cada campo)."""
    pasta = Path(pasta) / PASTA
    pasta.mkdir(parents=True, exist_ok=True)
    banco = pasta / "lexico.sqlite"
    banco.unlink(missing_ok=True)
    con = sqlite3.connect(banco)
    con.execute("CREATE TABLE texto (linha INTEGER, campo TEXT, norm TEXT)")
    avgdl = {}
    n = 0
    for campo, linhas in tokens_por_campo.items():
        n = len(linhas)
        postings: dict[str, list[tuple[int, int]]] = {}
        tamanhos = np.empty(n, dtype=np.float32)
        for i, tokens in enumerate(linhas):
            tokens = tokens or [SENTINELA]
            tamanhos[i] = len(tokens)
            contagem: dict[str, int] = {}
            for t in tokens:
                contagem[t] = contagem.get(t, 0) + 1
            for t, tf in contagem.items():
                postings.setdefault(t, []).append((i, tf))
        avgdl[campo] = float(tamanhos.sum()) / n if n else 0.0
        termos = sorted(postings)
        ptr = np.zeros(len(termos) + 1, dtype=np.int64)
        for k, t in enumerate(termos):
            ptr[k + 1] = ptr[k] + len(postings[t])
        doc = np.empty(int(ptr[-1]), dtype=np.int32)
        tf = np.empty(int(ptr[-1]), dtype=np.uint16)
        for k, t in enumerate(termos):
            a, b = ptr[k], ptr[k + 1]
            lista = postings[t]
            doc[a:b] = [d for d, _ in lista]
            tf[a:b] = [min(f, 65535) for _, f in lista]
        for sufixo, arr in zip(("ptr", "doc", "tf", "len"), (ptr, doc, tf, tamanhos)):
            np.save(pasta / f"{campo}.{sufixo}.npy", arr)
        # IDF do Lucene, sempre positivo (ver build/construir_app.py, _corrigir_idf).
        con.execute(f"CREATE TABLE vocab_{campo} (termo TEXT PRIMARY KEY, col INTEGER, idf REAL)"
                    " WITHOUT ROWID")
        con.executemany(
            f"INSERT INTO vocab_{campo} VALUES (?, ?, ?)",
            ((t, k, math.log(1 + (n - (ptr[k + 1] - ptr[k]) + 0.5) / ((ptr[k + 1] - ptr[k]) + 0.5)))
             for k, t in enumerate(termos)))
        con.executemany("INSERT INTO texto VALUES (?, ?, ?)",
                        ((i, campo, " ".join(t)) for i, t in enumerate(linhas) if t))
    con.commit()
    con.execute("VACUUM")
    con.close()
    meta = {"versao": VERSAO, "campos": list(tokens_por_campo), "pesos": dict(pesos),
            "k1": K1, "b": B, "documentos": n, "avgdl": avgdl,
            "documentos_sem_token": documentos_sem_token}
    (pasta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    return meta


# ---------------------------------------------------------------------------
# Consulta (no app)
# ---------------------------------------------------------------------------

class Lexico:
    """O índice aberto para leitura. Nada do corpus é copiado para a memória."""

    def __init__(self, pasta: Path, doc_uids: list[str]):
        self.pasta = Path(pasta) / PASTA
        self.meta = json.loads((self.pasta / "meta.json").read_text("utf-8"))
        self.campos: list[str] = self.meta["campos"]
        self.pesos: dict[str, float] = self.meta["pesos"]
        self.doc_uids = doc_uids
        self._arr = {
            campo: {s: np.load(self.pasta / f"{campo}.{s}.npy", mmap_mode="r")
                    for s in ("ptr", "doc", "tf", "len")}
            for campo in self.campos
        }
        self._uri = f"{(self.pasta / 'lexico.sqlite').resolve().as_uri()}?mode=ro&immutable=1"

    def __len__(self) -> int:
        return int(self.meta["documentos"])

    def _sql(self, consulta: str, parametros: Iterable = ()) -> list[tuple]:
        # Uma conexão por consulta: o Streamlit chama de várias threads, e abrir é barato.
        con = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        try:
            return con.execute(consulta, tuple(parametros)).fetchall()
        finally:
            con.close()

    def _vocab(self, campo: str, termos: list[str]) -> dict[str, tuple[int, float]]:
        if not termos:
            return {}
        marcas = ",".join("?" * len(termos))
        return {t: (c, idf) for t, c, idf in self._sql(
            f"SELECT termo, col, idf FROM vocab_{campo} WHERE termo IN ({marcas})", termos)}

    def _lista(self, campo: str, coluna: int) -> tuple[np.ndarray, np.ndarray]:
        arr = self._arr[campo]
        a, b = int(arr["ptr"][coluna]), int(arr["ptr"][coluna + 1])
        return np.asarray(arr["doc"][a:b]), np.asarray(arr["tf"][a:b], dtype=np.float64)

    def notas_do_campo(self, campo: str, tokens: list[str]) -> np.ndarray:
        """O `get_scores` do `BM25Okapi` (termo repetido na consulta conta de novo)."""
        notas = np.zeros(len(self), dtype=np.float64)
        vocab = self._vocab(campo, list(dict.fromkeys(tokens)))
        tamanhos = self._arr[campo]["len"]
        avgdl = self.meta["avgdl"][campo] or 1.0
        for t in tokens:
            if t not in vocab:
                continue
            coluna, idf = vocab[t]
            docs, tf = self._lista(campo, coluna)
            norma = K1 * (1 - B + B * np.asarray(tamanhos[docs], dtype=np.float64) / avgdl)
            notas[docs] += idf * (tf * (K1 + 1) / (tf + norma))
        return notas

    def termos_existentes(self, tokens: Iterable[str]) -> list[str]:
        tokens = list(dict.fromkeys(tokens))
        achados: set[str] = set()
        for campo in self.campos:
            achados |= set(self._vocab(campo, tokens))
        return [t for t in tokens if t in achados]

    def linhas_com_termo(self, termo: str) -> np.ndarray:
        """As linhas em que o termo aparece em algum campo (a evidência de termos da busca)."""
        partes = [self._lista(campo, col)[0]
                  for campo in self.campos
                  for col, _ in self._vocab(campo, [termo]).values()]
        return np.unique(np.concatenate(partes)) if partes else np.zeros(0, dtype=np.int32)

    def mascara_de_frase(self, frase: str) -> np.ndarray:
        """Quem tem a sequência literal `frase` em algum campo normalizado."""
        mascara = np.zeros(len(self), dtype=bool)
        linhas = [r[0] for r in self._sql(
            "SELECT DISTINCT linha FROM texto WHERE instr(norm, ?) > 0", (frase,))]
        mascara[linhas] = True
        return mascara

