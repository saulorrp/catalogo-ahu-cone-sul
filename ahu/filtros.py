"""filtros.py — WP 2.2: a tabela de filtros da §6.1, como máscara sobre `documentos.parquet`.

Sem Streamlit. Dois lados:

* **`Filtros`**: o estado dos controles, que o `ahu/consulta.py` serializa para a URL e para o
  bloco de proveniência. `mascara(dados)` devolve o vetor booleano que a busca recebe.
* **`opcoes(dados)`**: as listas que a interface oferece. Saem **dos dados** — coleções, pastas,
  tipologias, categorias, vetores, países e cidades (D6, AGENTS.md 7). Nenhuma delas está escrita
  no código, e por isso uma coleção nova aparece nos filtros sem mexer no app (A11).

Padrões (§6.1, preprint §4): **todo filtro que depende do DeepSeek começa desligado** — categoria
social, vetor, impacto e SRSP. (As lentes metodológicas da §6.1 foram retiradas pela **D9**: o
autor não as usava, e um filtro preenchido por uma lente escondia de onde ele tinha vindo.)

D2 em dois lugares: o filtro de origem do remetente só alcança quem tem nascimento exibível, e a
caixa "só confiança ALTA" aperta ainda mais (desmarcada = ALTA + MÉDIA).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable

import numpy as np
import pandas as pd

from ahu.texto import normalizar_busca

# Faixas do slider do SRSP (§6.2). O intervalo em si é um dado da camada, lido do manifesto.
FAIXAS_SRSP = {"baixo": (1, 3), "medio": (4, 6), "alto": (7, 9)}
NOTA_SRSP = ("SRSP do DeepSeek v3, auditado no preprint §2.1: MAE 1,13 · r = 0,48 · viés +0,59 · "
             "N = 705 · um avaliador. Diferenças de ±1 ponto não são distinguíveis.")

# Filtros que vêm de inferência de IA: começam desligados (preprint §4).
CAMPOS_DEEPSEEK = ("categorias", "vetores", "impactos", "srsp_de", "srsp_ate")


@dataclass
class Filtros:
    """O estado dos controles da §6.1. Vazio = filtro desligado."""

    colecoes: list[str] = field(default_factory=list)
    ano_de: int | None = None
    ano_ate: int | None = None
    sem_data: bool = True          # "incluir documentos sem data" (186 documentos)
    pastas: list[str] = field(default_factory=list)
    tipologias: list[str] = field(default_factory=list)
    pessoas: list[str] = field(default_factory=list)      # pessoa_uid
    destinatario: str | None = None                       # texto, autocompletar (§6.1)
    # --- daqui para baixo, inferência do DeepSeek: desligados por padrão ---
    categorias: list[str] = field(default_factory=list)
    vetores: list[str] = field(default_factory=list)
    impactos: list[str] = field(default_factory=list)
    srsp_de: int | None = None
    srsp_ate: int | None = None
    # --- origem do remetente (camada de biografias, D2) ---
    paises_origem: list[str] = field(default_factory=list)
    cidades_origem: list[str] = field(default_factory=list)
    so_confianca_alta: bool = False

    # -----------------------------------------------------------------
    def vazio(self) -> bool:
        padrao = Filtros()
        return all(getattr(self, f.name) == getattr(padrao, f.name) for f in fields(self))

    def usa_deepseek(self) -> bool:
        padrao = Filtros()
        return any(getattr(self, c) != getattr(padrao, c) for c in CAMPOS_DEEPSEEK)

    def ativos(self) -> dict:
        """Só o que está diferente do padrão — é o que vai para a URL e para a proveniência."""
        padrao = Filtros()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(padrao, f.name)}

    # -----------------------------------------------------------------
    def mascara(self, dados) -> np.ndarray | None:
        """Vetor booleano sobre `documentos.parquet`, ou None se nada estiver filtrado."""
        if self.vazio():
            return None
        docs = dados.documentos
        m = np.ones(len(docs), dtype=bool)

        m &= _em(docs["colecao"], self.colecoes)
        m &= _em(docs.get("folder"), self.pastas)
        m &= _em(docs.get("tipologia"), self.tipologias)
        m &= _em(docs.get("sender_category"), self.categorias)
        m &= _em(docs.get("vector"), self.vetores)
        m &= _em(docs.get("typology_impact"), self.impactos)

        if self.ano_de is not None or self.ano_ate is not None or not self.sem_data:
            anos = pd.to_numeric(docs["ano_isolado"], errors="coerce")
            dentro = np.ones(len(docs), dtype=bool)
            if self.ano_de is not None:
                dentro &= (anos >= self.ano_de).fillna(False).to_numpy()
            if self.ano_ate is not None:
                dentro &= (anos <= self.ano_ate).fillna(False).to_numpy()
            sem_ano = anos.isna().to_numpy()
            # "incluir documentos sem data" salva quem não tem ano de ser cortado pelo período.
            m &= (dentro & ~sem_ano) | (sem_ano & self.sem_data)

        if self.srsp_de is not None or self.srsp_ate is not None:
            srsp = pd.to_numeric(docs["vernacular_score"], errors="coerce")
            if self.srsp_de is not None:
                m &= (srsp >= self.srsp_de).fillna(False).to_numpy()
            if self.srsp_ate is not None:
                m &= (srsp <= self.srsp_ate).fillna(False).to_numpy()

        if self.destinatario:
            alvo = normalizar_busca(self.destinatario)
            coluna = docs["recipient_name"].fillna("").map(normalizar_busca)
            m &= coluna.str.contains(alvo, regex=False).to_numpy()

        if self.pessoas:
            m &= _documentos_das_pessoas(dados, self.pessoas)

        if self.paises_origem or self.cidades_origem or self.so_confianca_alta:
            m &= _documentos_por_origem(dados, self.paises_origem, self.cidades_origem,
                                        self.so_confianca_alta)
        return m


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _em(coluna, valores: Iterable[str]) -> np.ndarray | bool:
    valores = [v for v in (valores or []) if v]
    if coluna is None or not valores:
        return True
    return coluna.isin(valores).to_numpy()


def _linhas_de(dados, doc_uids: Iterable[str]) -> np.ndarray:
    m = np.zeros(len(dados.documentos), dtype=bool)
    for uid in doc_uids:
        i = dados.posicao.get(uid)
        if i is not None:
            m[i] = True
    return m


def _documentos_das_pessoas(dados, pessoa_uids: list[str]) -> np.ndarray:
    vinculos = dados.doc_pessoa
    escolhidas = vinculos["pessoa_uid"].isin([u for u in pessoa_uids if u])
    return _linhas_de(dados, vinculos.loc[escolhidas, "doc_uid"].unique())


def _documentos_por_origem(dados, paises: list[str], cidades: list[str],
                           so_alta: bool) -> np.ndarray:
    """Documentos cujo remetente nasceu onde se pede. Só pessoas com nascimento exibível (D2)."""
    pessoas = dados.pessoas
    alcance = pessoas["nascimento_exibivel"].astype(bool)
    if so_alta:
        alcance &= pessoas["nivel"].eq("ALTA")
    if paises:
        alcance &= pessoas["pais_atual"].isin(paises)
    if cidades:
        alcance &= pessoas["cidade_nascimento"].isin(cidades)
    uids = set(pessoas.loc[alcance, "pessoa_uid"])
    vinculos = dados.doc_pessoa
    return _linhas_de(dados, vinculos.loc[vinculos["pessoa_uid"].isin(uids), "doc_uid"].unique())


# ---------------------------------------------------------------------------
# Opções para a interface — sempre dos dados (D6, A11)
# ---------------------------------------------------------------------------

def _contagem(coluna) -> list[tuple[str, int]]:
    if coluna is None:
        return []
    return [(str(k), int(v)) for k, v in coluna.dropna().value_counts().items()]


def opcoes(dados) -> dict:
    """O que cada controle da §6.1 oferece, com contagem onde a §6.1 pede."""
    docs = dados.documentos
    pessoas = dados.pessoas
    exibiveis = pessoas[pessoas["nascimento_exibivel"].astype(bool)]
    anos = pd.to_numeric(docs["ano_isolado"], errors="coerce")
    return {
        "colecoes": sorted(docs["colecao"].dropna().unique().tolist()),
        # §6.1: o controle de coleção fica oculto enquanto houver só uma.
        "mostrar_colecao": int(docs["colecao"].nunique()) > 1,
        "ano_minimo": None if anos.dropna().empty else int(anos.min()),
        "ano_maximo": None if anos.dropna().empty else int(anos.max()),
        "documentos_sem_data": int(anos.isna().sum()),
        "pastas": sorted(docs["folder"].dropna().unique().tolist()),
        # §6.1: tipologia com contagem, por frequência.
        "tipologias": _contagem(docs.get("tipologia")),
        "categorias": _contagem(docs.get("sender_category")),
        "vetores": _contagem(docs.get("vector")),
        "impactos": _contagem(docs.get("typology_impact")),
        "escala_srsp": escala_srsp(dados),
        "faixas_srsp": dict(FAIXAS_SRSP),
        "nota_srsp": NOTA_SRSP,
        "paises_origem": sorted(exibiveis["pais_atual"].dropna().unique().tolist()),
        "cidades_origem": sorted(exibiveis["cidade_nascimento"].dropna().unique().tolist()),
        "pessoas": _opcoes_de_pessoas(dados),
    }


def escala_srsp(dados) -> list[int]:
    """O intervalo do slider é um dado da camada (regra 9), que o `construir_app.py` copia para
    `app_dados/manifest.json`. Nada aqui tem 1 a 9 escrito."""
    escala = (dados.manifesto.get("srsp") or {}).get("escala")
    if isinstance(escala, list) and len(escala) == 2:
        return [int(escala[0]), int(escala[1])]
    srsp = pd.to_numeric(dados.documentos["vernacular_score"], errors="coerce").dropna()
    return [1, 9] if srsp.empty else [int(srsp.min()), int(srsp.max())]


def auditoria_srsp(dados) -> dict:
    """Os números do preprint §2.1 que o bloco do SRSP mostra (§6.2)."""
    return (dados.manifesto.get("srsp") or {}).get("auditoria", {})


def _opcoes_de_pessoas(dados) -> list[dict]:
    """Nome, variantes e títulos de cada pessoa, para o autocompletar (§6.1)."""
    contagem = dados.doc_pessoa["pessoa_uid"].value_counts()
    saida = []
    for uid, nome, variantes, titulos, tipo in zip(
            dados.pessoas["pessoa_uid"], dados.pessoas["nome_exibicao"],
            dados.pessoas["variantes_nome"], dados.pessoas["titulos"], dados.pessoas["tipo"]):
        saida.append({
            "pessoa_uid": uid,
            "nome": nome,
            "tipo": tipo,
            "busca": sorted({str(nome), *(_lista(variantes)), *(_lista(titulos))} - {""}),
            "n_documentos": int(contagem.get(uid, 0)),
        })
    return sorted(saida, key=lambda p: (-p["n_documentos"], str(p["nome"])))


def _lista(valor) -> list[str]:
    if valor is None or isinstance(valor, float):
        return []
    return [str(v) for v in valor if v is not None]
