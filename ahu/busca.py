"""busca.py — WP 2.1: rota de cotas, BM25 por campos, busca semântica e fusão RRF (spec §5.2–§5.5).

Sem Streamlit: tudo aqui é função pura sobre os artefatos de `ahu/dados.py`.

Três motores, nesta ordem:
  1. **Cotas (§5.3).** Uma expressão regular reconhece as três famílias de cota do catálogo. Se a
     consulta *inteira* é uma cota, a resposta é a tabela de lookup e mais nada — nunca um
     `str.contains` solto, que era o C5. Se a cota está dentro de uma frase, esses documentos vêm
     primeiro e o resto da frase segue a busca normal.
  2. **Léxico (§5.2).** BM25 por campo, com os pesos do `lexico.pkl`: pessoas 2,0 · destinatário 1,5
     · tipologia 1,5 · descrição 1,0 · pasta 0,5.
  3. **Semântico (§5.7).** Cosseno entre a consulta (prefixo `query: `) e os vetores normalizados.
     O `rigor` aplica um limiar ao cosseno **bruto**.

A fusão é por **posição**, não por nota normalizada (§5.4, C4):

    nota(d) = Σ_motor  peso_motor / (60 + posição_motor(d))

Cada motor entrega no máximo `LIMITE_MOTOR` documentos, e é isso que impede uma consulta sem
sentido de devolver o corpus inteiro.

Os limiares de rigor e os pesos do RRF são **provisórios** até o WP 4.1–4.2 medirem no conjunto de
ouro. Estão todos neste módulo, em constantes, para que a calibração mexa num lugar só.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ahu.texto import tokenizar

MODOS = ("automatico", "exata", "conceitual")
RIGORES = ("baixo", "medio", "alto")

LIMITE_MOTOR = 300          # §5.4: cada motor contribui só com o seu topo
K_RRF = 60                  # §5.4: a constante da fórmula

# Pesos do RRF. Consulta curta (até 2 palavras) pesa mais o léxico; consulta longa, o semântico
# (§5.4). Provisórios até o WP 4.2.
PALAVRAS_CONSULTA_CURTA = 2
PESOS_RRF = {
    "curta": {"lexico": 1.0, "semantico": 0.4},
    "longa": {"lexico": 0.6, "semantico": 1.0},
}

# Rigor (§5.5): quanto o cosseno bruto pode ficar **abaixo do melhor cosseno daquela consulta**.
#
# A §5.5 pede um limiar absoluto "em escala fixa". Medido no corpus, um limiar absoluto não
# funciona, porque o e5 comprime os cossenos e desloca a faixa a cada consulta (08/10/2026,
# 7.051 documentos):
#     "deserção de soldados"  max 0,843 — 7.051 documentos acima de 0,76 e 1.225 acima de 0,80
#     "Lavradio"              max 0,779 —   135 documentos acima de 0,76 e     0 acima de 0,80
# O mesmo corte dá o corpus inteiro numa consulta e nada na outra. A margem abaixo do máximo da
# própria consulta se comporta igual nas duas, e continua sendo o cosseno bruto, sem inventar uma
# "relevância" de 0 a 1 (C4):
#     margem 0,010 -> 2 a 21 documentos | 0,020 -> 8 a 149 | 0,030 -> 19 a 540
# Valores provisórios até o WP 4.2. A §5.5 precisa ser corrigida para descrever a margem.
MARGENS_RIGOR = {"baixo": 0.035, "medio": 0.020, "alto": 0.010}

# Abaixo disto o e5 não distingue nada: vale como piso absoluto para uma consulta sem sentido.
COSSENO_MINIMO = 0.70

PREFIXO_CONSULTA_PADRAO = "query: "


# ---------------------------------------------------------------------------
# Rota de cotas (§5.3)
# ---------------------------------------------------------------------------

def _achatar(texto: str) -> str:
    """Tudo que não é letra ou número vira um espaço: as três famílias de cota ficam comparáveis.

    'PT/AHU/CU/021/0006/00390'      -> 'pt ahu cu 021 0006 00390'
    'AHU_ACL_CU_021, Cx. 6, D. 390.'-> 'ahu acl cu 021 cx 6 d 390'
    'ahu acl cu 021 cx 6 d 390'     -> igual (é o que o usuário digita)
    """
    t = unicodedata.normalize("NFKD", str(texto or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# `AHU_ACL_CU_021, Cx. 6, D. 390` e as suas variações de pontuação e espaço.
_RE_CU_CX = re.compile(r"\b(?:pt )?ahu (?:acl )?cu (\d{1,4}) cx (\d{1,4}) (?:d|doc) (\d{1,6})\b")
# `PT/AHU/CU/021/0006/00390`: três números seguidos, sem as palavras cx/d.
_RE_CU_REF = re.compile(r"\b(?:pt )?ahu (?:acl )?cu (\d{1,4}) (\d{1,5}) (\d{1,6})\b")
# `AHU-Santa Catarina, cx. 8, doc. 14`: a região no lugar do número do fundo.
_RE_REGIAO = re.compile(r"\bahu ([a-z]+(?: [a-z]+){0,3}?) cx (\d{1,4}) (?:d|doc) (\d{1,6})\b")

_MOTIVOS = (("cu_cx", _RE_CU_CX), ("cu_ref", _RE_CU_REF), ("regiao", _RE_REGIAO))


def chave_cota(texto: str) -> str | None:
    """Chave canônica de uma cota, ou None se o texto não contiver uma.

    As três famílias caem na mesma chave quando são o mesmo documento: zeros à esquerda somem.
    """
    achatado = _achatar(texto)
    for motivo, padrao in _MOTIVOS:
        m = padrao.search(achatado)
        if not m:
            continue
        if motivo == "regiao":
            regiao, cx, doc = m.group(1).strip(), int(m.group(2)), int(m.group(3))
            # 'ahu acl cu' já foi tratado acima; aqui a "região" nunca é a palavra 'cu'.
            if regiao in {"cu", "acl cu", "acl"}:
                continue
            return f"reg|{regiao}|{cx}|{doc}"
        return f"cu|{int(m.group(1))}|{int(m.group(2))}|{int(m.group(3))}"
    return None


def _trecho_da_cota(texto: str) -> tuple[str | None, str]:
    """Devolve (chave, resto do texto sem a cota). O resto segue a busca normal (§5.3)."""
    achatado = _achatar(texto)
    for motivo, padrao in _MOTIVOS:
        m = padrao.search(achatado)
        if not m:
            continue
        if motivo == "regiao" and m.group(1).strip() in {"cu", "acl cu", "acl"}:
            continue
        chave = chave_cota(texto)
        resto = re.sub(r"\s+", " ", achatado[: m.start()] + " " + achatado[m.end():]).strip()
        return chave, resto
    return None, achatado


def tabela_de_cotas(dados) -> dict[str, list[int]]:
    """chave de cota -> linhas de `documentos.parquet`. Lida das três colunas de cota.

    Construída uma vez e guardada no próprio `Dados` (é derivada, não um artefato do §3.4).
    """
    cache = getattr(dados, "_tabela_cotas", None)
    if cache is not None:
        return cache
    tabela: dict[str, list[int]] = {}
    colunas = [c for c in ("reference_code", "new_code", "old_code") if c in dados.documentos.columns]
    for coluna in colunas:
        for i, valor in enumerate(dados.documentos[coluna]):
            if not valor or valor is None:
                continue
            chave = chave_cota(valor)
            if chave is None:
                continue
            linhas = tabela.setdefault(chave, [])
            if i not in linhas:
                linhas.append(i)
    object.__setattr__(dados, "_tabela_cotas", tabela)
    return tabela


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class Resultado:
    """Uma linha da resposta. A nota exibida é a **posição**, não uma relevância 0–1 (C4)."""

    doc_uid: str
    linha: int
    posicao: int = 0
    nota_rrf: float = 0.0
    posicoes: dict[str, int] = field(default_factory=dict)   # motor -> posição naquele motor
    cosseno: float | None = None
    termos: list[str] = field(default_factory=list)

    @property
    def motores(self) -> list[str]:
        return sorted(self.posicoes)


@dataclass
class Resposta:
    resultados: list[Resultado]
    total: int
    modo: str
    rigor: str
    cota: str | None = None
    pesos: dict[str, float] = field(default_factory=dict)
    termos_consulta: list[str] = field(default_factory=list)
    truncado_por_motor: bool = False

    def pagina(self, numero: int = 1, tamanho: int = 25) -> list[Resultado]:
        """§5.5: 25 por página, com o total real em `total`."""
        inicio = max(0, (numero - 1)) * tamanho
        return self.resultados[inicio: inicio + tamanho]


# ---------------------------------------------------------------------------
# Motor léxico (§5.2)
# ---------------------------------------------------------------------------

def _notas_lexico(dados, tokens: list[str]) -> np.ndarray:
    lexico = dados.lexico
    pesos = lexico["pesos"]
    total = np.zeros(len(lexico["doc_uids"]), dtype=np.float64)
    for campo in lexico["campos"]:
        peso = float(pesos.get(campo, 0.0))
        if not peso:
            continue
        notas = np.asarray(lexico["indices"][campo].get_scores(tokens), dtype=np.float64)
        maximo = notas.max()
        if maximo > 0:
            # Min-max por campo antes de somar: sem isso o campo com mais texto manda sempre,
            # e o peso da §5.2 não significaria nada.
            notas = notas / maximo
        total += peso * notas
    return total


def _frases_exigidas(texto: str) -> list[str]:
    """As sequências entre aspas, já normalizadas (§5.5)."""
    return [f for f in (" ".join(tokenizar(bruto)) for bruto in re.findall(r'"([^"]+)"', texto)) if f]


def _mascara_de_frases(dados, frases: list[str]) -> np.ndarray | None:
    """Quem contém todas as frases exigidas, na sequência literal."""
    if not frases:
        return None
    normalizados = dados.lexico.get("normalizados") or {}
    if not normalizados:
        return None
    n = len(dados.lexico["doc_uids"])
    mascara = np.ones(n, dtype=bool)
    for frase in frases:
        tem = np.zeros(n, dtype=bool)
        for campo, textos in normalizados.items():
            del campo
            for i, texto in enumerate(textos):
                if not tem[i] and frase in texto:
                    tem[i] = True
        mascara &= tem
    return mascara


# ---------------------------------------------------------------------------
# Motor semântico (§5.7)
# ---------------------------------------------------------------------------

Encoder = Callable[[list[str]], np.ndarray]


def criar_encoder_consulta(dados, dispositivo: str | None = None) -> Encoder:
    """Codificador da consulta, com o modelo e o prefixo que o manifesto registra."""
    from sentence_transformers import SentenceTransformer  # import tardio: pesado

    info = dados.modelo_embeddings
    modelo = SentenceTransformer(info["nome"], device=dispositivo)
    prefixo = info.get("prefixo_consulta") or PREFIXO_CONSULTA_PADRAO

    def encoder(textos: list[str]) -> np.ndarray:
        return modelo.encode([prefixo + t for t in textos], normalize_embeddings=True,
                             convert_to_numpy=True)

    return encoder


def cossenos(dados, texto: str, encoder: Encoder) -> np.ndarray:
    """Cosseno bruto entre a consulta e cada documento (os vetores já são L2-normalizados)."""
    vetor = np.asarray(encoder([texto]), dtype=np.float32)
    if vetor.ndim == 2:
        vetor = vetor[0]
    norma = float(np.linalg.norm(vetor))
    if norma and abs(norma - 1.0) > 1e-3:
        vetor = vetor / norma
    return np.asarray(dados.embeddings, dtype=np.float32) @ vetor


# ---------------------------------------------------------------------------
# Fusão (§5.4)
# ---------------------------------------------------------------------------

def _topo(notas: np.ndarray, limite: int, minimo: float = 0.0,
          permitidos: np.ndarray | None = None) -> list[int]:
    """As `limite` melhores linhas, só as que passam de `minimo` (e da máscara, se houver)."""
    validas = notas > minimo
    if permitidos is not None:
        validas &= permitidos
    indices = np.flatnonzero(validas)
    if indices.size == 0:
        return []
    ordem = np.argsort(-notas[indices], kind="stable")[:limite]
    return indices[ordem].tolist()


def pesos_do_rrf(tokens: Iterable[str]) -> dict[str, float]:
    n = len(list(tokens))
    return dict(PESOS_RRF["curta" if n <= PALAVRAS_CONSULTA_CURTA else "longa"])


def _fundir(por_motor: dict[str, list[int]], pesos: dict[str, float]) -> dict[int, Resultado]:
    fundidos: dict[int, Resultado] = {}
    for motor, linhas in por_motor.items():
        peso = float(pesos.get(motor, 1.0))
        for posicao, linha in enumerate(linhas, start=1):
            r = fundidos.get(linha)
            if r is None:
                r = fundidos[linha] = Resultado(doc_uid="", linha=linha)
            r.posicoes[motor] = posicao
            r.nota_rrf += peso / (K_RRF + posicao)
    return fundidos


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def buscar(
    dados, texto: str, modo: str = "automatico", rigor: str = "medio",
    encoder: Encoder | None = None, limite_motor: int = LIMITE_MOTOR,
    cosseno_minimo: float = COSSENO_MINIMO,
) -> Resposta:
    """A busca da §5.2–§5.5. `encoder=None` desliga o motor semântico (busca só léxica).

    `cosseno_minimo` é o piso absoluto do motor semântico, calibrado para o e5. Outro modelo tem
    outra escala de cosseno: quem troca o modelo (WP 4.2) passa o piso dele aqui.
    """
    if modo not in MODOS:
        raise ValueError(f"Modo desconhecido: {modo!r}. Use um de {', '.join(MODOS)}.")
    if rigor not in RIGORES:
        raise ValueError(f"Rigor desconhecido: {rigor!r}. Use um de {', '.join(RIGORES)}.")

    uids = dados.lexico["doc_uids"]
    chave, resto = _trecho_da_cota(texto)
    linhas_da_cota = tabela_de_cotas(dados).get(chave, []) if chave else []
    frases = _frases_exigidas(texto)
    tokens = tokenizar(resto if chave else texto)

    # §5.3: a consulta inteira é uma cota -> só a tabela de lookup, nada de busca textual.
    if chave and not tokens:
        resultados = [Resultado(doc_uid=uids[i], linha=i, posicao=p, nota_rrf=1.0,
                                posicoes={"cota": p})
                      for p, i in enumerate(linhas_da_cota, start=1)]
        return Resposta(resultados=resultados, total=len(resultados), modo=modo, rigor=rigor,
                        cota=chave, termos_consulta=[])

    por_motor: dict[str, list[int]] = {}
    mascara = _mascara_de_frases(dados, frases)
    cos = None

    if tokens and modo in ("automatico", "exata"):
        notas = _notas_lexico(dados, tokens)
        por_motor["lexico"] = _topo(notas, limite_motor, permitidos=mascara)

    if tokens and modo in ("automatico", "conceitual") and encoder is not None:
        cos = cossenos(dados, texto if not chave else resto, encoder)
        limiar = max(cosseno_minimo, float(cos.max()) - MARGENS_RIGOR[rigor])
        por_motor["semantico"] = _topo(cos, limite_motor, minimo=limiar, permitidos=mascara)

    pesos = pesos_do_rrf(tokens)
    fundidos = _fundir(por_motor, pesos)

    # §5.3: a cota dentro da frase -> esses documentos primeiro, o resto da frase segue a busca.
    ordem = sorted(fundidos.values(), key=lambda r: (-r.nota_rrf, r.linha))
    if linhas_da_cota:
        da_cota = []
        for p, linha in enumerate(linhas_da_cota, start=1):
            r = fundidos.pop(linha, None) or Resultado(doc_uid="", linha=linha)
            r.posicoes["cota"] = p
            da_cota.append(r)
        ordem = da_cota + [r for r in ordem if r.linha in fundidos]

    normalizados = dados.lexico.get("normalizados") or {}
    conjunto = set(tokens)
    for posicao, r in enumerate(ordem, start=1):
        r.doc_uid = uids[r.linha]
        r.posicao = posicao
        if cos is not None:
            r.cosseno = round(float(cos[r.linha]), 4)
        if conjunto:
            achados = set()
            for textos in normalizados.values():
                achados |= conjunto & set(textos[r.linha].split())
            r.termos = sorted(achados)

    return Resposta(
        resultados=ordem, total=len(ordem), modo=modo, rigor=rigor, cota=chave, pesos=pesos,
        termos_consulta=tokens,
        truncado_por_motor=any(len(v) >= limite_motor for v in por_motor.values()),
    )


