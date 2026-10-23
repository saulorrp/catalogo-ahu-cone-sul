"""busca.py — WP 2.1: rota de cotas, BM25 por campos, busca semântica e fusão RRF (spec §5.2–§5.5).

Sem Streamlit: tudo aqui é função pura sobre os artefatos de `ahu/dados.py`.

Três motores, nesta ordem:
  1. **Cotas (§5.3).** Uma expressão regular reconhece as três famílias de cota do catálogo. Se a
     consulta *inteira* é uma cota, a resposta é a tabela de lookup e mais nada — nunca um
     `str.contains` solto, que era o C5. Se a cota está dentro de uma frase, esses documentos vêm
     primeiro e o resto da frase segue a busca normal.
  2. **Léxico (§5.2).** BM25 por campo (índice em disco, `ahu/lexico.py`), com os pesos de
     `app_dados/lexico/meta.json`: pessoas 2,0 · destinatário 1,5 · tipologia 1,5 · descrição 1,0
     · pasta 0,5.
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

from ahu.lexico import K1
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

# Aviso de consulta distante (§5.5). Regra decidida em 08/10, depois de medir: o aviso aparece só
# quando **as duas** coisas valem:
#   (a) nenhum termo da consulta existe no corpus (o motor léxico não tem onde casar), e
#   (b) o melhor documento não se destaca da distribuição de cossenos (`destaque_do_topo`).
#
# O piso absoluto sobre o cosseno, tentado antes, errava nos dois sentidos: "Lavradio" (0,7787,
# cinco acertos perfeitos no topo) levava o aviso e "xkq zzv plorb" (0,7871) não. A condição (a)
# resolve o primeiro caso de graça — "lavradio" está no corpus — e a (b) separa o resto.
# Continua sendo **só um aviso**: nada é cortado. O 4,5 é provisório, calibrado no WP 4.2.
LIMITE_DESTAQUE = 4.5
AVISO_DISTANTE = ("Nenhum documento muito próximo desta consulta; "
                  "os resultados podem ser pouco relevantes.")

# A consulta inteira caiu na lista curta de stopwords da §5.1: não há o que buscar, e a lista
# mostrada é o corpus que passou pelos filtros.
AVISO_SO_STOPWORDS = ("A consulta só tem palavras muito comuns, que a busca ignora. "
                      "A lista abaixo é o corpus com os filtros aplicados.")

PREFIXO_CONSULTA_PADRAO = "query: "

# Nota léxica (§5.2), ver `_notas_lexico`. Medido em 23/10/2026 com 167 mil documentos: com estes
# valores, "ordem terceira de são francisco" devolve a Ordem Terceira nos 7 primeiros (antes, o
# topo era de ORDENS régias e de pessoas chamadas Francisco). Provisórios até o WP 4.2.
EXPOENTE_COBERTURA = 2
BONUS_FRASE_INTEIRA = 2.0


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
    cosseno_maximo: float | None = None
    destaque: float | None = None
    sem_correspondencia_lexica: bool = False
    avisos: list[str] = field(default_factory=list)

    @property
    def consulta_distante(self) -> bool:
        """§5.5: nenhum termo da consulta existe no corpus **e** o topo não se destaca.

        É aviso, não corte: a lista de resultados é a mesma com ou sem ele.
        """
        return AVISO_DISTANTE in self.avisos

    def pagina(self, numero: int = 1, tamanho: int = 25) -> list[Resultado]:
        """§5.5: 25 por página, com o total real em `total`."""
        inicio = max(0, (numero - 1)) * tamanho
        return self.resultados[inicio: inicio + tamanho]


# ---------------------------------------------------------------------------
# Motor léxico (§5.2)
# ---------------------------------------------------------------------------

def _notas_lexico(dados, tokens: list[str]) -> np.ndarray:
    """A nota léxica de cada documento: BM25 por campo, cobertura e proximidade.

    1. **Teto teórico por campo.** Cada campo é dividido pela nota que um documento teria se
       tivesse todos os termos da consulta naquele campo (Σ idf·(k1+1)), não pelo melhor
       documento encontrado. Com o máximo observado, um só termo comum num campo curto ("ORDEM"
       na tipologia, "Francisco" em pessoas) valia o máximo do campo e passava na frente de quem
       tinha a consulta inteira na descrição.
    2. **Cobertura.** A nota é multiplicada por (fração dos termos presentes)², com cada termo
       pesando o seu IDF: faltar "terceira" custa mais que faltar "são".
    3. **Proximidade.** Quem tem termos vizinhos da consulta em sequência ("ordem terceira",
       "são francisco") sobe; quem tem a consulta inteira sobe mais.
    """
    lexico = dados.lexico
    total = np.zeros(len(lexico), dtype=np.float64)
    for campo in lexico.campos:
        peso = float(lexico.pesos.get(campo, 0.0))
        if not peso:
            continue
        vocab = lexico._vocab(campo, list(dict.fromkeys(tokens)))
        teto = sum(idf for _, idf in vocab.values()) * (K1 + 1)
        if teto > 0:
            total += peso * lexico.notas_do_campo(campo, tokens) / teto

    idf = lexico.idf_maximo(tokens)
    soma = sum(idf.values())
    if soma > 0:
        cobertura = np.zeros(len(lexico), dtype=np.float64)
        for termo, valor in idf.items():
            cobertura[lexico.linhas_com_termo(termo)] += valor / soma
        total *= cobertura ** EXPOENTE_COBERTURA

    return total * _fator_de_proximidade(lexico, tokens)


def _fator_de_proximidade(lexico, tokens: list[str]) -> np.ndarray | float:
    """1 + (pares vizinhos achados + BONUS_FRASE_INTEIRA·frase inteira) / número de pares."""
    pares = [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a != b]
    if not pares:
        return 1.0
    frases = pares + ([" ".join(tokens)] if len(tokens) > 2 else [])
    achou = lexico.frases_por_linha(frases).astype(np.float64)
    bonus = achou[:, :len(pares)].sum(axis=1)
    if len(frases) > len(pares):
        bonus += BONUS_FRASE_INTEIRA * achou[:, -1]
    return 1.0 + bonus / len(pares)


def termos_no_corpus(dados, tokens: Iterable[str]) -> list[str]:
    """Quais termos da consulta existem em algum campo do léxico.

    Lê o vocabulário do índice, e por isso vale em qualquer modo — inclusive no Conceitual, em
    que o motor léxico não roda. É a condição (a) do aviso da §5.5.
    """
    return dados.lexico.termos_existentes(tokens)


def _frases_exigidas(texto: str) -> list[str]:
    """As sequências entre aspas, já normalizadas (§5.5)."""
    return [f for f in (" ".join(tokenizar(bruto)) for bruto in re.findall(r'"([^"]+)"', texto)) if f]


def _mascara_de_frases(dados, frases: list[str]) -> np.ndarray | None:
    """Quem contém todas as frases exigidas, na sequência literal."""
    if not frases:
        return None
    mascara = np.ones(len(dados.lexico), dtype=bool)
    for frase in frases:
        mascara &= dados.lexico.mascara_de_frase(frase)
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


def destaque_do_topo(cos: np.ndarray) -> float:
    """Quantos desvios-padrão o melhor documento está acima da mediana da distribuição.

    Candidato a substituir o piso absoluto do aviso da §5.5 no WP 4.2. Medido em 08/10/2026:

        consulta                      máximo   destaque
        "mapa de habitantes"           0,857      7,48
        "demarcação de limites"        0,843      5,73
        "deserção de soldados"         0,843      5,60
        "cavalos"                      0,809      4,82
        "Lavradio"                     0,779      4,61
        ----------------------------------------------- consultas sem sentido
        "zzzz"                         0,758      4,41
        "plorb"                        0,764      3,88
        "qwerty asdf"                  0,776      3,82
        "xkq zzv plorb"                0,787      3,74

    O máximo das boas (0,779–0,857) e o das sem sentido (0,758–0,787) se sobrepõem; o destaque
    (4,61–7,48 contra 3,74–4,41) quase não.
    """
    desvio = float(cos.std())
    if desvio <= 0:
        return 0.0
    return (float(cos.max()) - float(np.median(cos))) / desvio


def cossenos(dados, texto: str, encoder: Encoder) -> np.ndarray:
    """Cosseno bruto entre a consulta e cada documento (os vetores já são L2-normalizados)."""
    vetor = np.asarray(encoder([texto]), dtype=np.float32)
    if vetor.ndim == 2:
        vetor = vetor[0]
    norma = float(np.linalg.norm(vetor))
    if norma and abs(norma - 1.0) > 1e-3:
        vetor = vetor / norma
    emb = dados.embeddings
    if hasattr(emb, "produto"):
        return emb.produto(vetor)
    return np.asarray(emb, dtype=np.float32) @ vetor


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
    cosseno_minimo: float = COSSENO_MINIMO, limite_destaque: float = LIMITE_DESTAQUE,
    filtro: np.ndarray | None = None,
) -> Resposta:
    """A busca da §5.2–§5.5. `encoder=None` desliga o motor semântico (busca só léxica).

    `cosseno_minimo` é o piso absoluto do motor semântico, calibrado para o e5. Outro modelo tem
    outra escala de cosseno: quem troca o modelo (WP 4.2) passa o piso dele aqui.
    """
    if modo not in MODOS:
        raise ValueError(f"Modo desconhecido: {modo!r}. Use um de {', '.join(MODOS)}.")
    if rigor not in RIGORES:
        raise ValueError(f"Rigor desconhecido: {rigor!r}. Use um de {', '.join(RIGORES)}.")

    uids = dados.lexico.doc_uids
    chave, resto = _trecho_da_cota(texto)
    linhas_da_cota = tabela_de_cotas(dados).get(chave, []) if chave else []
    if filtro is not None:
        # Os filtros valem para a rota de cotas também: o que o filtro exclui não volta por cota.
        linhas_da_cota = [i for i in linhas_da_cota if filtro[i]]
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
    if filtro is not None:
        mascara = filtro if mascara is None else (mascara & filtro)
    cos = None

    # Sem texto nenhum, a resposta é o corpus que passou pelos filtros: é assim que uma lente, ou
    # um filtro sozinho, serve para navegar. O total continua sendo o real (§5.5).
    if not tokens and not chave:
        linhas = (np.flatnonzero(mascara) if mascara is not None
                  else np.arange(len(uids))).tolist()
        resultados = [Resultado(doc_uid=uids[i], linha=i, posicao=p, posicoes={"corpus": p})
                      for p, i in enumerate(linhas, start=1)]
        # Se o usuário digitou algo e sobrou nada, ele tem de saber por quê.
        avisos = [AVISO_SO_STOPWORDS] if (texto or "").strip() else []
        return Resposta(resultados=resultados, total=len(resultados), modo=modo, rigor=rigor,
                        termos_consulta=[], avisos=avisos)

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

    # A evidência de termos: em que linhas cada termo da consulta aparece (listas do índice).
    linhas_do_termo = {t: set(dados.lexico.linhas_com_termo(t).tolist())
                       for t in dict.fromkeys(tokens)} if ordem else {}
    for posicao, r in enumerate(ordem, start=1):
        r.doc_uid = uids[r.linha]
        r.posicao = posicao
        if cos is not None:
            r.cosseno = round(float(cos[r.linha]), 4)
        if linhas_do_termo:
            r.termos = sorted(t for t, linhas in linhas_do_termo.items() if r.linha in linhas)

    maximo = None if cos is None else round(float(cos.max()), 4)
    destaque = None if cos is None else round(destaque_do_topo(cos), 2)
    sem_lexico = bool(tokens) and not termos_no_corpus(dados, tokens)
    avisos = ([AVISO_DISTANTE]
              if sem_lexico and destaque is not None and destaque < limite_destaque else [])
    return Resposta(
        resultados=ordem, total=len(ordem), modo=modo, rigor=rigor, cota=chave, pesos=pesos,
        termos_consulta=tokens,
        truncado_por_motor=any(len(v) >= limite_motor for v in por_motor.values()),
        cosseno_maximo=maximo, destaque=destaque, sem_correspondencia_lexica=sem_lexico,
        avisos=avisos,
    )


