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

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ahu import conversor
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
# Quantos do topo da fusão o reranker relê (docs/plano_busca_servidor.md).
LIMITE_RERANKER = int(os.environ.get("AHU_RERANKER_TOPO", "30"))
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
    'AHU_CU_005-01, Cx. 105, D. 20.367' -> 'ahu cu 005 01 cx 105 d 20367'
    """
    t = unicodedata.normalize("NFKD", str(texto or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    # Milhar com ponto, como nos catálogos impressos: `D. 20.367` é o documento 20367.
    t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# A série pode trazer subsérie: `AHU_CU_017-01` no Resgate, `PT/AHU/CU/017-001` na referência.
# Achatadas, ficam `cu 017 01` e `cu 017 001`; a chave guarda as duas como `17-1`.
_SERIE = r"(?:pt )?ahu (?:acl )?cu (\d{1,4})(?: (\d{1,3}))?"
# `AHU_ACL_CU_021, Cx. 6, D. 390`, `AHU_CU_017-01, Cx. 57, D. 13257-13259` e as variações de
# pontuação e espaço. O fim de um intervalo (`13259`) fica no grupo 5.
_RE_CU_CX = re.compile(_SERIE + r" cx (\d{1,4}) (?:d|doc|documento) (\d{1,6})(?: (\d{1,6}))?\b")
# `PT/AHU/CU/021/0006/00390`, `PT/AHU/CU/017-001/0057/13257`: números seguidos, sem cx/d.
_RE_CU_REF = re.compile(_SERIE + r" (\d{1,5}) (\d{1,6})\b")
# `AHU-Santa Catarina, cx. 8, doc. 14`: a região no lugar do número do fundo.
_RE_REGIAO = re.compile(r"\bahu ([a-z]+(?: [a-z]+){0,3}?) cx (\d{1,4}) (?:d|doc) (\d{1,6})\b")
# Os pares seguintes de uma cota antiga: `cx. 183, doc. 60; cx. 188, doc. 39`.
_RE_PAR = re.compile(r"\bcx (\d{1,4}) (?:d|doc) (\d{1,6})(?: (\d{1,6}))?\b")

_MOTIVOS = (("cu_cx", _RE_CU_CX), ("cu_ref", _RE_CU_REF), ("regiao", _RE_REGIAO))

# Um intervalo do Resgate (`D. 13257-13259`) responde por cada documento dele, até este tamanho.
INTERVALO_MAXIMO = 50


def _chave_cu(serie: str, subserie: str | None, caixa: str, documento: int) -> str:
    fundo = f"{int(serie)}" + (f"-{int(subserie)}" if subserie is not None else "")
    return f"cu|{fundo}|{int(caixa)}|{documento}"


def _intervalo(inicio: int, fim: str | None) -> list[int]:
    if fim is None or not (inicio < int(fim) <= inicio + INTERVALO_MAXIMO):
        return [inicio]
    return list(range(inicio, int(fim) + 1))


def _casar(achatado: str):
    """O primeiro padrão de cota que casa, como (motivo, match); ou (None, None)."""
    for motivo, padrao in _MOTIVOS:
        for m in padrao.finditer(achatado):
            if motivo == "regiao" and m.group(1).strip() in {"cu", "acl cu", "acl"}:
                # 'ahu acl cu' já foi tratado acima; aqui a "região" nunca é a palavra 'cu'.
                continue
            return motivo, m
    return None, None


def _chave_do_match(motivo: str, m) -> str:
    if motivo == "regiao":
        return f"reg|{m.group(1).strip()}|{int(m.group(2))}|{int(m.group(3))}"
    return _chave_cu(m.group(1), m.group(2), m.group(3), int(m.group(4)))


def _pelo_conversor(texto: str):
    """Plano B da rota de cotas: (chaves, início, fim) pela leitura do conversor, ou None.

    O conversor foi escrito para as grafias tortas dos catálogos (`017Cx. 48`, `C x. 1`,
    `017, 8, D. 818`, `D. 245A`), que os padrões acima não leem; sem isto, 82 documentos da base
    não voltavam pela própria cota. Só vale quando os padrões de cima não acham nada.
    """
    achado = conversor.localizar(texto)
    if achado is None:
        return None
    cota, inicio, fim = achado
    fim_doc = str(cota.documento_fim) if cota.documento_fim else None
    chaves = [_chave_cu(str(cota.serie), None if cota.subserie is None else str(cota.subserie),
                        str(cota.caixa), d) for d in _intervalo(cota.documento, fim_doc)]
    return chaves, inicio, fim


def chave_cota(texto: str) -> str | None:
    """Chave canônica de uma cota, ou None se o texto não contiver uma.

    As três famílias caem na mesma chave quando são o mesmo documento: zeros à esquerda somem, e
    a subsérie `01`/`001` vira `-1` (`cu|17-1|57|13257`). De um intervalo vale o primeiro documento.
    """
    motivo, m = _casar(_achatar(texto))
    if m:
        return _chave_do_match(motivo, m)
    plano_b = _pelo_conversor(texto)
    return plano_b[0][0] if plano_b else None


def chaves_do_campo(texto: str) -> list[str]:
    """Todas as chaves que um campo de cota da base responde, para a tabela de lookup.

    Além da chave da `chave_cota`: cada documento de um intervalo (`D. 13257-13259` responde por
    13257, 13258 e 13259) e cada par de uma cota antiga com várias (`cx. 183, doc. 60; cx. 188,
    doc. 39`).
    """
    achatado = _achatar(texto)
    motivo, m = _casar(achatado)
    if not m:
        plano_b = _pelo_conversor(texto)
        return plano_b[0] if plano_b else []
    if motivo == "cu_cx":
        return [_chave_cu(m.group(1), m.group(2), m.group(3), d)
                for d in _intervalo(int(m.group(4)), m.group(5))]
    if motivo == "cu_ref":
        return [_chave_do_match(motivo, m)]
    regiao = m.group(1).strip()
    chaves = [_chave_do_match(motivo, m)]
    for par in _RE_PAR.finditer(achatado, m.start(2) - 3):
        for d in _intervalo(int(par.group(2)), par.group(3)):
            chave = f"reg|{regiao}|{int(par.group(1))}|{d}"
            if chave not in chaves:
                chaves.append(chave)
    return chaves


def _trecho_da_cota(texto: str) -> tuple[str | None, str]:
    """Devolve (chave, resto do texto sem a cota). O resto segue a busca normal (§5.3)."""
    achatado = _achatar(texto)
    motivo, m = _casar(achatado)
    if not m:
        plano_b = _pelo_conversor(texto)
        if plano_b is None:
            return None, achatado
        chaves, inicio, fim = plano_b
        return chaves[0], _achatar(texto[:inicio] + " " + texto[fim:])
    resto = re.sub(r"\s+", " ", achatado[: m.start()] + " " + achatado[m.end():]).strip()
    return _chave_do_match(motivo, m), resto


# v2: subsérie na chave (`cu|17-1|...`), intervalos e cotas antigas com vários pares.
# v3: as grafias tortas lidas pelo conversor (plano B).
_ATRIBUTO_TABELA = "_tabela_cotas_v3"


def tabela_de_cotas(dados) -> dict[str, list[int]]:
    """chave de cota -> linhas de `documentos.parquet`. Lida das três colunas de cota.

    Construída uma vez e guardada no próprio `Dados` (é derivada, não um artefato do §3.4). O
    nome do atributo leva a versão das chaves: o `Dados` em `st.cache_resource` sobrevive a uma
    atualização do código, e uma tabela com as chaves antigas não pode ser reaproveitada.
    """
    cache = getattr(dados, _ATRIBUTO_TABELA, None)
    if cache is not None:
        return cache
    tabela: dict[str, list[int]] = {}
    colunas = [c for c in ("reference_code", "new_code", "old_code") if c in dados.documentos.columns]
    for coluna in colunas:
        for i, valor in enumerate(dados.documentos[coluna]):
            if not isinstance(valor, str) or not valor:
                continue
            for chave in chaves_do_campo(valor):
                linhas = tabela.setdefault(chave, [])
                if i not in linhas:
                    linhas.append(i)
    object.__setattr__(dados, _ATRIBUTO_TABELA, tabela)
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
    # O codificador estático (ahu/estatico.py) traz os vetores dele; o e5 usa os de `dados`.
    emb = encoder if hasattr(encoder, "produto") else dados.embeddings
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


def pesos_do_rrf(tokens: Iterable[str], tabela: dict | None = None) -> dict[str, float]:
    n = len(list(tokens))
    return dict((tabela or PESOS_RRF)["curta" if n <= PALAVRAS_CONSULTA_CURTA else "longa"])


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


def reordenar(dados, texto: str, ordem: list[Resultado], reranker,
              limite: int | None = None) -> list[Resultado]:
    """O cross-encoder relê os `limite` primeiros da fusão; o resto segue na ordem do RRF.

    A nota dele não substitui a do RRF de quem fica de fora: só a ordem do topo muda, e o total
    continua o mesmo.
    """
    from ahu.reranker import texto_do_documento

    limite = limite or LIMITE_RERANKER
    topo, cauda = ordem[:limite], ordem[limite:]
    docs = dados.documentos
    textos = [texto_do_documento(docs.iloc[r.linha]) for r in topo]
    notas = np.asarray(reranker(texto, textos), dtype=np.float32)
    posicoes = np.argsort(-notas, kind="stable")
    for p, i in enumerate(posicoes, start=1):
        topo[i].posicoes["reranker"] = p
    return [topo[i] for i in posicoes] + cauda


def refinar(dados, texto: str, resposta: "Resposta", reranker) -> "Resposta":
    """Passa o reranker por uma `Resposta` que saiu sem ele, e devolve outra, sem tocar na primeira.

    É o segundo tempo da busca em dois tempos da interface web: a fusão responde na hora, e o
    cross-encoder relê o topo depois. O resultado é **o mesmo** de `buscar(..., reranker=...)`
    (há teste): a ordem da fusão se refaz pela nota do RRF, o topo é relido, e os documentos da
    cota citada na frase continuam na frente.
    """
    from dataclasses import replace

    if reranker is None or resposta.modo == "exata" or not resposta.resultados:
        return resposta
    chave, resto = _trecho_da_cota(texto)
    copias = [replace(r, posicoes=dict(r.posicoes)) for r in resposta.resultados]
    da_cota = [r for r in copias if "cota" in r.posicoes]
    # Quem veio só pela cota não passou pela fusão; os demais voltam à ordem que ela deu.
    fundidos = sorted((r for r in copias if set(r.posicoes) - {"cota"}),
                      key=lambda r: (-r.nota_rrf, r.linha))
    if not fundidos:
        return resposta
    ordem = reordenar(dados, resto if chave else texto, fundidos, reranker)
    ordem = da_cota + [r for r in ordem if "cota" not in r.posicoes]
    for posicao, r in enumerate(ordem, start=1):
        r.posicao = posicao
    return replace(resposta, resultados=ordem)


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def buscar(
    dados, texto: str, modo: str = "automatico", rigor: str = "medio",
    encoder: Encoder | None = None, limite_motor: int = LIMITE_MOTOR,
    cosseno_minimo: float = COSSENO_MINIMO, limite_destaque: float = LIMITE_DESTAQUE,
    filtro: np.ndarray | None = None, reranker=None,
) -> Resposta:
    """A busca da §5.2–§5.5. `encoder=None` desliga o motor semântico (busca só léxica).

    `cosseno_minimo` é o piso absoluto do motor semântico, calibrado para o e5. Outro modelo tem
    outra escala de cosseno: quem troca o modelo (WP 4.2) passa o piso dele aqui. Um `encoder`
    que traga `cosseno_minimo`, `margens_rigor` e `pesos_rrf` (o estático) usa os dele.

    `reranker` (ver `ahu/reranker.py`) relê o topo da fusão e o reordena; o modo Exata não passa
    por ele, porque ali a ordem tem de ser explicável pelas palavras.
    """
    cosseno_minimo = getattr(encoder, "cosseno_minimo", cosseno_minimo)
    margens = getattr(encoder, "margens_rigor", MARGENS_RIGOR)
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
        limiar = max(cosseno_minimo, float(cos.max()) - margens[rigor])
        por_motor["semantico"] = _topo(cos, limite_motor, minimo=limiar, permitidos=mascara)

    pesos = pesos_do_rrf(tokens, getattr(encoder, "pesos_rrf", None))
    fundidos = _fundir(por_motor, pesos)

    # §5.3: a cota dentro da frase -> esses documentos primeiro, o resto da frase segue a busca.
    ordem = sorted(fundidos.values(), key=lambda r: (-r.nota_rrf, r.linha))
    if reranker is not None and modo != "exata" and ordem:
        ordem = reordenar(dados, resto if chave else texto, ordem, reranker)
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


