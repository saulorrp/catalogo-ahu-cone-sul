"""texto.py — normalização de texto e de nomes.

Esta é a **única** casa de `dobrar`, `chave_nome` e `chave_titulo`. As funções vieram do
`comum.py` do pipeline de biografias v2 e têm exatamente o mesmo comportamento: é por elas que
o `pessoa_uid` é calculado (spec §3.3), por isso mudá-las muda identificadores públicos.

Estado: o WP 1.3 trouxe a parte de nomes, que o `build/importar_biografias_v2.py` usa. O WP 1.5
trouxe a normalização de busca da spec §5.1, que o léxico BM25 precisa. Falta ao WP 2.1 a rota de
cotas (§5.3) e fazer o pipeline de biografias importar daqui, para não ficarem duas cópias.

**Regra:** `chave_nome` e `chave_titulo` são congeladas. Qualquer mudança na sua saída muda a
assinatura das pessoas e, com ela, os `pessoa_uid` já publicados; ela exige subir
`ASSINATURA_VERSAO` em `build/importar_biografias_v2.py` e migrar o `registro_pessoas.json`.
"""
from __future__ import annotations

import re
import unicodedata

# Partículas de ligação: não distinguem nomes ("Melo e Castro" ~ "Melo Castro").
PARTICULAS = frozenset({"de", "da", "do", "das", "dos", "e", "d'", "del", "la", "y", "van", "von",
                        "o", "a"})

# Tratamentos que o nome pode ou não trazer ("D. Diogo de Castro" ~ "Diogo de Castro").
TRATAMENTOS = frozenset({"d", "dom", "dona", "frei", "fr", "padre", "dr"})

# Grafias antigas -> modernas, aplicadas palavra a palavra sobre o texto já dobrado.
_GRAFIAS = [
    (r"th", "t"), (r"ph", "f"), (r"y", "i"), (r"ll", "l"), (r"ss", "s"), (r"tt", "t"),
    (r"z$", "s"), (r"ç", "c"),
]

# Variantes gráficas do mesmo nome, depois das regras acima.
_EQUIV = {
    "manoel": "manuel", "luiz": "luis", "antonio": "antonio", "affonso": "afonso",
    "balthazar": "baltazar", "baltasar": "baltazar", "christovao": "cristovao",
    "christoval": "cristovao", "jeronymo": "jeronimo", "ignacio": "inacio", "ayres": "aires",
    "ruy": "rui", "felipe": "filipe", "thome": "tome", "correa": "correia", "mello": "melo",
    "menezes": "meneses", "souza": "sousa", "benevides": "benavides", "araujo": "araujo",
    "cypriano": "cipriano", "custodio": "custodio", "cequeira": "sequeira", "matheus": "mateus",
    "mathias": "matias", "nicolao": "nicolau", "domingo": "domingos",
}

_RE_ORDINAL = re.compile(r"^\d+\s*[ºo°]?\s*")


def dobrar(texto: str) -> str:
    """Minúsculas, sem acentos, sem pontuação; espaços simples."""
    if not texto:
        return ""
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = t.replace("-", " ")
    return re.sub(r"\s+", " ", t).strip()


def chave_nome(nome: str) -> str:
    """Chave canônica para agrupar variantes gráficas do MESMO nome.

    'D. Manoel de Mello e Castro' e 'Manuel de Melo Castro' dão a mesma chave.
    """
    toks = []
    for palavra in dobrar(nome).split():
        if palavra in TRATAMENTOS:
            continue
        for antiga, nova in _GRAFIAS:
            palavra = re.sub(antiga, nova, palavra)
        palavra = _EQUIV.get(palavra, palavra)
        if palavra in PARTICULAS:
            continue
        toks.append(palavra)
    return " ".join(toks)


def chave_titulo(titulo: str) -> str:
    """'9º conde da Feira' -> 'conde feira'; 'marquês do Lavradio' -> 'marques lavradio'."""
    t = _RE_ORDINAL.sub("", dobrar(titulo))
    return " ".join(w for w in t.split() if w not in PARTICULAS)


# ---------------------------------------------------------------------------
# Normalização de busca (spec §5.1), usada igualmente no corpus e na consulta
# ---------------------------------------------------------------------------

# Lista curta e revisada (spec §5.1, item 4). Fora dela, nada é removido.
STOPWORDS = frozenset({"de", "do", "da", "dos", "das", "a", "o", "as", "os", "e", "em", "para",
                       "ao", "aos", "que", "com", "por", "na", "no"})

_RE_DOBRADA = re.compile(r"(.)\1")
_RE_SEM_ACENTO = re.compile(r"[^a-z0-9\s]+")


def fonetica(palavra: str) -> str:
    """Neutraliza grafia antiga: 'Paes'~'Pais', 'Jozé'~'José', 'Mello'~'Melo', 'Xássim'~'Chassim'.

    São as regras do `_fonetica` do `p4_validar.py`, como a spec §5.1 manda. Duas diferenças em
    relação à lista resumida da §5.1, que segue o p4: todo `z` vira `s` (não só no fim) e há
    também `oe -> oi`.
    """
    t = _RE_DOBRADA.sub(r"\1", palavra)
    for antiga, nova in (("ae", "ai"), ("oe", "oi"), ("ch", "x"), ("z", "s"), ("y", "i"),
                         ("th", "t"), ("ph", "f")):
        t = t.replace(antiga, nova)
    return t


def normalizar_busca(texto: str) -> str:
    """NFKD sem acentos, minúsculas, pontuação virando espaço (spec §5.1, itens 1 e 2)."""
    if not texto:
        return ""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", _RE_SEM_ACENTO.sub(" ", t)).strip()


def tokenizar(texto: str, stopwords: bool = True, foneticamente: bool = True) -> list[str]:
    """Tokens de busca de um campo ou de uma consulta.

    `stopwords=False` mantém a lista curta da §5.1 (útil para frase entre aspas, no WP 2.1);
    `foneticamente=False` mantém a grafia como está.
    """
    tokens = []
    for palavra in normalizar_busca(texto).split():
        if stopwords and palavra in STOPWORDS:
            continue
        tokens.append(fonetica(palavra) if foneticamente else palavra)
    return tokens
