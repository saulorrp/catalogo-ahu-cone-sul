"""selecao.py — a seleção de documentos que atravessa buscas (spec §8.1, plano v3.1 R2).

Sem Streamlit. A `Selecao` é uma lista ordenada de `doc_uid` — nada mais. O `app.py` a guarda em
`st.session_state`, e é só isso: **nada é persistido no servidor** (R4).

Três coisas que a §8.1 pede e que esta classe garante:

1. **a seleção atravessa buscas.** Ela não guarda posição, nota nem página: só o `doc_uid`, que é
   o identificador estável (AGENTS.md 6). Mudar a busca não mexe nela;
2. **teto de 500 documentos.** `adicionar` nunca passa do teto e diz quantos ficaram de fora, para
   a interface poder avisar em vez de truncar em silêncio;
3. **ordem significa algo.** A ordem é a de chegada — a ordem em que o pesquisador marcou —, e é
   nessa ordem que o PDF sai. Nunca é a ordem da busca.

**Como a seleção sobrevive a um recarregamento (R2).** Nada é guardado no servidor, e o JSON de
baixar e carregar saiu com a D7. O que ficou é o espelho na **URL**: `para_url` escreve um prefixo
curto do `doc_uid` de cada documento e `de_url` os resolve de volta. O prefixo é o mais curto que
**não colide em nenhum documento do corpus ativo** — calculado nos dados, nunca escrito no código
—, e se o resultado passar de `LIMITE_URL` caracteres o espelho é desligado e a interface diz por
quê. Um recarregamento, um favorito ou um link copiado trazem a seleção de volta sem que nenhum
identificador apareça na tela (D10).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LIMITE = 500            # §8.1

# O espelho na URL (R2). 1.800 caracteres é um teto conservador: o limite prático dos navegadores
# e dos servidores fica acima disso, mas uma URL mais longa que esta já não se copia nem se
# compartilha com conforto. Acima do teto, a interface avisa e manda gerar o PDF.
LIMITE_URL = 1800
CHAVE_URL = "sel"
SEPARADOR_URL = ","
MINIMO_PREFIXO = 6


@dataclass
class Selecao:
    """Os `doc_uid` marcados, na ordem em que foram marcados."""

    uids: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.uids)

    def __contains__(self, doc_uid: str) -> bool:
        return doc_uid in self.uids

    @property
    def cheia(self) -> bool:
        return len(self.uids) >= LIMITE

    def limpar(self) -> None:
        self.uids = []

    def adicionar(self, doc_uids) -> int:
        """Acrescenta o que ainda não está, sem passar do teto. Devolve quantos **não** couberam."""
        if isinstance(doc_uids, str):
            doc_uids = [doc_uids]
        novos = [u for u in dict.fromkeys(doc_uids) if u and u not in self.uids]
        espaco = LIMITE - len(self.uids)
        self.uids.extend(novos[:espaco])
        return max(0, len(novos) - espaco)

    def remover(self, doc_uids) -> None:
        if isinstance(doc_uids, str):
            doc_uids = [doc_uids]
        fora = set(doc_uids)
        self.uids = [u for u in self.uids if u not in fora]

    def alternar(self, doc_uid: str, marcado: bool) -> int:
        """O que a caixa "Selecionar" de cada resultado faz. Devolve quantos não couberam."""
        if marcado:
            return self.adicionar(doc_uid)
        self.remover(doc_uid)
        return 0

    # -----------------------------------------------------------------
    # Ver (R2): uma linha por documento, e nada mais
    # -----------------------------------------------------------------
    def linhas(self, dados) -> list[dict]:
        """Uma linha por documento, na ordem em que foram marcados.

        É o que o painel mostra e o que o PDF imprime: ano, cota, código de referência e resumo.
        O `doc_uid` volta só em `uid`, para o ✕ saber que linha remover; ele não é para a tela
        (D10).
        """
        from ahu.links import referencia_crav

        docs = dados.documentos
        saida = []
        for posicao, uid in enumerate(self.uids, start=1):
            linha = docs.iloc[dados.indice_de(uid)]

            def valor(campo, linha=linha):
                v = linha.get(campo)
                return "" if v is None or pd.isna(v) else v

            ano = valor("ano_isolado")
            cota = valor("new_code") or valor("reference_code") or ""
            if str(cota).upper() == "UNFORMATTED_CODE":
                cota = valor("reference_code") or ""
            saida.append({
                "n": posicao,
                "uid": uid,
                "ano": "" if ano == "" else int(ano),
                "cota": str(cota).rstrip("."),
                "referencia": (referencia_crav(valor("reference_code"))
                               or referencia_crav(valor("new_code")) or ""),
                "resumo": str(valor("description") or ""),
                "tipologia": str(valor("tipologia") or ""),
            })
        return saida

    def tabela(self, dados) -> pd.DataFrame:
        """As linhas em forma de tabela. Sem identificador (D10)."""
        rotulos = {"n": "Nº", "ano": "Ano", "cota": "Cota",
                   "referencia": "Código de referência"}
        return pd.DataFrame(
            [{rotulo: linha[campo] for campo, rotulo in rotulos.items()}
             for linha in self.linhas(dados)],
            columns=list(rotulos.values()))

    # -----------------------------------------------------------------
    # Espelho na URL (R2)
    # -----------------------------------------------------------------
    def para_url(self, dados) -> str | None:
        """Os prefixos curtos, separados por vírgula. `None` quando não cabe na URL."""
        if not self.uids:
            return ""
        tamanho = tamanho_do_prefixo(dados)
        texto = SEPARADOR_URL.join(_parte_variavel(u)[:tamanho] for u in self.uids)
        return None if len(texto) > LIMITE_URL else texto

    @classmethod
    def de_url(cls, texto: str | None, dados) -> tuple["Selecao", int]:
        """Resolve os prefixos de volta. Devolve a seleção e quantos não foram reconhecidos."""
        selecao = cls()
        if not texto:
            return selecao, 0
        por_prefixo = _indice_de_prefixos(dados)
        tamanho = tamanho_do_prefixo(dados)
        achados, perdidos = [], 0
        for pedaco in texto.split(SEPARADOR_URL):
            pedaco = pedaco.strip()
            if not pedaco:
                continue
            uid = por_prefixo.get(pedaco[:tamanho])
            if uid is None:
                perdidos += 1
            else:
                achados.append(uid)
        selecao.adicionar(achados)
        return selecao, perdidos


# ---------------------------------------------------------------------------
# Prefixos curtos (R2)
# ---------------------------------------------------------------------------

def _parte_variavel(doc_uid: str) -> str:
    """A parte que distingue um documento do outro: o que vem depois da coleção."""
    return str(doc_uid).split(":")[-1]


def tamanho_do_prefixo(dados) -> int:
    """O prefixo mais curto que **não colide** em nenhum documento do corpus ativo.

    Começa em `MINIMO_PREFIXO` e cresce até não haver dois documentos com o mesmo prefixo. O
    número sai dos dados, nunca do código: uma coleção nova, com mais documentos, pode exigir um
    caractere a mais, e aí ele cresce sozinho (D6, A11).
    """
    cache = getattr(dados, "_tamanho_prefixo_selecao", None)
    if cache is not None:
        return cache
    partes = [_parte_variavel(u) for u in dados.documentos["doc_uid"]]
    maior = max((len(p) for p in partes), default=MINIMO_PREFIXO)
    tamanho = MINIMO_PREFIXO
    while tamanho < maior and len({p[:tamanho] for p in partes}) != len(partes):
        tamanho += 1
    object.__setattr__(dados, "_tamanho_prefixo_selecao", tamanho)
    return tamanho


def _indice_de_prefixos(dados) -> dict[str, str]:
    cache = getattr(dados, "_indice_prefixos_selecao", None)
    if cache is not None:
        return cache
    tamanho = tamanho_do_prefixo(dados)
    indice = {_parte_variavel(u)[:tamanho]: str(u) for u in dados.documentos["doc_uid"]}
    object.__setattr__(dados, "_indice_prefixos_selecao", indice)
    return indice


def maximo_na_url(dados) -> int:
    """Quantos documentos cabem no espelho da URL, com o prefixo deste corpus."""
    por_documento = tamanho_do_prefixo(dados) + len(SEPARADOR_URL)
    return (LIMITE_URL + len(SEPARADOR_URL)) // por_documento
