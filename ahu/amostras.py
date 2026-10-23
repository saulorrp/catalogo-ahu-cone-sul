"""amostras.py — WP 3.5: amostra diacrônica, amostra estratificada e plano EINS (§7.2–§7.4).

Sem Streamlit. Três coisas:

* **`amostra_diacronica`** (§7.2) — os documentos de uma pessoa, em ordem cronológica, em lotes de
  10, com a opção de no máximo N por ano para a amostra não se concentrar num único ano.
* **`amostra_estratificada`** (§7.3) — sobre o resultado atual (busca + filtros), 1 ou 2 estratos
  entre década, categoria social, tipologia, pasta e país de nascimento, N por estrato, com
  **semente visível** e um relatório de disponíveis × sorteados.
* **`plano_de_coleta`** (§7.4) — o `plano_coleta.eins.json`, no esquema `eins-plano-coleta/1`, que
  o Acervo do EINS Lite importa.

**A7 — mesma semente, mesma amostra.** O sorteio usa um `random.Random(semente)` próprio, nunca o
estado global do `random`: duas chamadas com a mesma semente, sobre o mesmo conjunto, dão o mesmo
resultado, e isso vale mesmo que outra parte do programa sorteie coisas no meio.

**`autoria_verificada` por entidade (§7.2).** O que vem do pipeline é um sinalizador por
*documento*, copiado para todos os remetentes dele. A §7.2 pede que ele passe a ser calculado por
**entidade**, e é o que `autoria_verificada` faz aqui: verdadeiro quando um nome, uma variante ou
um título daquela pessoa aparece na descrição do documento, comparado pela normalização da §5.1
(acento e grafia antiga neutralizados, de modo que "Mello" casa com "Melo"). A regra está no
README. O cálculo é feito **sobre `app_dados/`**, sem reconstruir camada nenhuma; levá-lo para
dentro da camada de entidades é possível depois, e aí é uma camada nova.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ahu import links as mod_links
from ahu import proveniencia as mod_prov
from ahu.texto import chave_nome

TAMANHO_LOTE = 10           # §7.2: "em lotes de 10, como hoje"
ESQUEMA_EINS = "eins-plano-coleta/1"

# §7.3: os estratos possíveis. A chave é o que a interface mostra; o valor, como se calcula.
ESTRATOS = {
    "decada": "Década",
    "categoria_social": "Categoria social",
    "tipologia": "Tipologia",
    "pasta": "Pasta / região",
    "pais_nascimento": "País de nascimento do remetente",
}
SEM_VALOR = "(sem valor)"


class ErroDeAmostra(ValueError):
    """Pedido de amostra impossível: estrato desconhecido, mais de dois, ou N inválido."""


# ---------------------------------------------------------------------------
# autoria_verificada por entidade (§7.2)
# ---------------------------------------------------------------------------

def _formas_da_pessoa(dados, pessoa_uid: str) -> list[list[str]]:
    """Nome, variantes e títulos de uma pessoa, cada um como lista de tokens canônicos."""
    recorte = dados.pessoas[dados.pessoas["pessoa_uid"] == pessoa_uid]
    if recorte.empty:
        return []
    linha = recorte.iloc[0]
    brutas = [linha.get("nome_exibicao"), linha.get("nome_principal")]
    for campo in ("variantes_nome", "titulos"):
        valor = linha.get(campo)
        if valor is not None and not isinstance(valor, float):
            brutas.extend(valor)

    formas, vistas = [], set()
    for bruta in brutas:
        if bruta is None or (isinstance(bruta, float) and pd.isna(bruta)):
            continue
        tokens = tuple(chave_nome(str(bruta)).split())
        if not tokens or tokens in vistas:
            continue
        # Uma forma de um só token curto ("Sá", "Dias") casaria por acidente: é descartada.
        if len(tokens) == 1 and len(tokens[0]) <= 3:
            continue
        vistas.add(tokens)
        formas.append(list(tokens))
    return formas


def _sublista(agulha: list[str], palheiro: list[str]) -> bool:
    """A sequência de tokens aparece inteira e seguida no palheiro."""
    n = len(agulha)
    if not n or n > len(palheiro):
        return False
    return any(palheiro[i:i + n] == agulha for i in range(len(palheiro) - n + 1))


def autoria_verificada(dados, doc_uid: str, pessoa_uid: str) -> bool:
    """§7.2: um nome, uma variante ou um título da pessoa aparece na descrição do documento.

    A comparação é feita com a **`chave_nome`** do `ahu/texto.py` — a mesma canonicalização que
    produz o `pessoa_uid` (§3.3) — aplicada aos dois lados. É ela, e não a `fonetica` da busca,
    que sabe que "Manoel de Mello e Castro" e "Manuel de Melo Castro" são o mesmo nome: a
    `fonetica` neutraliza grafia (`oe -> oi`) e acabaria lendo "Manoel" como "manoil".

    Casa por **sequência de tokens**, não por pedaço de palavra: a forma tem de aparecer inteira e
    seguida, de modo que "Melo" não casa dentro de "Melosa".
    """
    formas = _formas_da_pessoa(dados, pessoa_uid)
    if not formas:
        return False
    linha = dados.documentos.iloc[dados.indice_de(doc_uid)]
    descricao = linha.get("description")
    if descricao is None or pd.isna(descricao):
        return False
    tokens = chave_nome(str(descricao)).split()
    return any(_sublista(forma, tokens) for forma in formas)


# ---------------------------------------------------------------------------
# Amostra diacrônica (§7.2)
# ---------------------------------------------------------------------------

@dataclass
class Amostra:
    """O resultado de qualquer das duas amostragens, já em lotes."""

    doc_uids: list[str]
    lotes: list[list[str]]
    tipo: str
    relatorio: dict = field(default_factory=dict)
    pessoa_uid: str | None = None
    estratos: list[str] = field(default_factory=list)
    por_estrato: int | None = None
    maximo_por_ano: int | None = None
    semente: int | None = None
    # Para a amostra estratificada: qual estrato gerou cada lote.
    estrato_do_lote: list[str | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.doc_uids)


def _em_lotes(uids: list[str], tamanho: int = TAMANHO_LOTE) -> list[list[str]]:
    return [uids[i:i + tamanho] for i in range(0, len(uids), tamanho)]


def amostra_diacronica(dados, pessoa_uid: str, maximo_por_ano: int | None = None,
                       so_autoria_verificada: bool = True,
                       tamanho_lote: int = TAMANHO_LOTE) -> Amostra:
    """Os documentos de uma pessoa em ordem cronológica, em lotes (§7.2).

    `maximo_por_ano` é o espaçamento da §7.2: com 2, no máximo dois documentos por ano entram, e
    são os dois primeiros daquele ano na ordem do corpus — a escolha é determinística, não
    sorteada, porque a amostra diacrônica não tem semente.

    Documento sem data não entra: uma amostra diacrônica sem ano não tem onde se colocar na
    série. O relatório diz quantos ficaram de fora e por quê.
    """
    if pessoa_uid not in set(dados.pessoas["pessoa_uid"]):
        raise ErroDeAmostra(f"pessoa_uid desconhecido: {pessoa_uid}")
    if maximo_por_ano is not None and maximo_por_ano < 1:
        raise ErroDeAmostra(f"O máximo por ano deve ser 1 ou mais, veio {maximo_por_ano}.")

    vinculos = dados.doc_pessoa
    meus = vinculos.loc[vinculos["pessoa_uid"] == pessoa_uid, "doc_uid"].astype(str).tolist()
    # dict.fromkeys: um documento com a mesma pessoa em duas posições conta uma vez.
    meus = list(dict.fromkeys(meus))

    docs = dados.documentos
    com_ano, sem_ano = [], []
    for uid in meus:
        ano = docs.iloc[dados.indice_de(uid)].get("ano_isolado")
        (sem_ano if ano is None or pd.isna(ano) else com_ano).append(uid)

    nao_verificados = []
    if so_autoria_verificada:
        mantidos = []
        for uid in com_ano:
            (mantidos if autoria_verificada(dados, uid, pessoa_uid)
             else nao_verificados).append(uid)
        com_ano = mantidos

    com_ano.sort(key=lambda u: (int(docs.iloc[dados.indice_de(u)]["ano_isolado"]), u))

    espacados, por_ano, cortados_pelo_ano = [], {}, 0
    for uid in com_ano:
        ano = int(docs.iloc[dados.indice_de(uid)]["ano_isolado"])
        if maximo_por_ano is not None and por_ano.get(ano, 0) >= maximo_por_ano:
            cortados_pelo_ano += 1
            continue
        por_ano[ano] = por_ano.get(ano, 0) + 1
        espacados.append(uid)

    return Amostra(
        doc_uids=espacados, lotes=_em_lotes(espacados, tamanho_lote), tipo="diacronica",
        pessoa_uid=pessoa_uid, maximo_por_ano=maximo_por_ano,
        estrato_do_lote=[None] * len(_em_lotes(espacados, tamanho_lote)),
        relatorio={
            "vinculados": len(meus),
            "sem_data": len(sem_ano),
            "sem_autoria_verificada": len(nao_verificados),
            "cortados_pelo_maximo_por_ano": cortados_pelo_ano,
            "sorteados": len(espacados),
            "anos": sorted(por_ano),
            "por_ano": {str(a): n for a, n in sorted(por_ano.items())},
        },
    )


# ---------------------------------------------------------------------------
# Amostra estratificada (§7.3)
# ---------------------------------------------------------------------------

def valor_do_estrato(dados, doc_uid: str, estrato: str) -> str:
    """O valor de um documento num estrato, como texto. `SEM_VALOR` quando não há dado."""
    if estrato not in ESTRATOS:
        raise ErroDeAmostra(
            f"Estrato desconhecido: {estrato!r}. Use um de {', '.join(ESTRATOS)}.")
    linha = dados.documentos.iloc[dados.indice_de(doc_uid)]

    def texto(campo):
        v = linha.get(campo)
        return SEM_VALOR if v is None or pd.isna(v) else str(v)

    if estrato == "decada":
        ano = linha.get("ano_isolado")
        return SEM_VALOR if ano is None or pd.isna(ano) else f"{int(ano) // 10 * 10}"
    if estrato == "categoria_social":
        return texto("sender_category")
    if estrato == "tipologia":
        return texto("tipologia")
    if estrato == "pasta":
        return texto("folder")
    return _pais_do_remetente(dados, doc_uid)


def _pais_do_remetente(dados, doc_uid: str) -> str:
    """País de nascimento do primeiro remetente com biografia exibível (D2)."""
    vinculos = dados.doc_pessoa
    recorte = vinculos[vinculos["doc_uid"] == doc_uid].sort_values("posicao")
    if recorte.empty:
        return SEM_VALOR
    pessoas = dados.pessoas.set_index("pessoa_uid")
    for uid in recorte["pessoa_uid"]:
        if uid is None or pd.isna(uid) or uid not in pessoas.index:
            continue
        p = pessoas.loc[uid]
        if not bool(p["nascimento_exibivel"]):
            continue
        pais = p.get("pais_atual")
        if pais is not None and not pd.isna(pais):
            return str(pais)
    return SEM_VALOR


def amostra_estratificada(dados, doc_uids, estratos: list[str], por_estrato: int,
                          semente: int, tamanho_lote: int = TAMANHO_LOTE) -> Amostra:
    """N documentos por estrato, sobre o conjunto dado, com semente visível (§7.3, A7).

    `estratos` tem 1 ou 2 nomes de `ESTRATOS`. Com dois, o estrato é o par de valores, e o
    relatório traz disponíveis × sorteados por par, mais a lista dos estratos que não tinham
    documentos suficientes.
    """
    estratos = list(estratos or [])
    if not 1 <= len(estratos) <= 2:
        raise ErroDeAmostra(
            f"A §7.3 pede 1 ou 2 estratos, veio {len(estratos)}.")
    for e in estratos:
        if e not in ESTRATOS:
            raise ErroDeAmostra(
                f"Estrato desconhecido: {e!r}. Use um de {', '.join(ESTRATOS)}.")
    if len(set(estratos)) != len(estratos):
        raise ErroDeAmostra(f"Os dois estratos são o mesmo: {estratos[0]!r}.")
    if por_estrato < 1:
        raise ErroDeAmostra(f"N por estrato deve ser 1 ou mais, veio {por_estrato}.")

    doc_uids = list(dict.fromkeys(doc_uids))
    if not doc_uids:
        raise ErroDeAmostra("Não há documentos no conjunto para sortear.")

    grupos: dict[str, list[str]] = {}
    for uid in doc_uids:
        chave = " · ".join(valor_do_estrato(dados, uid, e) for e in estratos)
        grupos.setdefault(chave, []).append(uid)

    sorteio = random.Random(semente)
    escolhidos: list[str] = []
    por_grupo, insuficientes = {}, []
    # A ordem dos estratos é a alfabética, e não a de aparição: é o que torna a amostra
    # reproduzível independentemente da ordem em que a busca devolveu os documentos.
    for chave in sorted(grupos):
        disponiveis = grupos[chave]
        quantos = min(por_estrato, len(disponiveis))
        # `sample` sobre a lista **ordenada**: a mesma semente dá a mesma amostra (A7).
        tirados = sorteio.sample(sorted(disponiveis), quantos)
        escolhidos.extend(tirados)
        por_grupo[chave] = {"disponiveis": len(disponiveis), "sorteados": quantos}
        if len(disponiveis) < por_estrato:
            insuficientes.append(chave)

    lotes, estrato_do_lote = [], []
    for chave in sorted(por_grupo):
        do_estrato = [u for u in escolhidos
                      if " · ".join(valor_do_estrato(dados, u, e) for e in estratos) == chave]
        for lote in _em_lotes(do_estrato, tamanho_lote):
            lotes.append(lote)
            estrato_do_lote.append(chave)

    return Amostra(
        doc_uids=escolhidos, lotes=lotes, tipo="estratificada", estratos=estratos,
        por_estrato=por_estrato, semente=semente, estrato_do_lote=estrato_do_lote,
        relatorio={
            "conjunto": len(doc_uids),
            "estratos": len(grupos),
            "sorteados": len(escolhidos),
            "por_estrato": por_grupo,
            "insuficientes": insuficientes,
        },
    )


# As colunas da tabela do lote, na ordem que o autor pediu (R9): o ano estreito à frente, o
# código de referência clicável, pasta e tipologia, e o resumo como a coluna larga.
COLUNAS_DO_LOTE = ("Incluir", "Nº", "Ano", "Código de referência", "Pasta", "Tipologia", "Resumo")


def tabela_dos_documentos(dados, doc_uids, excluidos=frozenset()) -> pd.DataFrame:
    """Os documentos de um lote, para a interface mostrar e editar.

    **Sem identificador interno** (D10): o que o pesquisador vê é ano, código de referência,
    pasta, tipologia e resumo. A **cota** do Resgate não entra — o autor pediu o código de
    referência, que é o que o DigitArq entende, e duas colunas de cota seriam repetição.

    `Código de referência` sai como **URL** do DigitArq; é a interface que a transforma em texto
    clicável (`LinkColumn` com `display_text`). `Incluir` começa marcada, menos para quem está
    em `excluidos`.
    """
    from ahu.links import digitarq_do_documento

    docs = dados.documentos
    linhas = []
    for posicao, uid in enumerate(doc_uids, start=1):
        linha = docs.iloc[dados.indice_de(uid)]

        def valor(campo, linha=linha):
            v = linha.get(campo)
            return "" if v is None or pd.isna(v) else v

        ano = valor("ano_isolado")
        linhas.append({
            "Incluir": uid not in excluidos,
            "Nº": posicao,
            "Ano": "" if ano == "" else int(ano),
            "Código de referência": digitarq_do_documento(linha)[0] or "",
            "Pasta": valor("folder"),
            "Tipologia": valor("tipologia"),
            "Resumo": str(valor("description") or ""),
        })
    return pd.DataFrame(linhas, columns=list(COLUNAS_DO_LOTE))


@dataclass
class LotesEditados:
    """Os lotes depois do que o pesquisador mexeu: ordem, inclusão e troca de lote (R9).

    Vive em `st.session_state`, não no `Amostra`: a amostra é o que o sorteio produziu, e isto é
    o que a pessoa decidiu em cima dele. Refazer a amostra joga isto fora, e é o que se quer.
    """

    lotes: list[list[str]] = field(default_factory=list)
    excluidos: set[str] = field(default_factory=set)

    @classmethod
    def de_amostra(cls, amostra: Amostra) -> "LotesEditados":
        return cls(lotes=[list(lote) for lote in amostra.lotes], excluidos=set())

    def incluidos(self, i: int) -> list[str]:
        """Os documentos do lote `i` que vão ao PDF, na ordem atual."""
        return [u for u in self.lotes[i] if u not in self.excluidos]

    def alternar(self, uid: str, incluir: bool) -> None:
        self.excluidos.discard(uid) if incluir else self.excluidos.add(uid)

    def mover(self, i: int, posicao: int, passo: int) -> bool:
        """Sobe (`passo=-1`) ou desce (`passo=+1`) uma linha dentro do lote."""
        lote = self.lotes[i]
        destino = posicao + passo
        if not (0 <= posicao < len(lote) and 0 <= destino < len(lote)):
            return False
        lote[posicao], lote[destino] = lote[destino], lote[posicao]
        return True

    def trocar_de_lote(self, origem: int, posicao: int, destino: int) -> bool:
        """Tira um documento de um lote e põe no fim de outro."""
        if not (0 <= origem < len(self.lotes) and 0 <= destino < len(self.lotes)):
            return False
        if origem == destino or not (0 <= posicao < len(self.lotes[origem])):
            return False
        self.lotes[destino].append(self.lotes[origem].pop(posicao))
        return True

    def total_incluido(self) -> int:
        return sum(len(self.incluidos(i)) for i in range(len(self.lotes)))


def tabela_do_relatorio(amostra: Amostra) -> pd.DataFrame:
    """O relatório da §7.3 em forma de tabela: disponíveis × sorteados por estrato."""
    por_estrato = amostra.relatorio.get("por_estrato") or {}
    return pd.DataFrame([
        {"Estrato": chave, "Disponíveis": v["disponiveis"], "Sorteados": v["sorteados"],
         "Suficiente": "não" if chave in amostra.relatorio.get("insuficientes", []) else "sim"}
        for chave, v in por_estrato.items()
    ])


# ---------------------------------------------------------------------------
# Plano de coleta para o EINS Lite (§7.4, A9)
# ---------------------------------------------------------------------------

def _pessoas_do_documento(dados, doc_uid: str) -> list[dict]:
    """As pessoas do plano. D2: cidade, país e nível só para biografia exibível."""
    vinculos = dados.doc_pessoa
    recorte = vinculos[vinculos["doc_uid"] == doc_uid].sort_values("posicao")
    pessoas = dados.pessoas.set_index("pessoa_uid")
    saida = []
    for _, v in recorte.iterrows():
        uid = v["pessoa_uid"]
        entrada = {
            "pessoa_uid": None if uid is None or pd.isna(uid) else str(uid),
            "nome": "(sem nome)" if pd.isna(v["nome"]) else str(v["nome"]),
            "tipo": None if pd.isna(v["tipo"]) else str(v["tipo"]),
            "pais_nascimento": None, "cidade_nascimento": None, "nivel": None,
        }
        if entrada["pessoa_uid"] and entrada["pessoa_uid"] in pessoas.index:
            p = pessoas.loc[entrada["pessoa_uid"]]
            if bool(p["nascimento_exibivel"]):
                for chave, campo in (("pais_nascimento", "pais_atual"),
                                     ("cidade_nascimento", "cidade_nascimento"),
                                     ("nivel", "nivel")):
                    valor = p.get(campo)
                    entrada[chave] = None if valor is None or pd.isna(valor) else str(valor)
        saida.append(entrada)
    return saida


def _documento_do_plano(dados, doc_uid: str, pessoa_uid: str | None) -> dict:
    linha = dados.documentos.iloc[dados.indice_de(doc_uid)]

    def texto(campo):
        v = linha.get(campo)
        return None if v is None or pd.isna(v) else str(v)

    ano = linha.get("ano_isolado")
    referencia = (mod_links.referencia_crav(texto("reference_code"))
                  or mod_links.referencia_crav(texto("new_code"))
                  or mod_links.referencia_crav(texto("old_code")))
    return {
        "doc_uid": doc_uid,
        "reference_code": referencia,
        "new_code": texto("new_code"),
        "old_code": texto("old_code"),
        "ano": None if ano is None or pd.isna(ano) else int(ano),
        "tipologia": texto("tipologia"),
        "titulo_catalogo": texto("description"),
        "pasta": texto("folder"),
        "autoria_verificada": (autoria_verificada(dados, doc_uid, pessoa_uid)
                               if pessoa_uid else False),
        "pessoas": _pessoas_do_documento(dados, doc_uid),
        "links": {"digitarq": mod_links.digitarq_do_documento(linha)[0],
                  "resgate": mod_links.resgate()},
    }


def plano_de_coleta(dados, amostra: Amostra, consulta=None) -> dict:
    """O `plano_coleta.eins.json` do §7.4, no esquema `eins-plano-coleta/1` (A9)."""
    prov = mod_prov.proveniencia(dados, consulta, len(amostra), origem="plano_eins")
    app = prov["app_versao"]
    lotes = []
    for numero, uids in enumerate(amostra.lotes, start=1):
        estrato = (amostra.estrato_do_lote[numero - 1]
                   if numero - 1 < len(amostra.estrato_do_lote) else None)
        lotes.append({
            "lote": numero,
            "estrato": estrato,
            "documentos": [_documento_do_plano(dados, uid, amostra.pessoa_uid) for uid in uids],
        })
    return {
        "schema": ESQUEMA_EINS,
        "gerado_em": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "origem": {
            "app": "classificador-ahu-sul",
            "versao_app": app.get("versao") or "?",
            "commit": app.get("commit"),
            "corpus": prov["corpus"]["manifesto_sha256"],
            "camadas_ativas": prov["corpus"]["camadas_ativas"],
            "consulta": prov["consulta"],
            "amostra": {
                "tipo": amostra.tipo,
                "pessoa_uid": amostra.pessoa_uid,
                "estratos": amostra.estratos,
                "por_estrato": amostra.por_estrato,
                "maximo_por_ano": amostra.maximo_por_ano,
            },
            "semente": amostra.semente,
            "doi": prov["doi"],
            "citacao": prov["citacao"],
        },
        "lotes": lotes,
        "relatorio": amostra.relatorio,
    }


def plano_em_json(dados, amostra: Amostra, consulta=None) -> bytes:
    return json.dumps(plano_de_coleta(dados, amostra, consulta), ensure_ascii=False,
                      indent=2, default=str).encode("utf-8")
