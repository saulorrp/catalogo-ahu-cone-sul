"""pessoas.py — WP 2.4: a aba Pessoas e o painel diatópico (spec §7.1, §9).

Sem Streamlit. Tudo aqui é função pura sobre `app_dados/`, como em `ahu/ficha.py`: é o que
permite ao teste do A2 procurar "Lavradio" por nome, variante e título e cobrar **uma** pessoa
com os seus 163 documentos, sem subir interface.

Três coisas vivem neste módulo:

* **a lista de pessoas** (`listar`), ordenável e filtrável pelos mesmos filtros de origem da
  §6.1, e a **lista separada de coletivos e instituições** (`listar_coletivos`), que a §7.1 pede
  à parte porque "Conselho Ultramarino" não é uma pessoa e não tem biografia;
* **a página da pessoa** (`pagina`): linha do tempo, tipologias, destinatários, o link que aplica
  o filtro de pessoa na aba Busca e o link de *issue* do GitHub já preenchido;
* **o painel diatópico** (`painel_diatopico`), que conta **por documento ou por pessoa** (§9).

D2 vale aqui como na ficha: cidade, país, ano, nível e fonte de nascimento só saem para quem tem
`nascimento_exibivel`. Para os outros sai o `rotulo` da §6.3 e mais nada. A regra já foi aplicada
em `app_dados/` pelo `construir_app.py`; aqui ela é conferida de novo, de graça.

Coletivos e instituições não têm `pessoa_uid` (eles não passam pelo pipeline de biografias): eles
existem só como nome em `doc_pessoa.parquet`. A lista deles é agrupada pela forma normalizada do
nome, para que "Oficiais da Câmara do Rio de Janeiro" e "oficiais da Câmara do Rio de Janeiro"
sejam uma linha só; as grafias encontradas ficam visíveis na linha. Nada além de caixa e acento é
unificado — juntar "Mesa de Inspeção" com "Mesa da Inspeção" seria decidir no código o que é a
mesma instituição, e isso é trabalho do pipeline, não da interface (AGENTS.md 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ahu import links
from ahu.texto import chave_titulo, normalizar_busca

# §7.1: a lista própria. Os papéis vêm da coluna `tipo` de `doc_pessoa.parquet`.
TIPOS_COLETIVOS = ("instituicao", "coletivo")
# Remetentes que não são entidade nenhuma: "Unknown" e nomes incompletos. Não entram em lista
# alguma; aparecem só como contagem, para o total fechar com o corpus.
TIPOS_SEM_ENTIDADE = ("nao_identificado", "pessoa_nome_incompleto")

ORDENS = ("documentos", "nome", "inicio", "fim")
ROTULOS_ORDEM = {
    "documentos": "Nº de documentos",
    "nome": "Nome",
    "inicio": "Início da atividade",
    "fim": "Fim da atividade",
}
POR = ("documento", "pessoa")

NOTA_DIATOPICA = (
    "Conta só remetentes com nascimento exibível (confiança ALTA ou MÉDIA, D2): "
    "{exibiveis} de {total} pessoas. Um documento com remetentes de países diferentes "
    "conta uma vez em cada país."
)
SEM_ORIGEM = "sem origem conhecida"


# ---------------------------------------------------------------------------
# Pessoas
# ---------------------------------------------------------------------------

@dataclass
class PessoaNaLista:
    """Uma linha da lista de pessoas (§7.1). Os campos de nascimento respeitam a D2."""

    pessoa_uid: str
    nome: str
    tipo: str
    variantes: list[str] = field(default_factory=list)
    titulos: list[str] = field(default_factory=list)
    cargos: list[str] = field(default_factory=list)
    ano_min: int | None = None
    ano_max: int | None = None
    n_documentos: int = 0
    # --- nascimento: só quando exibível (D2) ---
    exibe_nascimento: bool = False
    nivel: str | None = None
    cidade_nascimento: str | None = None
    pais_atual: str | None = None
    pais_epoca: str | None = None
    ano_nascimento: int | None = None
    origem_do_dado: str | None = None
    fonte: str | None = None
    rotulo: str | None = None       # §6.3, quando não há nascimento exibível

    @property
    def periodo(self) -> str:
        if self.ano_min is None and self.ano_max is None:
            return "sem data"
        if self.ano_min == self.ano_max:
            return str(self.ano_min)
        return f"{self.ano_min or '?'}–{self.ano_max or '?'}"

    @property
    def nascimento(self) -> str | None:
        """Lisboa, Portugal (1729) — ou None quando a biografia não é exibível (D2)."""
        if not self.exibe_nascimento:
            return None
        local = ", ".join(x for x in (self.cidade_nascimento, self.pais_atual) if x)
        if self.ano_nascimento is not None:
            return f"{local} ({self.ano_nascimento})" if local else f"({self.ano_nascimento})"
        return local or None

    def chaves_de_busca(self) -> set[str]:
        """Nome, variantes e títulos normalizados — por aqui "Lavradio" acha a pessoa (A2).

        Os títulos entram também pela `chave_titulo`, que tira o ordinal e as partículas
        ("9º conde da Feira" -> "conde feira"), a mesma chave que dá o `pessoa_uid` (§3.3).
        """
        chaves = {normalizar_busca(x) for x in (self.nome, *self.variantes, *self.titulos)}
        chaves |= {chave_titulo(t) for t in self.titulos}
        return {c for c in chaves if c}


def _texto(valor) -> str | None:
    if valor is None:
        return None
    try:
        if pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return str(valor)


def _inteiro(valor) -> int | None:
    if valor is None or pd.isna(valor):
        return None
    return int(valor)


def _lista(valor) -> list[str]:
    """Coluna de lista do parquet, que vem como ndarray, lista ou NaN."""
    if valor is None or isinstance(valor, float):
        return []
    return [str(v) for v in valor if v is not None and str(v)]


def _da_linha(linha) -> PessoaNaLista:
    exibe = bool(linha["nascimento_exibivel"])

    def campo(nome, mostrar=None):
        if (exibe if mostrar is None else mostrar) is False:
            return None
        return _texto(linha.get(nome))

    nome = (_texto(linha.get("nome_exibicao")) or _texto(linha.get("nome_principal"))
            or "(sem nome)")
    return PessoaNaLista(
        pessoa_uid=str(linha["pessoa_uid"]),
        nome=nome,
        tipo=str(linha.get("tipo") or "pessoa"),
        variantes=_lista(linha.get("variantes_nome")),
        titulos=_lista(linha.get("titulos")),
        cargos=_lista(linha.get("cargos")),
        ano_min=_inteiro(linha.get("ano_min")),
        ano_max=_inteiro(linha.get("ano_max")),
        n_documentos=_inteiro(linha.get("n_documentos")) or 0,
        exibe_nascimento=exibe,
        nivel=campo("nivel"),
        cidade_nascimento=campo("cidade_nascimento"),
        pais_atual=campo("pais_atual"),
        pais_epoca=campo("pais_nascimento_epoca"),
        ano_nascimento=_inteiro(linha.get("ano_nascimento")) if exibe else None,
        origem_do_dado=campo("origem_local"),
        fonte=campo("url_fonte"),
        rotulo=None if exibe else campo("rotulo", mostrar=True),
    )


def todas(dados) -> list[PessoaNaLista]:
    """Todas as pessoas de `pessoas.parquet`, sem filtro e sem ordem imposta."""
    return [_da_linha(linha) for _, linha in dados.pessoas.iterrows()]


def ordenar(pessoas: list[PessoaNaLista], ordem: str = "documentos") -> list[PessoaNaLista]:
    if ordem not in ORDENS:
        raise ValueError(f"Ordem desconhecida: {ordem!r}. Use uma de {', '.join(ORDENS)}.")
    chaves = {
        "documentos": lambda p: (-p.n_documentos, normalizar_busca(p.nome)),
        "nome": lambda p: normalizar_busca(p.nome),
        # Pessoa sem data vai para o fim nas duas ordens de atividade.
        "inicio": lambda p: (p.ano_min is None, p.ano_min or 0, normalizar_busca(p.nome)),
        "fim": lambda p: (p.ano_max is None, -(p.ano_max or 0), normalizar_busca(p.nome)),
    }
    return sorted(pessoas, key=chaves[ordem])


def listar(dados, texto: str | None = None, paises: list[str] | None = None,
           cidades: list[str] | None = None, so_confianca_alta: bool = False,
           ordem: str = "documentos") -> list[PessoaNaLista]:
    """A lista de pessoas da §7.1: filtrável pelos mesmos filtros de origem e ordenável.

    `texto` casa por nome, variante ou título (A2). Os filtros de origem só alcançam quem tem
    nascimento exibível (D2), como em `ahu/filtros.py`.
    """
    saida = todas(dados)
    if paises or cidades or so_confianca_alta:
        saida = [p for p in saida if p.exibe_nascimento
                 and (not so_confianca_alta or p.nivel == "ALTA")
                 and (not paises or p.pais_atual in paises)
                 and (not cidades or p.cidade_nascimento in cidades)]
    if texto and texto.strip():
        saida = _casar(saida, texto)
    return ordenar(saida, ordem)


def _casar(pessoas: list[PessoaNaLista], texto: str) -> list[PessoaNaLista]:
    """Pessoas cujo nome, variante ou título contém o texto buscado (§7.1, A2)."""
    alvo = normalizar_busca(texto)
    if not alvo:
        return pessoas
    return [p for p in pessoas if any(alvo in c for c in p.chaves_de_busca())]


def encontrar(dados, texto: str) -> list[PessoaNaLista]:
    """Busca de pessoa por nome, variante ou título. É o caminho do A2.

    Ordena por nº de documentos, e por isso "Lavradio" devolve uma única pessoa, com 163.
    """
    return ordenar(_casar(todas(dados), texto), "documentos")


def tabela(pessoas: list[PessoaNaLista]) -> pd.DataFrame:
    """A lista em forma de tabela, para a interface mostrar e ordenar. Sem campo proibido."""
    return pd.DataFrame([{
        "Nome": p.nome,
        "Nº de documentos": p.n_documentos,
        "Período de atividade": p.periodo,
        "Títulos": "; ".join(p.titulos),
        "Cargos": "; ".join(p.cargos),
        "Variantes": "; ".join(p.variantes),
        "Nascimento": p.nascimento or "—",
        "Confiança": p.nivel or "—",
        "Origem do dado": p.origem_do_dado or "—",
        "pessoa_uid": p.pessoa_uid,
    } for p in pessoas])


# ---------------------------------------------------------------------------
# Coletivos e instituições (§7.1) — lista própria, separada das pessoas
# ---------------------------------------------------------------------------

@dataclass
class Coletivo:
    nome: str
    tipo: str
    n_documentos: int
    ano_min: int | None = None
    ano_max: int | None = None
    grafias: list[str] = field(default_factory=list)

    @property
    def periodo(self) -> str:
        if self.ano_min is None and self.ano_max is None:
            return "sem data"
        if self.ano_min == self.ano_max:
            return str(self.ano_min)
        return f"{self.ano_min or '?'}–{self.ano_max or '?'}"


def _vinculos_com_ano(dados) -> pd.DataFrame:
    """`doc_pessoa` com o ano do documento ao lado. É a base da linha do tempo e do painel."""
    docs = dados.documentos
    anos = pd.DataFrame({
        "doc_uid": docs["doc_uid"].astype(str),
        "ano": pd.to_numeric(docs["ano_isolado"], errors="coerce"),
        "tipologia": docs["tipologia"] if "tipologia" in docs.columns else None,
        "destinatario": docs["recipient_name"] if "recipient_name" in docs.columns else None,
    })
    return dados.doc_pessoa.merge(anos, on="doc_uid", how="left")


ORDENS_COLETIVOS = {
    "documentos": lambda c: (-c.n_documentos, normalizar_busca(c.nome)),
    "nome": lambda c: normalizar_busca(c.nome),
    "inicio": lambda c: (c.ano_min is None, c.ano_min or 0, normalizar_busca(c.nome)),
    "fim": lambda c: (c.ano_max is None, -(c.ano_max or 0), normalizar_busca(c.nome)),
}


def listar_coletivos(dados, texto: str | None = None,
                     ordem: str = "documentos") -> list[Coletivo]:
    """Coletivos e instituições, agrupados pela forma normalizada do nome."""
    if ordem not in ORDENS_COLETIVOS:
        raise ValueError(
            f"Ordem desconhecida: {ordem!r}. Use uma de {', '.join(ORDENS_COLETIVOS)}.")
    vinculos = _vinculos_com_ano(dados)
    recorte = vinculos[vinculos["tipo"].isin(TIPOS_COLETIVOS)].copy()
    if recorte.empty:
        return []
    recorte["nome"] = recorte["nome"].fillna("(sem nome)").astype(str)
    recorte["chave"] = recorte["nome"].map(normalizar_busca)

    saida = []
    for _, grupo in recorte.groupby(["chave", "tipo"], sort=False):
        grafias = grupo["nome"].value_counts()
        anos = grupo["ano"].dropna()
        saida.append(Coletivo(
            nome=str(grafias.index[0]),      # a grafia mais frequente representa o grupo
            tipo=str(grupo["tipo"].iloc[0]),
            n_documentos=int(grupo["doc_uid"].nunique()),
            ano_min=int(anos.min()) if not anos.empty else None,
            ano_max=int(anos.max()) if not anos.empty else None,
            grafias=[str(g) for g in grafias.index[1:]],
        ))
    if texto and texto.strip():
        alvo = normalizar_busca(texto)
        saida = [c for c in saida
                 if alvo in normalizar_busca(c.nome)
                 or any(alvo in normalizar_busca(g) for g in c.grafias)]
    return sorted(saida, key=ORDENS_COLETIVOS[ordem])


def tabela_de_coletivos(coletivos: list[Coletivo]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Nome": c.nome,
        "Tipo": "Instituição" if c.tipo == "instituicao" else "Coletivo",
        "Nº de documentos": c.n_documentos,
        "Período": c.periodo,
        "Outras grafias": "; ".join(c.grafias) or "—",
    } for c in coletivos])


def contagem_de_remetentes(dados) -> dict[str, int]:
    """Quantos vínculos de remetente há em cada papel. O total fecha com `doc_pessoa`."""
    contagem = dados.doc_pessoa["tipo"].value_counts()
    return {str(k): int(v) for k, v in contagem.items()}


# ---------------------------------------------------------------------------
# Página da pessoa (§7.1)
# ---------------------------------------------------------------------------

@dataclass
class PaginaPessoa:
    pessoa: PessoaNaLista
    documentos: pd.DataFrame            # doc_uid, ano, tipologia, destinatario, em ordem
    linha_do_tempo: pd.DataFrame        # ano, documentos
    tipologias: list[tuple[str, int]] = field(default_factory=list)
    destinatarios: list[tuple[str, int]] = field(default_factory=list)
    sem_data: int = 0
    url_documentos: str = ""
    url_issue: str = ""


def url_dos_documentos(pessoa_uid: str) -> str:
    """O link que abre a aba Busca já com o filtro desta pessoa (§7.1, "Ver documentos").

    É a mesma URL versionada do `ahu/consulta.py`: um link relativo, que recarrega o app na
    primeira aba com o filtro aplicado — e que por isso também pode ser copiado e compartilhado
    (§8.3, A6). Montado aqui, e não no `app.py`, para que o teste o verifique sem interface.
    """
    from ahu.consulta import Consulta, para_query
    from ahu.filtros import Filtros

    return para_query(Consulta(filtros=Filtros(pessoas=[pessoa_uid])))


def pagina(dados, pessoa_uid: str) -> PaginaPessoa:
    """A página da pessoa: linha do tempo, tipologias, destinatários e os dois links."""
    recorte = dados.pessoas[dados.pessoas["pessoa_uid"] == pessoa_uid]
    if recorte.empty:
        raise KeyError(f"pessoa_uid desconhecido: {pessoa_uid}")
    pessoa = _da_linha(recorte.iloc[0])

    vinculos = _vinculos_com_ano(dados)
    meus = vinculos[vinculos["pessoa_uid"] == pessoa_uid].copy()
    documentos = meus[["doc_uid", "ano", "tipologia", "destinatario", "autoria_verificada"]]
    documentos = documentos.sort_values(["ano", "doc_uid"], na_position="last")

    anos = documentos["ano"].dropna().astype(int)
    if anos.empty:
        linha = pd.DataFrame({"ano": pd.Series(dtype="int64"),
                              "documentos": pd.Series(dtype="int64")})
    else:
        linha = (anos.value_counts().sort_index()
                 .rename_axis("ano").reset_index(name="documentos"))

    return PaginaPessoa(
        pessoa=pessoa,
        documentos=documentos.reset_index(drop=True),
        linha_do_tempo=linha,
        tipologias=[(str(k), int(v)) for k, v in documentos["tipologia"].dropna()
                    .value_counts().items()],
        destinatarios=[(str(k), int(v)) for k, v in documentos["destinatario"].dropna()
                       .value_counts().items()],
        sem_data=int(documentos["ano"].isna().sum()),
        url_documentos=url_dos_documentos(pessoa_uid),
        url_issue=links.issue_de_pessoa(pessoa_uid, pessoa.nome,
                                        dado=_dado_para_a_issue(pessoa), fonte=pessoa.fonte),
    )


def _dado_para_a_issue(pessoa: PessoaNaLista) -> str:
    """O dado que a *issue* já traz preenchido (§7.1): o que o app está mostrando hoje."""
    if pessoa.exibe_nascimento:
        partes = [f"nascimento: {pessoa.nascimento or '—'}", f"confiança: {pessoa.nivel}"]
        if pessoa.pais_epoca:
            partes.append(f"país à época: {pessoa.pais_epoca}")
        if pessoa.origem_do_dado:
            partes.append(f"origem do dado: {pessoa.origem_do_dado}")
        return " · ".join(partes)
    return f"sem nascimento exibível · rótulo: {pessoa.rotulo or '—'}"


# ---------------------------------------------------------------------------
# Painel diatópico (§9) — conta por documento ou por pessoa
# ---------------------------------------------------------------------------

@dataclass
class PainelDiatopico:
    por: str
    paises: pd.DataFrame            # pais, contagem
    decadas: pd.DataFrame           # decada, pais, contagem
    total: int = 0
    sem_origem: int = 0
    nota: str = ""


def painel_diatopico(dados, por: str = "documento") -> PainelDiatopico:
    """Contagem por país de nascimento do remetente, por documento ou por pessoa (§9).

    * **por documento:** quantos documentos têm ao menos um remetente nascido naquele país. Um
      documento com remetentes de dois países conta uma vez em cada um — por isso a soma das
      barras pode passar o total de documentos, e o `total` é o nº de documentos distintos.
    * **por pessoa:** quantas pessoas distintas nasceram naquele país. Na quebra por década,
      uma pessoa ativa em duas décadas conta uma vez em cada.

    D2: só entram pessoas com nascimento exibível. Quem não tem fica em `sem_origem`, e é esse
    número que impede a leitura do painel como se o corpus inteiro tivesse origem conhecida.
    """
    if por not in POR:
        raise ValueError(f"Contagem desconhecida: {por!r}. Use uma de {', '.join(POR)}.")

    pessoas = dados.pessoas
    exibiveis = pessoas[pessoas["nascimento_exibivel"].astype(bool)]
    origem = pd.DataFrame({
        "pessoa_uid": exibiveis["pessoa_uid"].astype(str),
        "pais": exibiveis["pais_atual"].fillna(SEM_ORIGEM).astype(str),
    })

    vinculos = _vinculos_com_ano(dados)
    com_origem = vinculos.merge(origem, on="pessoa_uid", how="inner")
    com_origem["decada"] = (com_origem["ano"] // 10 * 10).astype("Int64")

    if por == "documento":
        unidade = "doc_uid"
        # O total é o corpus inteiro, não só os documentos que têm vínculo de remetente: um
        # documento sem remetente identificado também é um documento sem origem conhecida.
        total = int(len(dados.documentos))
        sem_origem = total - int(com_origem["doc_uid"].nunique())
    else:
        unidade = "pessoa_uid"
        total = int(len(pessoas))
        sem_origem = total - int(len(exibiveis))

    if com_origem.empty:
        paises = pd.DataFrame({"pais": pd.Series(dtype="object"),
                               "contagem": pd.Series(dtype="int64")})
        decadas = pd.DataFrame({"decada": pd.Series(dtype="Int64"),
                                "pais": pd.Series(dtype="object"),
                                "contagem": pd.Series(dtype="int64")})
    else:
        paises = (com_origem.drop_duplicates(["pais", unidade])["pais"].value_counts()
                  .rename_axis("pais").reset_index(name="contagem"))
        decadas = (com_origem.dropna(subset=["decada"])
                   .drop_duplicates(["decada", "pais", unidade])
                   .groupby(["decada", "pais"], observed=True).size()
                   .reset_index(name="contagem").sort_values(["decada", "pais"])
                   .reset_index(drop=True))

    return PainelDiatopico(
        por=por, paises=paises, decadas=decadas, total=total, sem_origem=sem_origem,
        nota=NOTA_DIATOPICA.format(exibiveis=len(exibiveis), total=len(pessoas)),
    )


def ordem_dos_paises(painel: PainelDiatopico) -> list[str]:
    """Países do maior para o menor — a ordem das cores e da legenda do gráfico."""
    return [str(p) for p in painel.paises["pais"].tolist()]
