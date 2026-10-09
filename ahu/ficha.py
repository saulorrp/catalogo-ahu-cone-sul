"""ficha.py — WP 2.3: a ficha de um documento, montada como dado (spec §6.2, §6.3, §9).

A ficha é montada aqui, **sem Streamlit**, e o `app.py` só a desenha. É o que permite ao teste do
A1 percorrer as 7.051 fichas e procurar campo e nível proibidos, sem subir interface.

O que a ficha tem (§9):
  * título com ano, tipologia, primeira pessoa e pasta;
  * resumo do AHU, com os termos da busca marcados;
  * as cotas, para copiar, e os links do §9;
  * as pessoas, com origem e nível — **só ALTA e MÉDIA** (D2);
  * um bloco "Inferências (IA) — DeepSeek v3", recolhido por padrão (§6.2).

O que a ficha **nunca** tem:
  * `scribe_mediation_likely` — **nunca**, em lugar nenhum (D8). A D1 o mandava para o dossiê em
    PDF; a D8 retirou o dossiê e o campo do documento. Ele segue no dado, em
    `pdf_extra.parquet`, e é explicado — com a ressalva do preprint — só no "saiba mais";
  * biografia REJEITADA ou BAIXA, nem o nível delas (D2/A1). Isso já foi retirado em
    `app_dados/` pelo `construir_app.py`; aqui a regra é conferida de novo, de graça.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ahu import links

# D1/A1: nomes que não podem aparecer em ficha nenhuma.
CAMPOS_PROIBIDOS = ("scribe_mediation_likely",)
NIVEIS_PROIBIDOS = ("BAIXA", "REJEITADO", "IDENTIFICADO_SEM_LOCAL", "NAO_ENCONTRADO")

TITULO_IA = "Inferências (IA) — DeepSeek v3"


@dataclass
class Pessoa:
    pessoa_uid: str
    nome: str
    tipo: str
    nivel: str | None = None            # só ALTA ou MÉDIA chega aqui (D2)
    nascimento: str | None = None       # "Lisboa, Portugal (1660)"
    pais_epoca: str | None = None
    origem_do_dado: str | None = None   # "fonte" ou "wikidata"
    fonte: str | None = None            # URL
    rotulo: str | None = None           # §6.3, quando não há nascimento exibível

    @property
    def exibe_nascimento(self) -> bool:
        return self.nivel is not None


@dataclass
class Ficha:
    doc_uid: str
    titulo: str
    ano: int | None
    tipologia: str | None
    tipologia_a_revisar: bool
    pasta: str | None
    resumo: str
    resumo_marcado: str
    cotas: dict[str, str] = field(default_factory=dict)
    links: dict[str, str | None] = field(default_factory=dict)
    pessoas: list[Pessoa] = field(default_factory=list)
    destinatario: str | None = None
    inferencias: dict[str, object] = field(default_factory=dict)
    colecao: str | None = None

    def texto_visivel(self) -> str:
        """Tudo que a ficha mostra, num só texto — é o que o teste do A1 varre."""
        partes = [self.titulo, self.resumo, str(self.ano), self.tipologia or "", self.pasta or ""]
        partes += list(self.cotas.values())
        partes += [str(v) for v in self.links.values() if v]
        partes += [f"{k}: {v}" for k, v in self.inferencias.items()]
        for p in self.pessoas:
            partes += [p.nome, p.tipo, p.nivel or "", p.nascimento or "", p.pais_epoca or "",
                       p.origem_do_dado or "", p.fonte or "", p.rotulo or ""]
        return "\n".join(x for x in partes if x)


# ---------------------------------------------------------------------------
# Destaque dos termos
# ---------------------------------------------------------------------------

def marcar_termos(texto: str, termos: list[str], abre: str = "**", fecha: str = "**") -> str:
    """Marca no resumo as palavras que casaram, comparando pela forma normalizada (§5.5).

    A comparação é feita sobre a forma neutralizada da §5.1, mas o texto devolvido é o original:
    é assim que 'Mello' aparece marcado quando se buscou 'Melo'.
    """
    from ahu.texto import fonetica, normalizar_busca

    alvos = {fonetica(normalizar_busca(t)) for t in termos if t}
    if not texto or not alvos:
        return texto or ""

    def troca(m: re.Match) -> str:
        palavra = m.group(0)
        return f"{abre}{palavra}{fecha}" if fonetica(normalizar_busca(palavra)) in alvos else palavra

    return re.sub(r"\w+", troca, texto)


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------

def _valor(linha, campo):
    """Valor de uma coluna, com None para ausente, NaN e pd.NA."""
    if campo not in linha.index:
        return None
    v = linha[campo]
    return None if v is None or pd.isna(v) else v


def _vinculos_por_documento(dados) -> dict:
    """`doc_uid` -> as linhas de `doc_pessoa`, já na ordem de posição.

    Guardado no próprio `Dados` (é derivado, não artefato do §3.4). Antes, montar uma ficha fazia
    uma varredura de `doc_pessoa` **e** reconstruía o índice de `pessoas` a cada chamada: 25
    fichas numa página pagavam isso 25 vezes. Com o fragmento do R2 redesenhando a lista a cada
    clique, esse desperdício ficou visível.
    """
    cache = getattr(dados, "_vinculos_por_doc", None)
    if cache is not None:
        return cache
    agrupado = {uid: grupo.sort_values("posicao")
                for uid, grupo in dados.doc_pessoa.groupby("doc_uid", sort=False)}
    object.__setattr__(dados, "_vinculos_por_doc", agrupado)
    return agrupado


def _pessoas_indexadas(dados):
    cache = getattr(dados, "_pessoas_indexadas", None)
    if cache is None:
        cache = dados.pessoas.set_index("pessoa_uid")
        object.__setattr__(dados, "_pessoas_indexadas", cache)
    return cache


def _pessoas_do_documento(dados, doc_uid: str) -> list[Pessoa]:
    recorte = _vinculos_por_documento(dados).get(doc_uid)
    if recorte is None or recorte.empty:
        return []
    pessoas = _pessoas_indexadas(dados)
    saida = []
    for _, v in recorte.iterrows():
        uid = v["pessoa_uid"]
        nome = "(sem nome)" if pd.isna(v["nome"]) else str(v["nome"])
        if pd.isna(uid) or uid not in pessoas.index:
            saida.append(Pessoa(pessoa_uid="", nome=nome, tipo=str(v["tipo"])))
            continue
        p = pessoas.loc[uid]
        exibe = bool(p["nascimento_exibivel"])

        def campo(nome_campo, linha=p, mostrar=None):
            """Campo de nascimento: só sai quando a biografia é exibível (D2)."""
            mostrar = exibe if mostrar is None else mostrar
            if not mostrar or nome_campo not in linha.index or pd.isna(linha[nome_campo]):
                return None
            return str(linha[nome_campo])

        nascimento = None
        if exibe:
            local = ", ".join(x for x in (campo("cidade_nascimento"), campo("pais_atual")) if x)
            ano = campo("ano_nascimento")
            if ano:
                nascimento = f"{local} ({int(float(ano))})" if local else f"({int(float(ano))})"
            else:
                nascimento = local or None
        saida.append(Pessoa(
            pessoa_uid=str(uid),
            nome=campo("nome_exibicao", mostrar=True) or nome,
            tipo=str(v["tipo"]),
            nivel=campo("nivel"),
            nascimento=nascimento,
            pais_epoca=campo("pais_nascimento_epoca"),
            origem_do_dado=campo("origem_local"),
            fonte=campo("url_fonte"),
            rotulo=None if exibe else campo("rotulo", mostrar=True),
        ))
    return saida


def montar(dados, doc_uid: str, termos: list[str] | None = None) -> Ficha:
    """A ficha de um documento. `termos` vêm da `Resposta` da busca, para o destaque."""
    linha = dados.documentos.iloc[dados.indice_de(doc_uid)]
    pessoas = _pessoas_do_documento(dados, doc_uid)
    resumo = str(_valor(linha, "description") or "")
    ano = _valor(linha, "ano_isolado")
    tipologia = _valor(linha, "tipologia")
    pasta = _valor(linha, "folder")

    primeira = next((p.nome for p in pessoas if p.nome), None)
    titulo = " · ".join(str(x) for x in (
        int(ano) if ano is not None else "sem data", tipologia or "tipologia a revisar",
        primeira, pasta) if x)

    inferencias = {}
    for campo, etiqueta in (("sender_category", "Categoria social do remetente"),
                            ("vector", "Vetor"),
                            ("typology_impact", "Impacto da tipologia"),
                            ("vernacular_score", "SRSP"),
                            ("justificativa", "Justificativa")):
        valor = _valor(linha, campo)
        if valor is not None:
            inferencias[etiqueta] = int(valor) if campo == "vernacular_score" else str(valor)

    cotas = {etiqueta: str(_valor(linha, campo))
             for campo, etiqueta in (("reference_code", "Código de referência"),
                                     ("new_code", "Código atual"),
                                     ("old_code", "Código antigo"),
                                     ("document_id_and_date", "Item e data"))
             if _valor(linha, campo) is not None}

    return Ficha(
        doc_uid=doc_uid,
        titulo=titulo,
        ano=int(ano) if ano is not None else None,
        tipologia=str(tipologia) if tipologia is not None else None,
        tipologia_a_revisar=not bool(_valor(linha, "typology_status")),
        pasta=str(pasta) if pasta is not None else None,
        resumo=resumo,
        resumo_marcado=marcar_termos(resumo, termos or []),
        cotas=cotas,
        links=links.do_documento({c: _valor(linha, c) for c in ("reference_code", "new_code")}),
        pessoas=pessoas,
        destinatario=str(_valor(linha, "recipient_name")) if _valor(
            linha, "recipient_name") is not None else None,
        inferencias=inferencias,
        colecao=str(_valor(linha, "colecao")) if _valor(linha, "colecao") is not None else None,
    )
