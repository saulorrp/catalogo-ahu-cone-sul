"""dossie.py — WP 3.3: o dossiê em PDF (spec §8.2, D1, A8).

Sem Streamlit. Devolve bytes, como o `ahu/lista_ahu.py`, e partilha com ele as fontes e os
formatadores de `ahu/pdf.py`.

O dossiê é o **único** lugar onde a inferência de mediação de escrivão
(`scribe_mediation_likely`) aparece — é a D1, e é por isso que ela vive sozinha em
`pdf_extra.parquet` e não em `documentos.parquet`. Aqui ela sai como **"Provável"** ou **"Pouco
provável"**, nunca como `true`/`false`, e **sempre** acompanhada da ressalva metodológica: a
regra geral do projeto é que, onde o código e o preprint divergirem, vale o preprint.

A ressalva é conservadora de propósito. O que se sabe do campo é o seguinte, e é só isso:

* ele é uma inferência do DeepSeek v3 **sobre o resumo do catálogo**, não sobre o manuscrito;
* a auditoria do preprint §2.1 mediu o **SRSP** (MAE 1,13 · r = 0,48 · viés +0,59 · N = 705 · um
  avaliador). Ela **não** mediu este campo binário em separado;
* portanto ele serve para **ordenar a leitura**, nunca como prova de que houve escrivão.

Estrutura (§8.2):

* **capa:** título, a consulta inteira, filtros, lente, nº de documentos, data, versão do app e do
  corpus, as camadas ativas, o SHA-256 do manifesto e a citação com o DOI;
* **uma seção por documento:** cotas e links, resumo do AHU, tipologia, pessoas com origem e
  nível (só ALTA e MÉDIA, D2), inferências da IA com a justificativa, e a mediação com a ressalva.

**Teto de 200 documentos por PDF** (`LIMITE`). Não é um número bonito: um dossiê de 500 fichas
levaria minutos no oitavo de núcleo do Community Cloud e nasceria ilegível. Acima do teto,
`gerar_pdf` erra alto e a interface avisa antes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ahu import exportar as mod_exportar
from ahu import ficha as mod_ficha
from ahu import pdf as mod_pdf

LIMITE = 200

TITULO_PADRAO = "Dossiê de triagem documental — AHU-Sul"
SUBTITULO = "Arquivo Histórico Ultramarino · coleção AHU-Sul · Projeto Resgate"

# D1: os dois rótulos que a §8.2 manda, no lugar de true/false.
ROTULOS_MEDIACAO = {True: "Provável", False: "Pouco provável"}
TITULO_MEDIACAO = "Mediação de escrivão (inferência de IA)"
RESSALVA_MEDIACAO = (
    "Ressalva metodológica (preprint §2). Esta é uma inferência do DeepSeek v3 feita sobre o "
    "resumo do catálogo, não sobre o manuscrito: ela indica a probabilidade de a escrita ter "
    "passado por um escrivão local (o \"filtro do escrivão\"), e não um fato estabelecido. A "
    "auditoria do preprint §2.1 mediu o SRSP (MAE 1,13 · r = 0,48 · viés +0,59 · N = 705 · um "
    "avaliador); este campo binário não foi auditado em separado. Use-o para ordenar a leitura, "
    "nunca como prova — a verificação depende do exame do documento original. É por isso que ele "
    "aparece só aqui, e não na interface, nos filtros, no CSV nem no JSON (D1)."
)

NOTA_IA = ("Inferências do DeepSeek v3, não revisadas documento a documento. O SRSP é um inteiro "
           "de 1 a 9; diferenças de ±1 ponto não são distinguíveis (preprint §2.1).")


class ErroDeDossie(ValueError):
    """Pedido impossível: nenhum documento, ou mais que o teto."""


@dataclass
class Secao:
    """O que entra na seção de um documento. Montada antes de desenhar, para poder ser testada."""

    doc_uid: str
    titulo: str
    cotas: dict
    links: dict
    resumo: str
    pessoas: list
    destinatario: str | None
    inferencias: dict
    mediacao: str | None        # "Provável" / "Pouco provável" / None quando não há dado


def _mediacao(dados, doc_uid: str) -> str | None:
    """O rótulo da D1, lido de `pdf_extra.parquet` — o único artefato que tem o campo."""
    extra = dados.pdf_extra
    campo = "scribe_mediation_likely"
    if campo not in extra.columns:
        return None
    recorte = extra.loc[extra["doc_uid"] == doc_uid, campo]
    if recorte.empty or pd.isna(recorte.iloc[0]):
        return None
    return ROTULOS_MEDIACAO[bool(recorte.iloc[0])]


def montar_secoes(dados, doc_uids) -> list[Secao]:
    """Uma seção por documento, na ordem dada. Reusa a `ahu/ficha.py` (D2 já aplicada lá)."""
    doc_uids = list(doc_uids)
    if not doc_uids:
        raise ErroDeDossie("O dossiê precisa de ao menos um documento.")
    if len(doc_uids) > LIMITE:
        raise ErroDeDossie(
            f"São {len(doc_uids)} documentos e o dossiê cabe {LIMITE}. Reduza a seleção, ou "
            f"gere mais de um dossiê."
        )
    secoes = []
    for uid in doc_uids:
        f = mod_ficha.montar(dados, uid)
        secoes.append(Secao(
            doc_uid=uid, titulo=f.titulo, cotas=dict(f.cotas), links=dict(f.links),
            resumo=f.resumo, pessoas=list(f.pessoas), destinatario=f.destinatario,
            inferencias=dict(f.inferencias), mediacao=_mediacao(dados, uid),
        ))
    return secoes


# ---------------------------------------------------------------------------
# O PDF
# ---------------------------------------------------------------------------

def _capa(documento, prov: dict, quantos: int, titulo: str) -> None:
    """A capa do §8.2: consulta, filtros, lente, contagem, data, versões e citação com o DOI."""
    documento.add_page()
    documento.set_font("Serif", "B", 17)
    documento.multi_cell(0, 9, titulo, new_x="LMARGIN", new_y="NEXT")
    documento.set_font("Serif", "", 10)
    documento.set_text_color(90, 90, 90)
    documento.multi_cell(0, 5.5, SUBTITULO, new_x="LMARGIN", new_y="NEXT")
    documento.set_text_color(0, 0, 0)
    documento.ln(4)

    consulta = prov.get("consulta") or {}
    app = prov.get("app_versao") or {}
    corpus = prov.get("corpus") or {}
    camadas = "; ".join(
        f"{papel}: {c['camada']}" + (f" (v{c['versao']})" if c.get("versao") else "")
        for papel, c in (corpus.get("camadas_ativas") or {}).items())
    filtros = consulta.get("filtros_ativos") or {}
    filtros_texto = "; ".join(f"{k} = {v}" for k, v in filtros.items()) or "nenhum"
    manifesto = corpus.get("manifesto_sha256") or "—"

    campos = (
        ("Documentos neste dossiê", str(quantos)),
        ("Consulta", consulta.get("texto") or "(sem texto: o corpus com os filtros)"),
        ("Modo e rigor", f"{consulta.get('modo', '—')} · rigor {consulta.get('rigor', '—')}"),
        ("Lente metodológica", consulta.get("lente") or "nenhuma"),
        ("Filtros ativos", filtros_texto),
        ("Ordenação", consulta.get("ordenacao") or "—"),
        ("Semente", str(consulta.get("semente")) if consulta.get("semente") else "—"),
        ("Gerado em", prov.get("gerado_em") or date.today().isoformat()),
        ("Versão do app", f"{app.get('versao') or '?'}"
                          + (f" · commit {app['commit']}" if app.get("commit") else "")),
        ("Artefatos do corpus", str(corpus.get("versao_artefatos") or "—")),
        ("Camadas ativas", camadas or "—"),
        ("Modelo de embeddings", (corpus.get("modelo_embeddings") or {}).get("nome") or "—"),
        ("Manifesto (SHA-256)", manifesto),
    )
    for rotulo, valor in campos:
        documento.set_font("Serif", "B", 9)
        documento.cell(45, 5, rotulo + ":")
        documento.set_font("Serif", "", 9)
        documento.multi_cell(0, 5, str(valor), new_x="LMARGIN", new_y="NEXT")
    documento.ln(4)

    documento.set_font("Serif", "B", 10)
    documento.cell(0, 6, "Como citar", new_x="LMARGIN", new_y="NEXT")
    documento.set_font("Serif", "", 9)
    doi = prov.get("doi") or {}
    citacao = prov.get("citacao") or mod_exportar.CITACAO
    if doi.get("preprint"):
        citacao += f" Base teórica: doi:{doi['preprint']}."
    if doi.get("zenodo"):
        citacao += f" Dados: doi:{doi['zenodo']}."
    documento.multi_cell(0, 5, citacao, new_x="LMARGIN", new_y="NEXT")
    documento.ln(2)
    documento.set_font("Serif", "I", 8.5)
    documento.multi_cell(0, 4.6, prov.get("credito") or mod_exportar.CREDITO,
                         new_x="LMARGIN", new_y="NEXT")
    documento.ln(3)
    documento.set_font("Serif", "I", 8.5)
    documento.set_text_color(90, 90, 90)
    documento.multi_cell(0, 4.6, RESSALVA_MEDIACAO, new_x="LMARGIN", new_y="NEXT")
    documento.set_text_color(0, 0, 0)


def _secao(documento, secao: Secao, numero: int) -> None:
    if documento.get_y() > documento.h - 60:
        documento.add_page()
    documento.set_font("Serif", "B", 11)
    documento.multi_cell(0, 5.8, f"{numero}. {secao.titulo}", new_x="LMARGIN", new_y="NEXT")
    documento.ln(0.5)

    documento.set_font("Serif", "", 8.5)
    for etiqueta, valor in secao.cotas.items():
        documento.set_font("Serif", "B", 8.5)
        documento.cell(42, 4.6, etiqueta + ":")
        documento.set_font("Serif", "", 8.5)
        documento.multi_cell(0, 4.6, str(valor), new_x="LMARGIN", new_y="NEXT")
    destinos = []
    if secao.links.get("digitarq"):
        destinos.append(("DigitArq", secao.links["digitarq"]))
    if secao.links.get("resgate"):
        destinos.append(("Projeto Resgate", secao.links["resgate"]))
    if destinos:
        documento.set_font("Serif", "B", 8.5)
        documento.cell(42, 4.6, "Links:")
        documento.set_font("Serif", "", 8.5)
        for i, (nome, url) in enumerate(destinos):
            documento.cell(documento.get_string_width(nome) + 2, 4.6, nome, link=url)
            if i < len(destinos) - 1:
                documento.cell(4, 4.6, "·")
        documento.ln(4.6)

    if secao.destinatario:
        documento.set_font("Serif", "B", 8.5)
        documento.cell(42, 4.6, "Destinatário:")
        documento.set_font("Serif", "", 8.5)
        documento.multi_cell(0, 4.6, secao.destinatario, new_x="LMARGIN", new_y="NEXT")

    documento.ln(1)
    documento.set_font("Serif", "", 9.5)
    documento.multi_cell(0, 4.9, secao.resumo or "(sem resumo no catálogo)",
                         new_x="LMARGIN", new_y="NEXT")

    if secao.pessoas:
        documento.ln(1)
        documento.set_font("Serif", "B", 8.5)
        documento.cell(0, 4.6, "Pessoas (remetentes)", new_x="LMARGIN", new_y="NEXT")
        documento.set_font("Serif", "", 8.5)
        for p in secao.pessoas:
            linha = f"— {p.nome} ({p.tipo})"
            if p.exibe_nascimento:
                if p.nascimento:
                    linha += f" · nasc. {p.nascimento}"
                if p.pais_epoca:
                    linha += f" · à época: {p.pais_epoca}"
                linha += f" · confiança {p.nivel}"
                if p.origem_do_dado:
                    linha += f" · origem do dado: {p.origem_do_dado}"
            elif p.rotulo:
                linha += f" · {p.rotulo}"
            documento.multi_cell(0, 4.4, linha, new_x="LMARGIN", new_y="NEXT")

    if secao.inferencias:
        documento.ln(1)
        documento.set_font("Serif", "B", 8.5)
        documento.cell(0, 4.6, mod_ficha.TITULO_IA, new_x="LMARGIN", new_y="NEXT")
        documento.set_font("Serif", "", 8.5)
        for etiqueta, valor in secao.inferencias.items():
            documento.multi_cell(0, 4.4, f"{etiqueta}: {valor}", new_x="LMARGIN", new_y="NEXT")

    # D1: aqui, e só aqui.
    if secao.mediacao:
        documento.ln(1)
        documento.set_font("Serif", "B", 8.5)
        documento.cell(0, 4.6, f"{TITULO_MEDIACAO}: {secao.mediacao}",
                       new_x="LMARGIN", new_y="NEXT")
        documento.set_font("Serif", "I", 7.5)
        documento.set_text_color(90, 90, 90)
        documento.multi_cell(0, 4, "Ver a ressalva metodológica na capa.",
                             new_x="LMARGIN", new_y="NEXT")
        documento.set_text_color(0, 0, 0)

    documento.ln(3)
    y = documento.get_y()
    documento.set_draw_color(200, 200, 200)
    documento.line(documento.l_margin, y, documento.w - documento.r_margin, y)
    documento.ln(3)


def gerar_pdf(dados, doc_uids, consulta=None, prov: dict | None = None,
              titulo: str | None = None) -> bytes:
    """O dossiê do §8.2: capa com proveniência e uma seção por documento (A8)."""
    doc_uids = list(doc_uids)
    secoes = montar_secoes(dados, doc_uids)
    prov = prov if prov is not None else mod_exportar.proveniencia(
        dados, consulta, len(doc_uids), origem="dossie")

    documento = mod_pdf.PDFBase(orientation="P", unit="mm", format="A4")
    documento.familia = "Serif"
    documento.set_margins(18, 16, 18)
    documento.set_auto_page_break(True, margin=16)
    mod_pdf.preparar(documento, "Serif")
    documento.rodape_texto = (
        f"Dossiê gerado em {date.today().strftime('%d/%m/%Y')} · "
        f"{len(secoes)} documento{'s' if len(secoes) != 1 else ''} · classificador AHU-Sul")
    documento.set_title(titulo or TITULO_PADRAO)
    documento.set_author((prov.get("app_versao") or {}).get("versao") or "")

    _capa(documento, prov, len(secoes), titulo or TITULO_PADRAO)

    documento.add_page()
    documento.set_font("Serif", "B", 12)
    documento.cell(0, 7, "Documentos", new_x="LMARGIN", new_y="NEXT")
    documento.set_font("Serif", "I", 8)
    documento.set_text_color(90, 90, 90)
    documento.multi_cell(0, 4.4, NOTA_IA, new_x="LMARGIN", new_y="NEXT")
    documento.set_text_color(0, 0, 0)
    documento.ln(3)
    for numero, secao in enumerate(secoes, start=1):
        _secao(documento, secao, numero)
    return bytes(documento.output())
