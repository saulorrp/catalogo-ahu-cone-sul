"""lista_ahu.py — a lista de documentos para consulta no AHU (plano v3.1, R3).

É o **único** PDF que o app gera, e o único arquivo que ele entrega: a D7 retirou o CSV, o JSON e
o dossiê. Ele existe para uma coisa, descrita pelo autor: levar ao arquivo uma folha por onde
conferir e fotografar.

O que ele tem, e nada mais:

| Nº | Ano | Cota | Resumo | DigitArq | caixa |
|---|---|---|---|---|---|

* **Nº** liga a linha à fotografia;
* **Cota** traz a cota do Resgate na primeira linha e o código `PT/AHU/CU/…` abaixo, em corpo
  menor;
* **Resumo** sai inteiro, com quebra de linha — não cortado;
* **DigitArq** é um "abrir" clicável, não a URL crua;
* a **caixa** é desenhada, para marcar à mão.

Sem explicação, sem proveniência, sem tipologia, sem remetente, sem SRSP e sem a inferência de
mediação de escrivão (D8). No alto só o título e a data; no pé, uma linha de 8 pt.

Duas entradas, as mesmas de sempre:

* **a seleção** (`pdf_da_selecao`), na ordem em que o pesquisador marcou;
* **uma lista colada** (`ler_lista`), uma cota por linha, em qualquer das formas usuais. Linha
  começando com `#` abre um grupo — um autor, um dia de consulta — e vira uma faixa de título no
  PDF; depois de `|` vem nota livre, e um `http…` do Resgate entra como link na cota.

**Cotas de fora do corpus funcionam**, e isso é de propósito: o código de referência e o link do
DigitArq saem da própria cota, pela regra do preprint §1, mesmo que o documento não seja AHU-Sul.
O que falta, nesse caso, é só o resumo, que vem do catálogo.

O catálogo é **`app_dados/`** (AGENTS.md 3), nunca um JSON à parte.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ahu import pdf as mod_pdf
from ahu.links import digitarq, digitarq_do_documento, referencia_crav
from ahu.links import limpar_cota as _limpar_cota

TITULO = "Lista para consulta no AHU"

# Medidas da página (A4 paisagem, 297 × 210, margens de 12).
MARGEM = 12
LARGURA_UTIL = 297 - 2 * MARGEM         # 273 mm
# Nº · Ano · Cota · Resumo · DigitArq · caixa — soma 273.
COLUNAS = (("Nº", 10), ("Ano", 14), ("Cota", 60), ("Resumo", 152),
           ("DigitArq", 18), ("", 19))
CORPO_PT = 9
ENTRELINHA = 1.35
ALTURA_LINHA = CORPO_PT * ENTRELINHA / 2.8346      # 9 pt × 1,35, em mm ≈ 4,29
RODAPE = "Classificador AHU-Sul"


def _dobrar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


# ---------------------------------------------------------------------------
# O índice de cotas, sobre `app_dados/`
# ---------------------------------------------------------------------------

@dataclass
class Indice:
    """Cota -> linhas de `documentos.parquet`, pelo código de referência e pela cota antiga."""

    por_referencia: dict[str, list[int]] = field(default_factory=dict)
    por_antiga: dict[str, list[int]] = field(default_factory=dict)

    def buscar(self, cota: str) -> tuple[str | None, list[int]]:
        """(código de referência, linhas). O código sai da cota mesmo sem documento no corpus."""
        referencia = referencia_crav(cota)
        if referencia:
            return referencia, self.por_referencia.get(referencia, [])
        linhas = self.por_antiga.get(_dobrar(cota), [])
        return (None, linhas) if not linhas else (self._referencia_da_linha(linhas[0]), linhas)

    _dados: object | None = None

    def _referencia_da_linha(self, linha: int) -> str | None:
        docs = self._dados.documentos.iloc[linha]
        for campo in ("reference_code", "new_code"):
            valor = docs.get(campo)
            if valor is not None and not pd.isna(valor):
                referencia = referencia_crav(str(valor))
                if referencia:
                    return referencia
        return None


def indice_de_cotas(dados) -> Indice:
    """Construído uma vez e guardado no próprio `Dados` (é derivado, não artefato do §3.4)."""
    cache = getattr(dados, "_indice_lista_ahu", None)
    if cache is not None:
        return cache
    indice = Indice(_dados=dados)
    docs = dados.documentos
    for campo in ("reference_code", "new_code"):
        if campo not in docs.columns:
            continue
        for i, valor in enumerate(docs[campo]):
            if valor is None or pd.isna(valor):
                continue
            referencia = referencia_crav(str(valor))
            if referencia is None:
                continue
            linhas = indice.por_referencia.setdefault(referencia, [])
            if i not in linhas:
                linhas.append(i)
    if "old_code" in docs.columns:
        for i, valor in enumerate(docs["old_code"]):
            if valor is None or pd.isna(valor):
                continue
            indice.por_antiga.setdefault(_dobrar(str(valor)), []).append(i)
    object.__setattr__(dados, "_indice_lista_ahu", indice)
    return indice


def referencia_do_documento(dados, doc_uid: str) -> str | None:
    """O código de referência CRAV de um documento do corpus, pelas suas três cotas."""
    linha = dados.documentos.iloc[dados.indice_de(doc_uid)]
    for campo in ("reference_code", "new_code", "old_code"):
        valor = linha.get(campo)
        if valor is None or pd.isna(valor):
            continue
        referencia = referencia_crav(str(valor))
        if referencia:
            return referencia
    return None


# ---------------------------------------------------------------------------
# A lista
# ---------------------------------------------------------------------------

@dataclass
class Item:
    cota_digitada: str
    nota: str = ""
    link_resgate: str | None = None
    referencia: str | None = None
    linhas: list[int] = field(default_factory=list)


@dataclass
class Grupo:
    titulo: str | None
    itens: list[Item] = field(default_factory=list)


def ler_lista(texto: str, dados) -> list[Grupo]:
    """Lê a lista colada. `#` abre grupo; `|` separa nota e link do Resgate."""
    indice = indice_de_cotas(dados)
    grupos = [Grupo(None)]
    for bruta in (texto or "").splitlines():
        linha = bruta.strip()
        if not linha:
            continue
        if linha.startswith("#"):
            grupos.append(Grupo(linha.lstrip("#").strip()))
            continue
        partes = [p.strip() for p in linha.split("|")]
        cota, resto = partes[0], partes[1:]
        links = [p for p in resto if p.lower().startswith("http")]
        nota = " · ".join(p for p in resto if p and not p.lower().startswith("http"))
        referencia, achadas = indice.buscar(cota)
        grupos[-1].itens.append(
            Item(cota, nota, links[0] if links else None, referencia, achadas))
    return [g for g in grupos if g.itens]


def lista_de_documentos(dados, doc_uids, agrupar_por: str | None = None) -> str:
    """O texto da caixa, a partir da seleção do WP 3.1.

    `agrupar_por` é uma coluna de `documentos.parquet` (`folder`, `tipologia`, `ano_isolado`…):
    cada valor diferente abre um grupo, na ordem em que aparece.
    """
    docs = dados.documentos
    saida, atual = [], object()
    for uid in doc_uids:
        linha = docs.iloc[dados.indice_de(uid)]
        if agrupar_por:
            valor = linha.get(agrupar_por)
            grupo = "—" if valor is None or pd.isna(valor) else str(valor)
            if grupo != atual:
                saida.append(f"# {grupo}")
                atual = grupo
        cota = None
        for campo in ("new_code", "reference_code", "old_code"):
            v = linha.get(campo)
            if v is not None and not pd.isna(v) and str(v).upper() != "UNFORMATTED_CODE":
                cota = str(v)
                break
        saida.append(cota or str(linha["doc_uid"]))
    return "\n".join(saida)


def grupos_da_selecao(dados, selecao) -> list[Grupo]:
    """A seleção como um grupo único, na ordem em que foi marcada (R2).

    Não passa pelo `ler_lista`: os documentos já são do corpus, e reescrevê-los como texto para
    reconhecê-los de volta seria dar a volta ao mundo para atravessar a rua.
    """
    itens = []
    for linha in selecao.linhas(dados):
        itens.append(Item(
            cota_digitada=linha["cota"] or linha["referencia"],
            referencia=linha["referencia"] or None,
            linhas=[dados.indice_de(linha["uid"])],
        ))
    return [Grupo(None, itens)] if itens else []


def pdf_da_selecao(dados, selecao, cabecalho: dict | None = None) -> bytes:
    """O PDF da lista de consulta, a partir da seleção (R2/R3)."""
    return gerar_pdf(grupos_da_selecao(dados, selecao), dados, cabecalho)


def relatorio(grupos: list[Grupo]) -> dict:
    """O que o painel mostra antes de gerar o PDF: quantos, quantos no catálogo, quantos sem cota."""
    itens = [i for g in grupos for i in g.itens]
    sem_referencia = [i.cota_digitada for i in itens if not i.referencia]
    return {
        "total": len(itens),
        "no_catalogo": sum(1 for i in itens if i.linhas),
        "com_digitarq": len(itens) - len(sem_referencia),
        "sem_referencia": sem_referencia,
        "grupos": len(grupos),
    }


# ---------------------------------------------------------------------------
# O PDF (R3)
# ---------------------------------------------------------------------------

def _dados_da_linha(dados, item: Item, numero: int) -> dict:
    """O que vai na linha do PDF: número, ano, cota, referência, resumo e link."""
    ano, resumo = "", ""
    link = digitarq(item.referencia) if item.referencia else None
    if item.linhas:
        linha = dados.documentos.iloc[item.linhas[0]]
        link = digitarq_do_documento(linha)[0] or link

        def valor(campo):
            v = linha.get(campo)
            return "" if v is None or pd.isna(v) else v

        bruto = valor("ano_isolado")
        ano = "" if bruto == "" else str(int(bruto))
        resumo = str(valor("description") or "")
        if len(item.linhas) > 1:
            resumo += f" [{len(item.linhas)} documentos no catálogo com esta cota]"
    else:
        resumo = "não consta no catálogo AHU-Sul"
    if item.nota:
        resumo = (resumo + " " if resumo else "") + f"— {item.nota}"
    return {
        "n": numero,
        "ano": ano,
        "cota": _limpar_cota(item.cota_digitada),
        "referencia": item.referencia or "",
        "resumo": resumo,
        "digitarq": link,
        "resgate": item.link_resgate,
    }


def _x_das_colunas() -> list[float]:
    """A borda esquerda de cada coluna, em mm."""
    bordas, x = [], MARGEM
    for _, largura in COLUNAS:
        bordas.append(x)
        x += largura
    return bordas


def _cabecalho_da_tabela(pdf) -> None:
    """A faixa com os nomes das colunas, repetida em cada página."""
    y = pdf.get_y()
    pdf.set_font(mod_pdf.FAMILIA, "B", 8)
    pdf.set_fill_color(236, 232, 221)
    pdf.rect(MARGEM, y, LARGURA_UTIL, ALTURA_LINHA + 1.6, style="F")
    for (nome, largura), x in zip(COLUNAS, _x_das_colunas()):
        if not nome:
            continue
        pdf.set_xy(x + 1.2, y + 0.8)
        pdf.cell(largura - 2.4, ALTURA_LINHA, nome)
    pdf.set_y(y + ALTURA_LINHA + 2.6)
    pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)


def _altura_da_linha(pdf, dados_linha: dict) -> float:
    """A altura de que a linha precisa: o maior entre o resumo e o bloco da cota."""
    largura_resumo = dict(COLUNAS)["Resumo"] - 2.4
    largura_cota = dict(COLUNAS)["Cota"] - 2.4

    pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)
    alto_resumo = pdf.multi_cell(w=largura_resumo, h=ALTURA_LINHA,
                                 text=dados_linha["resumo"] or "—",
                                 dry_run=True, output="HEIGHT")
    alto_cota = pdf.multi_cell(w=largura_cota, h=ALTURA_LINHA,
                               text=dados_linha["cota"] or "—", align="L",
                               dry_run=True, output="HEIGHT")
    if dados_linha["referencia"]:
        pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT - 1.5)
        alto_cota += pdf.multi_cell(w=largura_cota, h=ALTURA_LINHA - 0.6,
                                    text=dados_linha["referencia"], align="L",
                                    dry_run=True, output="HEIGHT")
        pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)
    return max(alto_resumo, alto_cota, ALTURA_LINHA)


def _desenhar_linha(pdf, dados_linha: dict, altura: float) -> None:
    """Uma linha da lista, com a caixa desenhada e o "abrir" clicável."""
    larguras = dict(COLUNAS)
    x_numero, x_ano, x_cota, x_resumo, x_link, x_caixa = _x_das_colunas()
    y = pdf.get_y()

    pdf.set_font(mod_pdf.FAMILIA, "B", CORPO_PT)
    pdf.set_xy(x_numero + 1.2, y)
    pdf.cell(larguras["Nº"] - 2.4, ALTURA_LINHA, str(dados_linha["n"]))

    pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)
    pdf.set_xy(x_ano + 1.2, y)
    pdf.cell(larguras["Ano"] - 2.4, ALTURA_LINHA, dados_linha["ano"] or "—")

    # Cota: a do Resgate em cima, o código de referência abaixo, em corpo menor.
    pdf.set_xy(x_cota + 1.2, y)
    # Alinhada à esquerda, nunca justificada: a cota é um código, e esticá-la abriria buracos
    # entre "Cx." e o número.
    pdf.multi_cell(larguras["Cota"] - 2.4, ALTURA_LINHA, dados_linha["cota"] or "—",
                   align="L", new_x="LEFT", new_y="NEXT", link=dados_linha["resgate"] or "")
    if dados_linha["referencia"]:
        pdf.set_x(x_cota + 1.2)
        pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT - 1.5)
        pdf.set_text_color(105, 105, 105)
        pdf.multi_cell(larguras["Cota"] - 2.4, ALTURA_LINHA - 0.6, dados_linha["referencia"],
                       align="L", new_x="LEFT", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)

    pdf.set_xy(x_resumo + 1.2, y)
    pdf.multi_cell(larguras["Resumo"] - 2.4, ALTURA_LINHA, dados_linha["resumo"] or "—",
                   new_x="LEFT", new_y="NEXT")

    if dados_linha["digitarq"]:
        pdf.set_xy(x_link + 1.2, y)
        pdf.set_text_color(45, 85, 125)
        pdf.cell(larguras["DigitArq"] - 2.4, ALTURA_LINHA, "abrir",
                 link=dados_linha["digitarq"])
        pdf.set_text_color(0, 0, 0)

    mod_pdf.caixa_de_marcar(pdf, x_caixa + 3, y + 0.6)

    # A régua fina que separa as linhas.
    base = y + altura + 1.2
    pdf.set_draw_color(205, 201, 180)
    pdf.set_line_width(0.1)
    pdf.line(MARGEM, base, 297 - MARGEM, base)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_y(base + 1.4)


def _faixa_de_grupo(pdf, titulo: str, quantos: int) -> None:
    pdf.ln(1.5)
    pdf.set_font(mod_pdf.FAMILIA, "B", CORPO_PT + 1)
    pdf.cell(0, ALTURA_LINHA + 1, f"{titulo}  ({quantos})", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(mod_pdf.FAMILIA, "", CORPO_PT)
    pdf.ln(0.8)


def gerar_pdf(grupos: list[Grupo], dados, cabecalho: dict | None = None) -> bytes:
    """O PDF da lista (R3): A4 paisagem, título, data, e uma linha por documento.

    `cabecalho` aceita só `titulo`; o resto do que o painel antigo pedia — pesquisador,
    instituição, projeto, datas, observação — saiu, porque o autor pediu uma folha sem cerimônia.

    A lista é montada linha a linha, e não pela tabela do fpdf2, por duas razões: para a caixa de
    marcar poder ser **desenhada** (a Noto Serif não tem glifo de caixa) e para **nenhuma linha se
    partir no meio do resumo** — a altura de cada linha é medida antes, e se não couber na página
    ela inteira vai para a seguinte.
    """
    cab = cabecalho or {}
    pdf = mod_pdf.PDFBase(orientation="L", unit="mm", format="A4")
    pdf.set_margins(MARGEM, MARGEM, MARGEM)
    pdf.set_auto_page_break(False)          # as quebras são decididas linha a linha
    mod_pdf.preparar(pdf)
    pdf.rodape_texto = RODAPE
    pdf.set_title(cab.get("titulo") or TITULO)
    pdf.add_page()

    pdf.set_font(mod_pdf.FAMILIA, "B", 15)
    pdf.cell(0, 9, cab.get("titulo") or TITULO, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(mod_pdf.FAMILIA, "", 9)
    pdf.set_text_color(105, 105, 105)
    pdf.cell(0, 5, date.today().strftime("%d/%m/%Y"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)
    _cabecalho_da_tabela(pdf)

    limite = 210 - 14          # a altura da página menos a faixa do rodapé
    numero = 0
    for grupo in grupos:
        if grupo.titulo:
            if pdf.get_y() + 3 * ALTURA_LINHA > limite:
                pdf.add_page()
                _cabecalho_da_tabela(pdf)
            _faixa_de_grupo(pdf, grupo.titulo, len(grupo.itens))
        for item in grupo.itens:
            numero += 1
            linha = _dados_da_linha(dados, item, numero)
            altura = _altura_da_linha(pdf, linha)
            if pdf.get_y() + altura + 2 > limite:
                pdf.add_page()
                _cabecalho_da_tabela(pdf)
            # Salvaguarda: um resumo mais alto que a página inteira não cabe em linha nenhuma. No
            # corpus de hoje o maior ocupa 86 mm contra 166 mm úteis, mas um catálogo futuro pode
            # trazer um maior — e aí é melhor deixar o fpdf2 quebrar a linha do que perder texto
            # para fora da folha.
            parte = altura + MARGEM + 2 > limite
            pdf.set_auto_page_break(parte, margin=14)
            _desenhar_linha(pdf, linha, altura)
            pdf.set_auto_page_break(False)
    return bytes(pdf.output())
