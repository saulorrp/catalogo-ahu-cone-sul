"""lista_ahu.py — WP 3.3a: a lista de documentos para consulta presencial no AHU (PDF).

Portado do `lista_ahu.py` do pacote antigo, com uma mudança de fundo: o catálogo não é mais um
JSON à parte, é **`app_dados/`** (AGENTS.md 3). O resto do comportamento é o mesmo, porque é o que
o autor leva ao arquivo: A4 paisagem, uma linha por documento, agrupável, com código de
referência CRAV, link do DigitArq, data, tipologia, remetente, resumo do AHU e duas caixas para
marcar à mão ("consultado", "fotografado").

Duas entradas, as mesmas do painel antigo:

* **a seleção** do WP 3.1, por `lista_de_documentos`, que vira texto na caixa;
* **uma lista colada**, uma cota por linha, em qualquer das formas usuais. Linha começando com
  `#` abre um grupo (um autor, um dia de consulta); depois de `|` vem nota livre, e um `http…` do
  Resgate vira a coluna *Resgate*.

**Cotas de fora do corpus funcionam**, e isso é de propósito: o código de referência e o link do
DigitArq saem da própria cota, pela regra do preprint §1, mesmo que o documento não seja AHU-Sul.
O que falta, nesse caso, é só o que vem do catálogo — data, tipologia, remetente e resumo.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from fpdf.fonts import FontFace

from ahu import pdf as mod_pdf
from ahu.links import digitarq, referencia_crav

TITULO_PADRAO = "Documentos para consulta presencial"
SUBTITULO = "Arquivo Histórico Ultramarino (AHU/DGLAB) · Lisboa"
MARCACOES = "☐ consultado\n☐ fotografado"

# Soma 273 mm: A4 paisagem (297) menos as margens de 12.
LARGURAS = (9, 48, 52, 108, 17, 17, 22)
CABECALHO_TABELA = ("Nº", "Cota (Resgate) / Código de referência",
                    "Data · Tipologia · Remetente", "Resumo do catálogo (AHU)",
                    "DigitArq", "Resgate", "Consulta")
LIMITE_RESUMO = 520
LIMITE_REMETENTE = 140


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
# O PDF
# ---------------------------------------------------------------------------

def _meta_do_documento(dados, linha_i: int, quantas: int) -> tuple[str, str]:
    """As duas colunas de texto: (data · tipologia · remetente, resumo)."""
    linha = dados.documentos.iloc[linha_i]

    def valor(campo):
        v = linha.get(campo)
        return "" if v is None or pd.isna(v) else str(v)

    meta = "\n".join(x for x in (
        mod_pdf.data_extenso(valor("document_id_and_date")),
        valor("tipologia"),
        mod_pdf.encurtar(valor("sender_name"), LIMITE_REMETENTE),
    ) if x)
    resumo = mod_pdf.encurtar(valor("description"), LIMITE_RESUMO)
    if quantas > 1:
        resumo += f"\n[{quantas} documentos no catálogo com esta cota]"
    return meta, resumo


def gerar_pdf(grupos: list[Grupo], dados, cabecalho: dict | None = None) -> bytes:
    """O PDF da lista. `cabecalho`: titulo, pesquisador, instituicao, projeto, datas_consulta,
    observacao, rodape."""
    cab = cabecalho or {}
    documento = mod_pdf.PDFBase(orientation="L", unit="mm", format="A4")
    documento.familia = "Sans"
    documento.set_margins(12, 12, 12)
    documento.set_auto_page_break(True, margin=14)
    mod_pdf.preparar(documento, "Sans")

    total = sum(len(g.itens) for g in grupos)
    documento.rodape_texto = cab.get("rodape") or (
        f"Lista gerada em {date.today().strftime('%d/%m/%Y')} · códigos de referência e links "
        "DigitArq calculados a partir das cotas")
    documento.set_title(cab.get("titulo") or TITULO_PADRAO)
    documento.set_author(cab.get("pesquisador") or "")
    documento.add_page()

    documento.set_font("Sans", "B", 14)
    documento.cell(0, 8, cab.get("titulo") or TITULO_PADRAO, new_x="LMARGIN", new_y="NEXT")
    documento.set_font("Sans", "", 9.5)
    documento.cell(0, 5, SUBTITULO, new_x="LMARGIN", new_y="NEXT")
    documento.ln(1.5)
    campos = (("Pesquisador(a)", cab.get("pesquisador")),
              ("Instituição", cab.get("instituicao")),
              ("Projeto", cab.get("projeto")),
              ("Consulta agendada (CRAV)", cab.get("datas_consulta")),
              ("Total", f"{total} documento{'s' if total != 1 else ''}"))
    for rotulo, valor in campos:
        if not valor:
            continue
        documento.set_font("Sans", "B", 9)
        documento.cell(50, 4.8, rotulo + ":")
        documento.set_font("Sans", "", 9)
        documento.multi_cell(0, 4.8, str(valor), new_x="LMARGIN", new_y="NEXT")
    if cab.get("observacao"):
        documento.ln(1)
        documento.set_font("Sans", "I", 8.5)
        documento.multi_cell(0, 4.3, cab["observacao"], new_x="LMARGIN", new_y="NEXT")
    documento.ln(3)

    estilo_cabecalho = FontFace(emphasis="BOLD", fill_color=(232, 232, 232), size_pt=8)
    numero = 0
    for grupo in grupos:
        if grupo.titulo:
            if documento.get_y() > documento.h - 45:
                documento.add_page()
            documento.set_font("Sans", "B", 10.5)
            documento.cell(0, 7, f"{grupo.titulo}  ({len(grupo.itens)})",
                           new_x="LMARGIN", new_y="NEXT")
        documento.set_font("Sans", "", 7.6)
        with documento.table(
                col_widths=LARGURAS,
                text_align=("CENTER", "LEFT", "LEFT", "LEFT", "CENTER", "CENTER", "LEFT"),
                line_height=3.6, headings_style=estilo_cabecalho, padding=1.4,
                repeat_headings=1, borders_layout="SINGLE_TOP_LINE") as tabela:
            linha_tabela = tabela.row()
            for celula in CABECALHO_TABELA:
                linha_tabela.cell(celula)
            for item in grupo.itens:
                numero += 1
                if item.linhas:
                    doc = dados.documentos.iloc[item.linhas[0]]
                    bruto = doc.get("new_code")
                    cota = (item.cota_digitada if bruto is None or pd.isna(bruto)
                            else str(bruto))
                    meta, resumo = _meta_do_documento(dados, item.linhas[0], len(item.linhas))
                else:
                    cota = item.cota_digitada
                    meta, resumo = "não consta no catálogo AHU-Sul", ""
                if item.nota:
                    resumo = (resumo + "\n" if resumo else "") + f"Nota: {item.nota}"
                linha_tabela = tabela.row()
                linha_tabela.cell(str(numero))
                linha_tabela.cell(f"{cota}\n{item.referencia or '— (cota não reconhecida)'}")
                linha_tabela.cell(meta)
                linha_tabela.cell(resumo or "—")
                if item.referencia:
                    linha_tabela.cell("abrir ↗", link=digitarq(item.referencia))
                else:
                    linha_tabela.cell("—")
                if item.link_resgate:
                    linha_tabela.cell("abrir ↗", link=item.link_resgate)
                else:
                    linha_tabela.cell("—")
                linha_tabela.cell(MARCACOES)
        documento.ln(3)
    return bytes(documento.output())
