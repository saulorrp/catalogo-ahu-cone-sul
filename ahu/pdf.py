"""pdf.py — fonte embutida e ajudantes do único PDF que o app gera (plano v3.1, R3).

Sem Streamlit. Aqui ficam a **fonte**, a classe base com o rodapé e dois formatadores de texto.

**Uma família só: Noto Serif** (SIL OFL, `fontes/LICENSE-NotoSerif-OFL.txt`), embutida no PDF para
os acentos do português saírem certos em qualquer leitor. Ela tem os acentos todos e o ⸿ (U+2E3F),
o capitulum da identidade do EINS.

**O que ela não tem, e o que isso decidiu.** Esta Noto Serif não traz nenhum glifo de caixa —
nem `☐` (U+2610), nem `□` (U+25A1). A caixa de marcar do R3 é, por isso, **desenhada** com
`rect`, e não escrita: a 9 pt um retângulo de traço fino sai mais limpo que qualquer glifo, e não
depende de o leitor ter uma fonte de reserva. É também a razão pela qual a lista é montada linha a
linha, em vez de pela tabela do fpdf2 — assim se desenha dentro da célula e se garante que
**nenhuma linha se parte no meio do resumo**.

As DejaVu, que serviam ao dossiê do WP 3.3, saíram com a D7.
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

PASTA_FONTES = Path(__file__).resolve().parent / "fontes"

FAMILIA = "Serif"
ARQUIVOS = {
    "": "NotoSerif-Regular.ttf",
    "B": "NotoSerif-Bold.ttf",
    "I": "NotoSerif-Italic.ttf",
}

MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro")

_RE_ORDINAL_DO_ITEM = re.compile(r"^\s*\d+\.\s*")


class ErroDeFonte(RuntimeError):
    """A fonte embutida não está no repositório: o PDF não pode sair sem acentos."""


def caminho_da_fonte(estilo: str = "") -> Path:
    try:
        nome = ARQUIVOS[estilo]
    except KeyError:
        raise ErroDeFonte(
            f"Estilo desconhecido: {estilo!r}. Há {', '.join(repr(e) for e in ARQUIVOS)}."
        ) from None
    caminho = PASTA_FONTES / nome
    if not caminho.exists():
        raise ErroDeFonte(
            f"Fonte ausente: {caminho}. Sem ela o PDF sairia sem os acentos do português."
        )
    return caminho


class PDFBase(FPDF):
    """FPDF com rodapé de uma linha: texto à esquerda, página à direita."""

    rodape_texto = ""

    def footer(self) -> None:
        self.set_y(-10)
        self.set_font(FAMILIA, "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, self.rodape_texto, align="L")
        self.cell(0, 5, f"página {self.page_no()}/{{nb}}", align="R")
        self.set_text_color(0, 0, 0)


def preparar(pdf: FPDF) -> FPDF:
    """Embute os três estilos. Chamar antes do primeiro `set_font`."""
    for estilo in ARQUIVOS:
        pdf.add_font(FAMILIA, estilo, str(caminho_da_fonte(estilo)))
    return pdf


def caixa_de_marcar(pdf: FPDF, x: float, y: float, lado: float = 3.2) -> None:
    """A caixa para marcar à mão, **desenhada**: a Noto Serif não tem glifo de caixa."""
    pdf.set_draw_color(90, 90, 90)
    pdf.set_line_width(0.2)
    pdf.rect(x, y, lado, lado)
    pdf.set_draw_color(0, 0, 0)


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------

def encurtar(texto: str | None, limite: int) -> str:
    """Corta no espaço anterior ao limite e fecha com reticências."""
    texto = (texto or "").strip()
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0] + "…"


def data_extenso(item_e_data: str | None) -> str:
    """`'7964. 1787, março, 28, Montevidéu'` -> `'28 mar. 1787 · Montevidéu'`.

    É o campo `document_id_and_date` do catálogo, que vem como uma frase com vírgulas: o número
    do item na frente, depois ano, mês, dia e local. Quando não dá para ler, devolve "sem data"
    em vez de inventar.
    """
    bruto = _RE_ORDINAL_DO_ITEM.sub("", str(item_e_data or ""))
    partes = [p.strip(" []") for p in bruto.split(",")]
    if not partes or not partes[0]:
        return "sem data"
    ano = partes[0]
    mes = partes[1].lower() if len(partes) > 1 else ""
    dia = partes[2] if len(partes) > 2 else ""
    local = ", ".join(partes[3:]) if len(partes) > 3 else ""
    abreviados = [m[:3] for m in MESES]
    mes_abrev = mes[:3] + "." if mes[:3] in abreviados else mes
    data = " ".join(p for p in (dia, mes_abrev, ano) if p)
    return data + (f" · {local}" if local else "")
