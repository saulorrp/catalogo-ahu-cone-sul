"""pdf.py — os ajudantes de PDF que a lista do AHU (WP 3.3a) e o dossiê (WP 3.3) partilham.

Sem Streamlit. Aqui ficam as três coisas que os dois PDFs precisam e que não deviam existir em
duas cópias: as **fontes embutidas**, a **classe base com rodapé e numeração**, e o punhado de
formatadores de texto (`encurtar`, `data_extenso`).

**Fontes.** As duas famílias são DejaVu, de licença livre (`fontes/LICENSE-DejaVu.txt`), embutidas
no PDF para os acentos do português saírem certos em qualquer leitor:

* **`Sans`** (DejaVu Sans Condensed) — a lista de consulta do WP 3.3a. Condensada porque a tabela
  em A4 paisagem tem sete colunas, e é a família que tem o `☐` das caixas "consultado/fotografado".
* **`Serif`** (DejaVu Serif) — o dossiê do WP 3.3, que é texto corrido para ler. A §8.2 pede Noto
  Serif; a DejaVu Serif cumpre o que a spec quer dela (serifada, com os acentos do português) e já
  está no repositório, sem uma dependência nova. **Ela não tem o glifo `☐`**, e é por isso que as
  caixas de marcar são coisa da lista, não do dossiê.
"""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

PASTA_FONTES = Path(__file__).resolve().parent / "fontes"

FAMILIAS = {
    "Sans": {
        "": "DejaVuSansCondensed.ttf",
        "B": "DejaVuSansCondensed-Bold.ttf",
        "I": "DejaVuSansCondensed-Oblique.ttf",
    },
    "Serif": {
        "": "DejaVuSerif.ttf",
        "B": "DejaVuSerif-Bold.ttf",
        "I": "DejaVuSerif-Italic.ttf",
    },
}

MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro")

_RE_ORDINAL_DO_ITEM = re.compile(r"^\s*\d+\.\s*")


class ErroDeFonte(RuntimeError):
    """Uma fonte embutida não está no repositório: o PDF não pode sair sem acentos."""


def caminho_da_fonte(familia: str, estilo: str = "") -> Path:
    try:
        nome = FAMILIAS[familia][estilo]
    except KeyError:
        raise ErroDeFonte(
            f"Família ou estilo desconhecido: {familia!r}/{estilo!r}. "
            f"Há {', '.join(FAMILIAS)}."
        ) from None
    caminho = PASTA_FONTES / nome
    if not caminho.exists():
        raise ErroDeFonte(
            f"Fonte ausente: {caminho}. Sem ela o PDF sairia sem os acentos do português."
        )
    return caminho


class PDFBase(FPDF):
    """FPDF com rodapé de uma linha: texto à esquerda, página à direita."""

    familia = "Sans"
    rodape_texto = ""

    def footer(self) -> None:
        self.set_y(-10)
        self.set_font(self.familia, "", 7)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, self.rodape_texto, align="L")
        self.cell(0, 5, f"{self.page_no()}/{{nb}}", align="R")


def preparar(pdf: FPDF, familia: str = "Sans") -> FPDF:
    """Embute os três estilos de uma família. Chamar antes do primeiro `set_font`."""
    for estilo in ("", "B", "I"):
        pdf.add_font(familia, estilo, str(caminho_da_fonte(familia, estilo)))
    return pdf


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
