"""links.py — WP 2.3: links para os acervos oficiais (spec §9).

Dois destinos:

* **DigitArq** (Arquivo Nacional da Torre do Tombo / ANTT): a busca aceita o `reference_code`
  (`PT/AHU/CU/021/0006/00390`) e leva à ficha de controle arquivístico.
* **Projeto Resgate** (Biblioteca Nacional): não há link por documento (Q8 da spec, extensão 5.3),
  por isso o link vai ao portal e o `new_code` é oferecido para copiar.

Nada aqui depende de Streamlit nem de rede: são só URLs montadas.
"""
from __future__ import annotations

from urllib.parse import quote

DIGITARQ = "https://digitarq.arquivos.pt/search?query={cota}&isAdvancedSearch=false"
RESGATE = "https://resgate.bn.gov.br/"


def digitarq(reference_code: str | None) -> str | None:
    """Busca no DigitArq pelo código de referência. None quando não há cota."""
    cota = (reference_code or "").strip()
    if not cota or cota.upper() == "UNFORMATTED_CODE":
        return None
    return DIGITARQ.format(cota=quote(cota, safe=""))


def resgate(new_code: str | None = None) -> str:
    """Portal do Projeto Resgate. Ainda sem link por documento (Q8)."""
    del new_code  # reservado para quando houver link direto (extensão 5.3)
    return RESGATE


def do_documento(documento) -> dict[str, str | None]:
    """Os links de uma linha de `documentos.parquet`."""
    import pandas as pd

    def valor(campo):
        v = documento.get(campo) if hasattr(documento, "get") else None
        return None if v is None or pd.isna(v) else str(v)

    return {
        "digitarq": digitarq(valor("reference_code")),
        "resgate": resgate(valor("new_code")),
    }
