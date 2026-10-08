"""links.py — WP 2.3: links para os acervos oficiais (spec §9).

Dois destinos:

* **DigitArq** (Arquivo Nacional da Torre do Tombo / ANTT): a busca aceita o `reference_code`
  (`PT/AHU/CU/021/0006/00390`) e leva à ficha de controle arquivístico.
* **Projeto Resgate** (Biblioteca Nacional): não há link por documento (Q8 da spec, extensão 5.3),
  por isso o link vai ao portal e o `new_code` é oferecido para copiar.

E, desde o WP 2.4, um terceiro: o **"Reportar erro"** da §7.1, uma *issue* do GitHub já preenchida
com o `pessoa_uid`, o dado e a fonte. Custo zero, e a correção volta ao pipeline pela coluna
`decisao_manual` do `revisao_biografias.csv`.

Nada aqui depende de Streamlit nem de rede: são só URLs montadas.
"""
from __future__ import annotations

import os
from urllib.parse import quote, urlencode

DIGITARQ = "https://digitarq.arquivos.pt/search?query={cota}&isAdvancedSearch=false"
RESGATE = "https://resgate.bn.gov.br/"

# O repositório onde as correções entram (spec §0). Em `AHU_REPO_GITHUB` para que um fork aponte
# para o seu próprio repositório sem editar o código.
REPOSITORIO = os.environ.get("AHU_REPO_GITHUB", "saulorrp/catalogo-ahu-cone-sul")
ETIQUETA_ISSUE = "correcao-de-dado"


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


def issue_de_pessoa(pessoa_uid: str, nome: str, dado: str | None = None,
                    fonte: str | None = None, repositorio: str | None = None) -> str:
    """"Reportar erro" da §7.1: uma *issue* do GitHub já preenchida.

    O corpo traz o `pessoa_uid`, o dado que o app está mostrando e a fonte, porque é isso que
    permite ao autor conferir a correção e levá-la ao `revisao_biografias.csv`.
    """
    corpo = [
        f"**Pessoa:** {nome}",
        f"**`pessoa_uid`:** `{pessoa_uid}`",
        f"**Dado exibido hoje:** {dado or '—'}",
        f"**Fonte do dado:** {fonte or '—'}",
        "",
        "**O que está errado:**",
        "",
        "**Qual é o dado correto, e com que fonte:**",
        "",
        "---",
        "Enviado pelo botão \"Reportar erro\" da aba Pessoas (spec §7.1). A correção entra no "
        "pipeline pela coluna `decisao_manual` do `revisao_biografias.csv`.",
    ]
    parametros = urlencode({
        "title": f"Correção de dado: {nome} ({pessoa_uid})",
        "body": "\n".join(corpo),
        "labels": ETIQUETA_ISSUE,
    })
    return f"https://github.com/{repositorio or REPOSITORIO}/issues/new?{parametros}"


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
