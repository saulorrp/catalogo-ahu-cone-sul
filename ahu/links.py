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

**`referencia_crav` (WP 3.3a)** é a conversão de cota em código de referência CRAV, trazida do
`lista_ahu.py` do pacote antigo: ela aceita as duas famílias de cota (Resgate e CRAV) e devolve
sempre `PT/AHU/CU/<série>/<caixa>/<documento>`, com os zeros à esquerda. Desde a aba Conversor
a regra mora em `ahu/conversor.py`, e esta função só a chama. Ela **não** é a
`busca.chave_cota`, e a diferença importa: a `chave_cota` é uma chave de *lookup* dentro do
corpus, que joga os zeros fora para poder casar as três grafias; esta monta o código oficial que o
DigitArq entende e que vai impresso na lista de consulta.
"""
from __future__ import annotations

import os
from urllib.parse import quote, urlencode

DIGITARQ = "https://digitarq.arquivos.pt/search?query={cota}&isAdvancedSearch=false"
RESGATE = "https://resgate.bn.gov.br/"


def limpar_cota(texto) -> str:
    """A cota como se imprime: sem a pontuação solta que o catálogo deixa no fim.

    O catálogo traz `AHU_ACL_CU_003, Cx. 4, D. 360.]` e `… D. 390.`; o ponto e o colchete são
    resto de extração, e na folha que vai ao arquivo eles só confundem quem procura a cota.
    """
    import pandas as pd

    if texto is None or (not isinstance(texto, str) and pd.isna(texto)):
        return ""
    return str(texto).strip().rstrip(" .];,")


def referencia_crav(texto: str | None) -> str | None:
    """Converte uma cota no código de referência: `PT/AHU/CU/005/0135/10451`.

    A regra é a do `ahu.conversor` (níveis 1 e 2): aceita a referência, a cota do Resgate e as
    cotas antigas de numeração igual (Castro e Almeida, Mendes Gouveia). Devolve None quando o
    texto não contém uma cota assim — o caso do `UNFORMATTED_CODE` e das cotas antigas comuns
    (`AHU-São Paulo, cx. 26, doc. 8`), cuja numeração não é a atual e por isso não tem código
    CRAV derivável.
    """
    if not texto:
        return None
    from ahu.conversor import para_referencia, reconhecer

    cota = reconhecer(str(texto))
    return para_referencia(cota) if cota else None

# O repositório onde as correções entram (spec §0). Em `AHU_REPO_GITHUB` para que um fork aponte
# para o seu próprio repositório sem editar o código.
REPOSITORIO = os.environ.get("AHU_REPO_GITHUB", "saulorrp/catalogo-ahu-cone-sul")
ETIQUETA_ISSUE = "correcao-de-dado"


def digitarq(cota: str | None) -> str | None:
    """Busca no DigitArq pelo código de referência. None quando não há cota reconhecível.

    Aceita qualquer das duas famílias de cota e normaliza pela `referencia_crav`: é o que faz o
    link funcionar também para os documentos cujo `reference_code` é `UNFORMATTED_CODE` mas cujo
    `new_code` do Resgate está inteiro.
    """
    referencia = referencia_crav(cota)
    if referencia is None:
        return None
    return DIGITARQ.format(cota=quote(referencia, safe=""))


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
        # A cota do Resgate entra como reserva: 13 documentos têm `UNFORMATTED_CODE` no
        # `reference_code` e um `new_code` inteiro, e eles também merecem link (WP 3.3a).
        "digitarq": digitarq(valor("reference_code")) or digitarq(valor("new_code")),
        "resgate": resgate(valor("new_code")),
    }
