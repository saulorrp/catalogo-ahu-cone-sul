"""cotas.py — acertos de cota feitos na construção do app_dados.

Dois acertos de cota, mecânicos e contados no manifesto:

* **`new_code` sem cabeçalho de página.** Na extração de alguns catálogos (Reino, Goiás, Piauí) o
  cabeçalho da página seguinte grudou na cota: `AHU_CU_Reino, Cx. 133, pasta 10. INSTRUMENTO DE
  PESQUISA DOS DOCUMENTOS…`. O cabeçalho sai; a cota fica.
* **`reference_code` refeito pelo conversor atual.** Parte da base trazia a referência montada por
  uma versão antiga do conversor, que lia `AHU_CU_0005-01` como série `000-501` e `AHU_ACL_CU_0013`
  como `001-003`. Quando o `new_code` tem cota reconhecível, a referência sai dele.

Os resumos não são corrigidos: erros que já vêm da fonte (`SãoSão Trindade`, `Barrreto`,
`deTávora`) ficam como estão, e o app os marca com um [sic] (`ficha.marcar_sic`).

E um encaixe: a **`referencia_digitarq`**, o registro do DigitArq que contém o documento. O
DigitArq descreve muitos documentos em intervalo (`…/017-001/0016/03265.03267`), e o intervalo dele
nem sempre começa onde começa o do catálogo; aí a busca pela referência do documento volta vazia.
O índice `corpus/digitarq_referencias.tsv.gz` (gerado por `coleta/scripts/varrer_digitarq.py`)
lista os registros do DigitArq caixa a caixa, e cada documento recebe o registro que tem o seu
número: o mesmo número de início, ou um intervalo que o contém. Sem registro, a coluna fica vazia e
o app oferece a busca pela caixa.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ahu.conversor import para_referencia, reconhecer

_CABECALHO = re.compile(r"\s*(INSTRUMENTO DE PESQUISA|ARQUIVO HISTÓRICO ULTRAMARINO)\b.*$", re.S)
_REGISTRO = re.compile(r"^(PT/AHU/CU/[\d-]+)/(\d{4})/(\d+)(?:\.(\d+))?$")


def limpar_new_code(texto):
    if not isinstance(texto, str):
        return texto
    return _CABECALHO.sub("", texto).strip()


def referencia(new_code, reference_code):
    """A referência pelo conversor atual: do `new_code` quando ele traz cota, senão a guardada."""
    for texto in (new_code, reference_code):
        cota = reconhecer(texto) if isinstance(texto, str) else None
        ref = para_referencia(cota) if cota else None
        if ref:
            return ref
    return None


class IndiceDigitArq:
    """Os registros do DigitArq por série: (início, fim, caixa, referência)."""

    def __init__(self, registros):
        self.por_serie = defaultdict(list)
        for ref in registros:
            m = _REGISTRO.match(ref)
            if m:
                serie, caixa, ini, fim = m.groups()
                self.por_serie[serie].append((int(ini), int(fim or ini), caixa, ref))

    @classmethod
    def ler(cls, caminho: Path) -> "IndiceDigitArq | None":
        if not caminho.exists():
            return None
        with gzip.open(caminho, "rt", encoding="utf-8") as f:
            return cls(linha.split("\t")[0].strip() for linha in f if linha.startswith("PT/"))

    def achar(self, ref: str | None) -> str | None:
        """O registro com o número do documento: mesmo início antes de intervalo que o contém, e
        a mesma caixa antes de outra caixa da série (há caixas trocadas entre catálogo e DigitArq)."""
        m = _REGISTRO.match(ref or "")
        if not m:
            return None
        serie, caixa, numero = m.group(1), m.group(2), int(m.group(3))
        candidatos = [r for r in self.por_serie.get(serie, ()) if r[0] <= numero <= r[1]]
        if not candidatos:
            return None
        candidatos.sort(key=lambda r: (r[0] != numero, r[2] != caixa, r[1] - r[0]))
        return candidatos[0][3]


def _mudou(antes: pd.Series, depois: pd.Series) -> int:
    return int((antes.fillna("") != depois.fillna("")).sum())


def ajustar(base: pd.DataFrame, indice: IndiceDigitArq | None) -> tuple[pd.DataFrame, dict]:
    """A base com os dois acertos e a `referencia_digitarq`, e a contagem de cada um."""
    base = base.copy()
    contagem = {}
    novo = base["new_code"].map(limpar_new_code)
    contagem["new_code_sem_cabecalho"] = _mudou(base["new_code"], novo)
    base["new_code"] = novo
    ref = pd.Series([referencia(n, r) for n, r in zip(base["new_code"], base["reference_code"])],
                    index=base.index, dtype="object")
    antiga = base["reference_code"].where(base["reference_code"].astype(str).str.startswith("PT/"))
    contagem["referencia_refeita"] = int((ref.notna() & (ref != antiga)).sum())
    base["reference_code"] = ref.where(ref.notna(), base["reference_code"])
    if indice is not None:
        base["referencia_digitarq"] = [indice.achar(r) if isinstance(r, str) else None
                                       for r in base["reference_code"]]
        com_ref = base["reference_code"].astype(str).str.startswith("PT/")
        contagem["digitarq_com_registro"] = int(base["referencia_digitarq"].notna().sum())
        contagem["digitarq_sem_registro"] = int((com_ref & base["referencia_digitarq"].isna()).sum())
        contagem["digitarq_registro_diferente"] = int(
            (base["referencia_digitarq"].notna()
             & (base["referencia_digitarq"] != base["reference_code"])).sum())
    return base, contagem
