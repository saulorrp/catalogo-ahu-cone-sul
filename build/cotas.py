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

Três acertos de dados, achados na auditoria de 12/11/2026 (`scripts/auditar_banco.py`):

* **Pasta com grafia única.** O ahu-sul escreve `Montividéu`, o ahu-brasil `Montevidéu` (a grafia da
  cifra): o filtro mostrava duas pastas para a mesma série. Fica a da cifra.
* **Ano que era número de documento.** Quando a data do verbete é só o intervalo de números
  (`1399. 1401`, Bahia, Luísa da Fonseca), o fim do intervalo virou `ano_isolado`. O ano fica vazio.
* **Cota repetida em bloco.** Na extração de um catálogo (Minas Gerais, Cx. 13) a mesma cota
  (`D. 1065`) ficou em 45 verbetes seguidos, cada um com o seu número de catálogo (1066 a 1110).
  Quando um bloco de 3 ou mais verbetes divide a cota, os números de catálogo são seguidos, cada
  um existe no DigitArq naquela caixa e nenhum já é de outro verbete, a cota de cada verbete passa
  a ser a do seu número. O DigitArq é a prova; sem ela, nada muda.
* **Verbetes repetidos não são apagados.** Mesma série, caixa, documento e resumo (sem acento,
  caixa e pontuação) é o mesmo verbete duas vezes: entre coleções (a cota veio torta, `D. .36`, e
  escapou do importador) ou dentro do ahu-brasil. `scripts/listar_duplicatas.py` lista as cópias
  em `corpus/duplicatas.ignore` (fica a primeira, na ordem da base: o ahu-sul vem antes). A
  construção do app marca as cópias na coluna `duplicata_de` e a busca não as devolve, mas elas
  continuam nos dados publicados, por transparência.

E um encaixe: a **`referencia_digitarq`**, o registro do DigitArq que contém o documento. O
DigitArq descreve muitos documentos em intervalo (`…/017-001/0016/03265.03267`), e o intervalo dele
nem sempre começa onde começa o do catálogo; aí a busca pela referência do documento volta vazia.
O índice `corpus/digitarq_referencias.tsv.gz` (gerado por `coleta/scripts/varrer_digitarq.py`)
lista os registros do DigitArq caixa a caixa, e cada documento recebe o registro que tem o seu
número: o mesmo número de início, ou um intervalo que o contém. Sem registro, a coluna fica vazia e
o app oferece a busca pela caixa.

E outro encaixe, o **`resgate_docreader`**: o quadro do visualizador de imagens do Projeto Resgate
(Biblioteca Nacional) em que o documento começa, como `021_SC#37` (rolo e quadro). Vem do índice
`corpus/resgate_docreader.tsv.gz` (gerado por `coleta/scripts/extrair_docreader.py` a partir dos
links que as páginas de verbetes do gov.br põem em cada número) e só existe para as capitanias
cujos catálogos estão em HTML no gov.br. Sem quadro, o app aponta para o portal.
"""
from __future__ import annotations

import gzip
import re
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ahu.conversor import para_referencia, reconhecer

_CABECALHO = re.compile(r"\s*(INSTRUMENTO DE PESQUISA|ARQUIVO HISTÓRICO ULTRAMARINO)\b.*$", re.S)
_REGISTRO = re.compile(r"^(PT/AHU/CU/[\d-]+)/(\d{4})/(\d+)(?:\.(\d+))?$")


GRAFIAS_PASTA = {"Montividéu": "Montevidéu"}
_SO_NUMEROS = re.compile(r"^\s*(\d+)\s*[.\-–]\s*(\d+)\s*\.?\s*(\[.*\])?\s*$")
_ANO = re.compile(r"\b(1[5-8]\d\d)\b")


def ano_falso(data, ano, cota):
    """Se a data é só `<início>. <fim>` e o ano é o fim (número de documento, não ano), devolve o
    ano certo: o que a data traz entre colchetes (`[ant. 1660, Novembro, 18]`) ou None. Sem erro,
    devolve False."""
    if ano is None or pd.isna(ano) or not isinstance(data, str):
        return False
    m = _SO_NUMEROS.match(data)
    if not m:
        return False
    ini, fim = int(m.group(1)), int(m.group(2))
    if int(ano) != fim:
        return False
    if not ((cota is not None and cota.documento_fim == fim) or 0 < fim - ini <= 60):
        return False
    ano_colchete = _ANO.search(m.group(3) or "")
    return int(ano_colchete.group(1)) if ano_colchete else None


def _texto_chave(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto or ""))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def duplicatas(base: pd.DataFrame, cotas_lidas=None) -> pd.Series:
    """Para cada linha que repete um verbete anterior (mesma série, caixa, documento e resumo), o
    doc_uid da primeira ocorrência; None nas demais."""
    if cotas_lidas is None:
        cotas_lidas = [reconhecer(t) if isinstance(t, str) else None for t in base["new_code"]]
    vistos: dict[tuple, str] = {}
    repete = []
    for uid, cota, texto in zip(base["doc_uid"], cotas_lidas, base["description"]):
        chave_texto = _texto_chave(texto)
        if cota is None or cota.serie is None or not chave_texto:
            repete.append(None)
            continue
        chave = (cota.codigo_serie, cota.caixa, cota.documento, cota.documento_fim, chave_texto)
        primeira = vistos.setdefault(chave, uid)
        repete.append(primeira if primeira != uid else None)
    return pd.Series(repete, index=base.index, dtype="object")


_REPETE = re.compile(r"repete\s+(\S+)")


def ler_ignorados(caminho: Path) -> dict[str, str]:
    """doc_uid -> doc_uid mantido, de um arquivo .ignore: o primeiro campo de cada linha é o uid
    ignorado, `# repete <uid>` diz qual fica; `#` sozinho abre comentário."""
    if not caminho.exists():
        return {}
    uids = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        dado, _, nota = linha.partition("#")
        campo = dado.split()
        if campo:
            m = _REPETE.search(nota)
            uids[campo[0]] = m.group(1) if m else ""
    return uids


_NUM_CATALOGO = re.compile(r"^\s*(\d+)\.")
_DOC_NA_COTA = re.compile(r"(D\.?\s*)(\d+)(\.?\s*)$")


def curar_blocos(base: pd.DataFrame, cotas_lidas, indice) -> dict:
    """Índice da linha -> (new_code, reference_code) novos, para os blocos de cota repetida que o
    DigitArq confirma (ver o cabeçalho)."""
    if indice is None or "document_id_and_date" not in base.columns:
        return {}
    grupos: dict[tuple, list] = defaultdict(list)
    for i, col, cota in zip(base.index, base.get("colecao", pd.Series("", index=base.index)),
                            cotas_lidas):
        if cota is not None and cota.serie is not None and not cota.documento_fim:
            grupos[(col, cota.codigo_serie, cota.caixa, cota.documento)].append((i, cota))
    usadas = set(base["reference_code"].dropna())
    novo = {}
    for linhas in grupos.values():
        if len(linhas) < 3:
            continue
        nums = []
        for i, _ in linhas:
            m = _NUM_CATALOGO.match(str(base.at[i, "document_id_and_date"] or ""))
            nums.append(int(m.group(1)) if m else None)
        if None in nums or len(set(nums)) != len(nums) or max(nums) - min(nums) != len(nums) - 1:
            continue
        refs = [para_referencia(replace(c, documento=n)) for (_, c), n in zip(linhas, nums)]
        if not all(indice.existe(r) for r in refs) or any(r in usadas for r in refs):
            continue
        for (i, _), n, r in zip(linhas, nums, refs):
            codigo = _DOC_NA_COTA.sub(lambda m, n=n: f"{m.group(1)}{n}{m.group(3)}",
                                      str(base.at[i, "new_code"]).strip())
            novo[i] = (codigo, r)
    return novo


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

    def existe(self, ref: str) -> bool:
        m = _REGISTRO.match(ref or "")
        if not m:
            return False
        serie, caixa, numero = m.group(1), m.group(2), int(m.group(3))
        return any(ini == numero and fim == numero and cx == caixa
                   for ini, fim, cx, _ in self.por_serie.get(serie, ()))

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


def ler_docreader(caminho: Path) -> dict[str, str] | None:
    """Referência CRAV -> `rolo#quadro` do DocReader do Resgate. None quando o índice não existe."""
    if not caminho.exists():
        return None
    with gzip.open(caminho, "rt", encoding="utf-8") as f:
        linhas = (linha.rstrip("\n").split("\t") for linha in f if linha.startswith("PT/"))
        return {ref: f"{bib}#{pagfis}" for ref, bib, pagfis in linhas}


def _mudou(antes: pd.Series, depois: pd.Series) -> int:
    return int((antes.fillna("") != depois.fillna("")).sum())


def ajustar(base: pd.DataFrame, indice: IndiceDigitArq | None,
            docreader: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """A base com os dois acertos, a `referencia_digitarq` e o `resgate_docreader`, e a contagem
    de cada um."""
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

    lidas = [reconhecer(t) if isinstance(t, str) else None for t in base["new_code"]]
    curadas = curar_blocos(base, lidas, indice)
    contagem["cota_em_bloco_curada"] = len(curadas)
    for i, (codigo, ref) in curadas.items():
        base.at[i, "new_code"], base.at[i, "reference_code"] = codigo, ref
    if curadas:
        lidas = [reconhecer(t) if isinstance(t, str) else None for t in base["new_code"]]
    if "folder" in base.columns:
        pasta = base["folder"].replace(GRAFIAS_PASTA)
        contagem["pasta_grafia_unificada"] = _mudou(base["folder"], pasta)
        base["folder"] = pasta
    if "ano_isolado" in base.columns and "document_id_and_date" in base.columns:
        certo = [ano_falso(d, a, c) for d, a, c in
                 zip(base["document_id_and_date"], base["ano_isolado"], lidas)]
        falso = pd.Series([c is not False for c in certo], index=base.index)
        novo = pd.Series([c if c not in (False, None) else pd.NA for c in certo],
                         index=base.index, dtype="Int64")
        contagem["ano_falso_corrigido"] = int(falso.sum())
        base["ano_isolado"] = base["ano_isolado"].mask(falso, novo).astype("Int64")
    if indice is not None:
        base["referencia_digitarq"] = [indice.achar(r) if isinstance(r, str) else None
                                       for r in base["reference_code"]]
        com_ref = base["reference_code"].astype(str).str.startswith("PT/")
        contagem["digitarq_com_registro"] = int(base["referencia_digitarq"].notna().sum())
        contagem["digitarq_sem_registro"] = int((com_ref & base["referencia_digitarq"].isna()).sum())
        contagem["digitarq_registro_diferente"] = int(
            (base["referencia_digitarq"].notna()
             & (base["referencia_digitarq"] != base["reference_code"])).sum())
    if docreader is not None:
        base["resgate_docreader"] = [docreader.get(r) if isinstance(r, str) else None
                                     for r in base["reference_code"]]
        contagem["resgate_com_quadro"] = int(base["resgate_docreader"].notna().sum())
    return base, contagem
