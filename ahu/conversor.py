"""conversor.py — a aba Conversor: uma cota em qualquer das três famílias, convertida nas outras.

As três famílias são as das três colunas de `documentos.parquet`:

| Família | Exemplo |
|---|---|
| código de referência (DigitArq/CRAV) | `PT/AHU/CU/017/0069/06420` |
| cota do Resgate | `AHU_ACL_CU_017, Cx. 69, D. 6420.` |
| cota antiga, por região | `AHU-Rio de Janeiro, cx. 76, doc. 41.` |

A conversão tem **três níveis**, e cada resposta diz por qual deles saiu:

1. **Formatação.** Referência e Resgate são a mesma cota com outra grafia: a regra reproduz 7.037
   dos 7.038 pares da base e vale para qualquer cota, esteja ou não no corpus. A única diferença
   que não é de zeros à esquerda: a subsérie tem três dígitos na referência (`017-001`) e dois no
   Resgate (`017-01`).
2. **Numeração igual.** Duas coleções antigas guardam caixa e documento do Resgate: Castro e
   Almeida (`AHU-Rio de Janeiro-Calmeida`, série 017-01) e Mendes Gouveia
   (`AHU-São Paulo-MGouveia`, série 023-01). Nelas a cota antiga também é só formatação.
3. **Concordância da base.** As demais cotas antigas vêm de uma arrumação anterior do AHU, com
   caixas e documentos numerados de outro jeito (só 2,4% dos documentos mantêm o número). Não há
   fórmula: elas só se convertem se o par existir em `documentos.parquet`. Fora da base o
   conversor diz que não dá, em vez de dar um palpite — um palpite errado manda o pesquisador
   pedir a caixa errada no arquivo.

A **cifra série-estado** (`SERIES`) foi tirada do cruzamento entre a série do `reference_code`, a
pasta (`folder`) e a região escrita na cota antiga; cada série corresponde a uma única pasta. Uma
série fora da cifra ainda é convertida entre referência e Resgate, só que sem nome e com aviso.
O teste `test_a_cifra_confere_com_as_pastas_da_base` falha quando uma importação nova trouxer uma
série que a cifra não conhece.

Nada aqui depende de Streamlit nem de rede.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# A cifra série-estado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Serie:
    codigo: str                   # como no Resgate: '017', '017-01'
    nome: str                     # o nome que o app mostra (a pasta da base)
    antiga: str | None = None     # a grafia da região na cota antiga, quando há uma
    numeracao_igual: bool = False  # nível 2: a cota antiga guarda caixa e documento


SERIES: tuple[Serie, ...] = (
    Serie("003", "Brasil Geral"),
    Serie("012", "Nova Colônia do Sacramento", "Nova Colónia do Sacramento"),
    Serie("017", "Rio de Janeiro", "Rio de Janeiro"),
    Serie("017-01", "Rio de Janeiro, Castro e Almeida", "Rio de Janeiro-Calmeida",
          numeracao_igual=True),
    Serie("019", "Rio Grande do Sul", "Rio Grande do Sul"),
    Serie("021", "Santa Catarina", "Santa Catarina"),
    Serie("023", "São Paulo", "São Paulo"),
    Serie("023-01", "São Paulo, Mendes Gouveia", "São Paulo-MGouveia", numeracao_igual=True),
    # 030 e 035 só aparecem na pasta "Ultramar", e nunca com cota antiga. Nome a confirmar.
    Serie("030", "Ultramar"),
    Serie("035", "Ultramar"),
    Serie("065", "Montevidéu", "Montevidéu"),
    Serie("066", "Buenos Aires", "Buenos Aires"),
    Serie("071", "Paraguai", "Paraguai"),
)
POR_CODIGO = {s.codigo: s for s in SERIES}

# A marca que, na cota antiga, separa as duas coleções de numeração igual da série comum.
_MARCAS_NUMERACAO_IGUAL = (("calmeida", "017-01"), ("gouveia", "023-01"))

NIVEIS = {
    1: "formatação",
    2: "numeração igual",
    3: "concordância da base",
}


def codigo_da_serie(serie: int, subserie: int | None) -> str:
    """A série como o Resgate a escreve: `017`, `017-01`."""
    return f"{serie:03d}" + (f"-{subserie:02d}" if subserie is not None else "")


# ---------------------------------------------------------------------------
# Reconhecer uma cota
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cota:
    """Uma cota decomposta. `serie` é None só numa cota antiga comum (que não a traz)."""

    familia: str                   # 'referencia' | 'resgate' | 'antiga'
    caixa: int
    documento: int
    serie: int | None = None
    subserie: int | None = None
    documento_fim: int | None = None   # o fim de um intervalo do Resgate (`D. 20-21`)
    regiao: str | None = None          # a região como foi escrita na cota antiga
    serie_curta: bool = False          # a série veio com menos de três dígitos (`cu 21`)

    @property
    def codigo_serie(self) -> str | None:
        return None if self.serie is None else codigo_da_serie(self.serie, self.subserie)


# `PT/AHU/CU/021/0006/00390`, `PT/AHU/CU/017-001/0003/00387`. O documento aceita seis dígitos
# porque a base tem dois assim (`PT/AHU/CU/017-001/0072/168746`), provavelmente erro de catálogo;
# o conversor os devolve como estão, sem adivinhar a correção.
_RE_REFERENCIA = re.compile(
    r"PT\s*/\s*AHU\s*/\s*CU\s*/\s*(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*/\s*(\d{1,4})\s*/\s*(\d{1,6})",
    re.I)
# `AHU_ACL_CU_021, Cx. 6, D. 390.` e as grafias tortas da base: sem vírgula, `Cx.9`, `CX.`,
# `C x.`, sem o `Cx.` (`AHU_ACL_CU_017, 8, D. 818.`), lixo entre `D.` e o número
# (`D. .387-389.`, `D. –1011-1015.`) e intervalo com hífen dobrado (`D. 9--9`).
_RE_RESGATE = re.compile(
    r"AHU[_\s]*ACL[_\s]*CU[_\s]*(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*,?\s*(?:C\s*x\.?)?\s*(\d+)\s*,?\s*"
    r"D(?:oc)?\.?[\s.\-–—]*(\d+)(?:\s*[-–—]{1,2}\s*(\d+))?", re.I)
# A cota antiga é lida sobre o texto dobrado (sem acento, minúsculo), porque a região vem com
# grafias variadas: `Colónia`/`Colônia`, `SANTA-CATARINA`, `São Paulo, MGouveia`.
_RE_ANTIGA_REGIAO = re.compile(r"\bahu\s*[-_]?\s*(?!acl\b)([a-z][a-z .\-]*?)\s*[,;_]")
_RE_ANTIGA_PAR = re.compile(r"\bcx\.?\s*(\d+)\s*,?\s*docs?\.?\s*(\d+)(?:\s*-\s*(\d+))?\b")
_RE_MARCA = re.compile(r"^\s*(mgouveia|calmeida)\b\s*[,;]?")


def dobrar(texto: str) -> str:
    """Sem acento e em minúsculas, mantendo a pontuação (a cota antiga precisa dela)."""
    t = unicodedata.normalize("NFKD", str(texto or ""))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _int(valor: str | None) -> int | None:
    return None if valor is None else int(valor)


def reconhecer_todas(texto: str | None) -> list[Cota]:
    """Todas as cotas de uma linha. Uma linha costuma ter uma; a cota antiga pode ter várias.

    `AHU-Rio de Janeiro, cx. 279, doc. 48; cx. 280, doc. 111.` são duas cotas da mesma região, e
    cada uma é convertida separadamente.
    """
    if not texto or not str(texto).strip():
        return []
    texto = str(texto)
    m = _RE_REFERENCIA.search(texto)
    if m:
        serie, sub, caixa, doc = m.groups()
        return [Cota("referencia", int(caixa), int(doc), int(serie), _int(sub),
                     serie_curta=len(serie) < 3)]
    m = _RE_RESGATE.search(texto)
    if m:
        serie, sub, caixa, doc, fim = m.groups()
        fim = _int(fim)
        return [Cota("resgate", int(caixa), int(doc), int(serie), _int(sub),
                     documento_fim=fim if fim and fim > int(doc) else None,
                     serie_curta=len(serie) < 3)]
    return _reconhecer_antiga(texto)


def reconhecer(texto: str | None) -> Cota | None:
    """A primeira cota da linha, ou None se não houver uma reconhecível."""
    cotas = reconhecer_todas(texto)
    return cotas[0] if cotas else None


def _reconhecer_antiga(texto: str) -> list[Cota]:
    dobrado = dobrar(texto)
    m = _RE_ANTIGA_REGIAO.search(dobrado)
    if not m:
        return []
    regiao = m.group(1).strip(" .-")
    resto = dobrado[m.end():]
    # `AHU-São Paulo, MGouveia, Cx. 3, doc. 317.`: a marca da coleção depois da vírgula.
    marca = _RE_MARCA.match(resto)
    if marca:
        regiao = f"{regiao}-{marca.group(1)}"
        resto = resto[marca.end():]
    pares = list(_RE_ANTIGA_PAR.finditer(resto))
    if not regiao or not pares:
        return []
    serie, sub = None, None
    for marca_colecao, codigo in _MARCAS_NUMERACAO_IGUAL:
        if marca_colecao in regiao:
            numero, _, subnumero = codigo.partition("-")
            serie, sub = int(numero), int(subnumero)
    escrita = _regiao_como_escrita(texto)
    return [Cota("antiga", int(p.group(1)), int(p.group(2)), serie, sub,
                 documento_fim=_int(p.group(3)), regiao=escrita) for p in pares]


def _regiao_como_escrita(texto: str) -> str:
    """A região com os acentos de quem escreveu: `São Paulo-MGouveia`, não `sao paulo-mgouveia`."""
    m = re.search(r"AHU\s*[-_]?\s*([^\d,;_]+?)\s*[,;_]", texto, re.I)
    return m.group(1).strip(" .-") if m else ""


# ---------------------------------------------------------------------------
# Escrever uma cota
# ---------------------------------------------------------------------------


def para_referencia(cota: Cota) -> str | None:
    """`PT/AHU/CU/<série>[-<subsérie em três dígitos>]/<caixa em quatro>/<documento em cinco>`."""
    if cota.serie is None:
        return None
    serie = f"{cota.serie:03d}" + (f"-{cota.subserie:03d}" if cota.subserie is not None else "")
    return f"PT/AHU/CU/{serie}/{cota.caixa:04d}/{cota.documento:05d}"


def para_resgate(cota: Cota) -> str | None:
    """`AHU_ACL_CU_<série>[-<subsérie em dois>], Cx. <caixa>, D. <documento>.`, sem zeros."""
    if cota.serie is None:
        return None
    documento = str(cota.documento)
    if cota.documento_fim:
        documento += f"-{cota.documento_fim}"
    return f"AHU_ACL_CU_{cota.codigo_serie}, Cx. {cota.caixa}, D. {documento}."


def para_antiga(cota: Cota) -> str | None:
    """A cota antiga, só quando a série é de numeração igual (nível 2)."""
    serie = POR_CODIGO.get(cota.codigo_serie or "")
    if serie is None or not serie.numeracao_igual:
        return None
    documento = str(cota.documento)
    if cota.documento_fim:
        documento += f"-{cota.documento_fim}"
    return f"AHU-{serie.antiga}, cx. {cota.caixa}, doc. {documento}."


def limpar(texto) -> str:
    """Uma cota da base como se imprime: sem a pontuação solta do fim (`D. 360.]`)."""
    if texto is None or (isinstance(texto, float) and texto != texto):
        return ""
    return str(texto).strip().rstrip(" .];,") + ("." if str(texto).strip() else "")


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


@dataclass
class Conversao:
    """Uma linha da tabela da aba: a entrada e o que ela vira nas outras famílias."""

    entrada: str
    familia: str | None = None
    referencia: str | None = None
    resgate: str | None = None
    antiga: str | None = None
    estado: str | None = None
    nivel: int | None = None
    avisos: list[str] = field(default_factory=list)
    linha: int | None = None          # a linha de `documentos.parquet`, quando está na base

    @property
    def convertida(self) -> bool:
        return self.referencia is not None

    @property
    def como(self) -> str:
        return NIVEIS.get(self.nivel, "sem conversão")


ROTULO_FAMILIA = {"referencia": "código de referência", "resgate": "cota do Resgate",
                  "antiga": "cota antiga"}


def converter(texto: str, dados=None) -> list[Conversao]:
    """Uma linha digitada -> uma ou mais conversões.

    Mais de uma quando a cota antiga traz várias cotas, ou quando uma cota antiga da base leva a
    vários documentos atuais (`AHU-Rio de Janeiro, cx. 76, doc. 41.` leva a doze). Sem `dados`, só
    os níveis 1 e 2: a concordância da base precisa da base.
    """
    entrada = str(texto or "").strip()
    cotas = reconhecer_todas(entrada)
    if not cotas:
        return [Conversao(entrada, avisos=[
            "Formato não reconhecido. Confira a digitação; sufixos (`doc. 1-A`, `cx. 9-A`) e "
            "códices (`n.v.`) ficam fora do conversor."])]
    saida: list[Conversao] = []
    for cota in cotas:
        saida.extend(_converter_uma(entrada, cota, dados))
    return saida


def _converter_uma(entrada: str, cota: Cota, dados) -> list[Conversao]:
    if cota.serie is None:
        return _pela_base(entrada, cota, dados)

    serie = POR_CODIGO.get(cota.codigo_serie)
    # Nível 2 sempre que a cota antiga entra na resposta pela numeração igual, venha ela de que
    # família vier: é a coluna que depende da regra mais estreita.
    nivel = 2 if serie is not None and serie.numeracao_igual else 1
    c = Conversao(entrada, cota.familia, para_referencia(cota), para_resgate(cota),
                  para_antiga(cota), serie.nome if serie else None, nivel)
    if serie is None:
        c.avisos.append(f"A série {cota.codigo_serie} não está na cifra: a forma foi convertida, "
                        "mas sem o nome do estado.")
    if cota.serie_curta:
        c.avisos.append(f"Série completada com zeros: {cota.codigo_serie}.")
    if cota.documento_fim and cota.familia != "antiga":
        n = cota.documento_fim - cota.documento + 1
        c.avisos.append(f"Intervalo de {n} documentos: o código de referência guarda só o "
                        "primeiro.")

    linha = _linha_na_base(dados, c.referencia)
    if linha is not None:
        c.linha = linha
        documento = dados.documentos.iloc[linha]
        if c.antiga is None:
            antiga = documento.get("old_code")
            if isinstance(antiga, str) and antiga.strip():
                c.antiga = limpar(antiga)
                c.avisos.append("Cota antiga tirada da base, não da formatação.")
        # Um intervalo que a referência perdeu, a base devolve.
        if cota.familia == "referencia":
            da_base = reconhecer(documento.get("new_code"))
            if da_base and da_base.documento_fim and da_base.documento == cota.documento:
                c.resgate = para_resgate(da_base)
    elif c.antiga is None and dados is not None:
        c.avisos.append("Fora da base: a cota antiga não é derivável desta série.")
    return [c]


def _pela_base(entrada: str, cota: Cota, dados) -> list[Conversao]:
    """Nível 3: a cota antiga comum só se converte pela concordância da base."""
    if dados is None:
        return [Conversao(entrada, "antiga", estado=cota.regiao, avisos=[
            "Cota antiga: a numeração mudou na arrumação atual, e a conversão depende da base."])]
    linhas = indice_de_antigas(dados).get(chave_antiga(cota), [])
    if not linhas:
        return [Conversao(entrada, "antiga", estado=cota.regiao, avisos=[
            "Sem conversão: esta cota antiga não está na base, e a numeração antiga não é "
            "derivável da atual. A concordância está no catálogo do Projeto Resgate."])]
    saida = []
    for posicao, i in enumerate(linhas, start=1):
        documento = dados.documentos.iloc[i]
        resgate = documento.get("new_code")
        atual = reconhecer(resgate) or reconhecer(documento.get("reference_code"))
        serie = POR_CODIGO.get(atual.codigo_serie) if atual else None
        c = Conversao(entrada, "antiga",
                      para_referencia(atual) if atual else None,
                      limpar(resgate) if isinstance(resgate, str) else None,
                      limpar(documento.get("old_code")),
                      serie.nome if serie else cota.regiao, 3, linha=i)
        if len(linhas) > 1:
            c.avisos.append(f"{posicao} de {len(linhas)} documentos atuais com esta cota antiga.")
        saida.append(c)
    return saida


def chave_antiga(cota: Cota) -> tuple[str, int, int]:
    """Região sem acento, caixa e pontuação, mais caixa e documento: `Colónia` casa `Colônia`."""
    regiao = re.sub(r"[^a-z0-9]+", " ", dobrar(cota.regiao)).strip()
    return regiao, cota.caixa, cota.documento


def indice_de_antigas(dados) -> dict[tuple[str, int, int], list[int]]:
    """Cota antiga -> linhas de `documentos.parquet`, cota por cota.

    Não é a `busca.tabela_de_cotas`: aquela lê só a primeira cota de cada campo, e aqui
    `cx. 279, doc. 48; cx. 280, doc. 111` tem de responder pelas duas. Guardado no próprio
    `Dados`, como os outros índices derivados.
    """
    cache = getattr(dados, "_indice_antigas", None)
    if cache is not None:
        return cache
    indice: dict[tuple[str, int, int], list[int]] = {}
    if "old_code" in dados.documentos.columns:
        for i, valor in enumerate(dados.documentos["old_code"]):
            if not isinstance(valor, str):
                continue
            for cota in _reconhecer_antiga(valor):
                linhas = indice.setdefault(chave_antiga(cota), [])
                if i not in linhas:
                    linhas.append(i)
    object.__setattr__(dados, "_indice_antigas", indice)
    return indice


def _linha_na_base(dados, referencia: str | None) -> int | None:
    if dados is None or referencia is None:
        return None
    from ahu.lista_ahu import indice_de_cotas

    linhas = indice_de_cotas(dados).por_referencia.get(referencia, [])
    return linhas[0] if linhas else None


def converter_lista(texto: str, dados=None) -> list[Conversao]:
    """Uma cota por linha; linhas vazias e as que começam com `#` ficam de fora."""
    saida: list[Conversao] = []
    for linha in (texto or "").splitlines():
        linha = linha.split("|")[0].strip()
        if linha and not linha.startswith("#"):
            saida.extend(converter(linha, dados))
    return saida


def tabela(conversoes: list[Conversao]):
    """A tabela que a aba mostra. O link do DigitArq sai do código de referência."""
    import pandas as pd

    from ahu.links import digitarq

    return pd.DataFrame([{
        "Entrada": c.entrada,
        "Como": c.como,
        "Estado": c.estado or "—",
        "Código de referência": c.referencia or "—",
        "Cota do Resgate": c.resgate or "—",
        "Cota antiga": c.antiga or "—",
        "DigitArq": digitarq(c.referencia) if c.referencia else None,
        "Avisos": " ".join(c.avisos),
    } for c in conversoes])


def tabela_da_cifra():
    """A cifra como o usuário a lê: o que o conversor cobre."""
    import pandas as pd

    return pd.DataFrame([{
        "Série": s.codigo,
        "Estado": s.nome,
        "Na cota antiga": s.antiga or "—",
        "Cota antiga convertível": "sim, por formatação" if s.numeracao_igual else
                                   ("só pela base" if s.antiga else "—"),
    } for s in SERIES])


def para_lista_ahu(conversoes: list[Conversao]) -> str:
    """As cotas convertidas, uma por linha, para a caixa da lista do AHU."""
    vistas, linhas = set(), []
    for c in conversoes:
        cota = c.resgate or c.referencia
        if cota and cota not in vistas:
            vistas.add(cota)
            linhas.append(cota)
    return "\n".join(linhas)
