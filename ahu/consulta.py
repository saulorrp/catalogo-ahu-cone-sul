"""consulta.py — WP 2.2: o objeto `Consulta`, com ida e volta para dicionário e para a URL.

`Consulta` é o objeto central do app (spec §4): texto, modo, rigor, filtros, lente, ordenação,
página e semente. A **mesma** estrutura vai para três lugares:

* a **URL**, compacta e versionada, para o "Copiar link desta busca" (§8.3, A6);
* o **bloco de proveniência** dos arquivos exportados (§8.2), em forma legível;
* a **capa do dossiê em PDF** (§8.2).

Duas regras que a ida e volta tem de respeitar (A6):

1. o que estava no padrão não vai para a URL — um link de busca não carrega estado morto;
2. o que vai, volta igual. `de_url(para_url(c)) == c` para qualquer `Consulta`, e é isso que o
   teste cobre com buscas variadas.

A versão (`v=1`) é o que permite mudar o formato depois sem quebrar links já copiados: uma URL
sem versão, ou com versão desconhecida, é recusada com mensagem clara em vez de virar silenciosamente
outra busca.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from ahu import busca
from ahu.filtros import LENTES, Filtros, aplicar_lente

VERSAO_URL = 1
ORDENACOES = ("relevancia", "ano_asc", "ano_desc")
TAMANHO_PAGINA = 25     # §5.5

# Chaves curtas da URL. Trocar qualquer uma delas exige subir VERSAO_URL.
CHAVES = {
    "texto": "q", "modo": "m", "rigor": "r", "pagina": "p", "ordenacao": "o",
    "semente": "s", "lente": "l",
}
CHAVES_FILTROS = {
    "colecoes": "fc", "ano_de": "a1", "ano_ate": "a2", "sem_data": "sd", "pastas": "fp",
    "tipologias": "ft", "pessoas": "fu", "destinatario": "fd", "categorias": "fg",
    "vetores": "fv", "impactos": "fi", "srsp_de": "s1", "srsp_ate": "s2",
    "paises_origem": "fn", "cidades_origem": "fy", "so_confianca_alta": "fa",
}
SEPARADOR = "|"     # separa itens de uma lista dentro de um valor da URL


class ErroDeLink(ValueError):
    """Link de busca inválido: versão desconhecida ou valor fora do permitido."""


@dataclass
class Consulta:
    texto: str = ""
    modo: str = "automatico"
    rigor: str = "medio"
    filtros: Filtros = field(default_factory=Filtros)
    lente: str | None = None
    ordenacao: str = "relevancia"
    pagina: int = 1
    semente: int | None = None      # §7.3: a semente da amostra viaja com a busca

    # -----------------------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.filtros, dict):
            self.filtros = Filtros(**self.filtros)
        self.validar()

    def validar(self) -> None:
        if self.modo not in busca.MODOS:
            raise ErroDeLink(f"Modo desconhecido: {self.modo!r}. Use um de {', '.join(busca.MODOS)}.")
        if self.rigor not in busca.RIGORES:
            raise ErroDeLink(
                f"Rigor desconhecido: {self.rigor!r}. Use um de {', '.join(busca.RIGORES)}."
            )
        if self.ordenacao not in ORDENACOES:
            raise ErroDeLink(
                f"Ordenação desconhecida: {self.ordenacao!r}. Use uma de {', '.join(ORDENACOES)}."
            )
        if self.lente is not None and self.lente not in LENTES:
            raise ErroDeLink(f"Lente desconhecida: {self.lente!r}. Use uma de {', '.join(LENTES)}.")
        if self.pagina < 1:
            raise ErroDeLink(f"Página deve ser 1 ou mais, veio {self.pagina}.")

    # -----------------------------------------------------------------
    # Dicionário (proveniência dos exportados, §8.2)
    # -----------------------------------------------------------------
    def para_dicionario(self) -> dict:
        """Forma legível e completa, para o bloco de proveniência e para a capa do PDF."""
        return {
            "versao": VERSAO_URL,
            "texto": self.texto,
            "modo": self.modo,
            "rigor": self.rigor,
            "lente": self.lente,
            "ordenacao": self.ordenacao,
            "pagina": self.pagina,
            "semente": self.semente,
            "filtros": asdict(self.filtros),
            "filtros_ativos": self.filtros.ativos(),
        }

    @classmethod
    def de_dicionario(cls, dados: dict) -> "Consulta":
        versao = dados.get("versao", VERSAO_URL)
        if int(versao) != VERSAO_URL:
            raise ErroDeLink(
                f"Esta consulta foi gravada na versão {versao} e o app usa a {VERSAO_URL}."
            )
        conhecidos = {f.name for f in fields(cls)}
        argumentos = {k: v for k, v in dados.items() if k in conhecidos and k != "filtros"}
        filtros = dados.get("filtros") or {}
        campos_de_filtro = {f.name for f in fields(Filtros)}
        desconhecidos = set(filtros) - campos_de_filtro
        if desconhecidos:
            raise ErroDeLink(f"Filtros desconhecidos: {', '.join(sorted(desconhecidos))}.")
        return cls(filtros=Filtros(**filtros), **argumentos)

    # -----------------------------------------------------------------
    # URL (§8.3, A6)
    # -----------------------------------------------------------------
    def para_url(self) -> dict[str, str]:
        """Só o que difere do padrão, com as chaves curtas e a versão."""
        padrao = Consulta()
        saida: dict[str, str] = {"v": str(VERSAO_URL)}
        for campo, chave in CHAVES.items():
            valor = getattr(self, campo)
            if valor != getattr(padrao, campo) and valor is not None:
                saida[chave] = _texto_do_valor(valor)
        padrao_filtros = Filtros()
        for campo, chave in CHAVES_FILTROS.items():
            valor = getattr(self.filtros, campo)
            if valor != getattr(padrao_filtros, campo) and valor is not None:
                saida[chave] = _texto_do_valor(valor)
        return saida

    @classmethod
    def de_url(cls, parametros: dict[str, Any]) -> "Consulta":
        """Lê o que `para_url` escreveu. Uma URL vazia dá a consulta padrão."""
        bruto = {k: _primeiro(v) for k, v in dict(parametros or {}).items()}
        if not bruto:
            return cls()
        versao = bruto.get("v")
        if versao is None:
            raise ErroDeLink("Link de busca sem versão (falta `v`). Copie o link de novo.")
        try:
            versao = int(versao)
        except (TypeError, ValueError) as e:
            raise ErroDeLink(f"Versão do link inválida: {versao!r}.") from e
        if versao != VERSAO_URL:
            raise ErroDeLink(
                f"Este link foi feito na versão {versao} e o app usa a {VERSAO_URL}. "
                "Refaça a busca."
            )

        padrao, padrao_filtros = cls(), Filtros()
        argumentos = {
            campo: _valor_do_texto(bruto[chave], getattr(padrao, campo), campo)
            for campo, chave in CHAVES.items() if chave in bruto
        }
        filtros = {
            campo: _valor_do_texto(bruto[chave], getattr(padrao_filtros, campo), campo)
            for campo, chave in CHAVES_FILTROS.items() if chave in bruto
        }
        usadas = {"v", *CHAVES.values(), *CHAVES_FILTROS.values()}
        sobrando = sorted(set(bruto) - usadas)
        if sobrando:
            raise ErroDeLink(f"Link com parâmetros desconhecidos: {', '.join(sobrando)}.")
        return cls(filtros=Filtros(**filtros), **argumentos)

    # -----------------------------------------------------------------
    def com_lente(self, dados, nome: str | None) -> "Consulta":
        """Aplica uma lente: ela **só preenche** os filtros, que seguem editáveis (§6.1)."""
        if nome is None:
            return Consulta(**{**asdict_raso(self), "lente": None, "filtros": Filtros(),
                               "pagina": 1})
        filtros = aplicar_lente(dados, nome)
        return Consulta(**{**asdict_raso(self), "lente": nome, "filtros": filtros, "pagina": 1})


def asdict_raso(consulta: Consulta) -> dict:
    """Campos da consulta sem desmontar os filtros (o `asdict` do dataclass desmontaria)."""
    return {f.name: getattr(consulta, f.name) for f in fields(consulta)}


# ---------------------------------------------------------------------------
# Serialização de valores
# ---------------------------------------------------------------------------

def _texto_do_valor(valor) -> str:
    if isinstance(valor, bool):
        return "1" if valor else "0"
    if isinstance(valor, list):
        return SEPARADOR.join(str(v) for v in valor)
    return str(valor)


def _valor_do_texto(texto: str, padrao, campo: str):
    if isinstance(padrao, bool):
        if texto not in ("0", "1"):
            raise ErroDeLink(f"O parâmetro {campo} deveria ser 0 ou 1, veio {texto!r}.")
        return texto == "1"
    if isinstance(padrao, list):
        return [p for p in texto.split(SEPARADOR) if p]
    if isinstance(padrao, int) or campo in ("ano_de", "ano_ate", "srsp_de", "srsp_ate", "semente",
                                            "pagina"):
        try:
            return int(texto)
        except (TypeError, ValueError) as e:
            raise ErroDeLink(f"O parâmetro {campo} deveria ser um número, veio {texto!r}.") from e
    return texto


def _primeiro(valor):
    """`st.query_params` pode devolver lista quando a chave repete."""
    if isinstance(valor, (list, tuple)):
        return valor[0] if valor else ""
    return valor


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def executar(dados, consulta: Consulta, encoder=None, **extra) -> busca.Resposta:
    """Roda a busca da consulta já com os filtros aplicados, e ordena como ela pede."""
    resposta = busca.buscar(
        dados, consulta.texto, modo=consulta.modo, rigor=consulta.rigor, encoder=encoder,
        filtro=consulta.filtros.mascara(dados), **extra,
    )
    if consulta.ordenacao != "relevancia":
        anos = dados.documentos["ano_isolado"]
        decrescente = consulta.ordenacao == "ano_desc"

        def chave(r):
            ano = anos.iloc[r.linha]
            # Documento sem data vai para o fim nas duas ordens.
            ausente = ano is None or ano != ano  # noqa: PLR0124  (teste de NaN)
            valor = 0 if ausente else int(ano)
            return (ausente, -valor if decrescente else valor, r.linha)

        resposta.resultados.sort(key=chave)
        for posicao, r in enumerate(resposta.resultados, start=1):
            r.posicao = posicao
    return resposta
