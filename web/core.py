import pathlib
import re
from urllib.parse import urlencode

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ahu.consulta import Consulta
from ahu.selecao import SEPARADOR_URL, Selecao

BASE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

COOKIE_SELECAO = "selecao"


def _marcas(texto: str) -> str:
    """`ahu.ficha.marcar_termos` marca com **…** (já sobre texto escapado): vira <mark>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", texto or "")


def _milhar(numero) -> str:
    """167137 -> 167.137, como se escreve em português."""
    try:
        return f"{int(numero):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(numero)


def _id_seguro(texto: str) -> str:
    """Um `doc_uid` em forma que serve de `id` no HTML (o original tem dois-pontos)."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", str(texto))


def conta_selecao(request: Request) -> int:
    """Quantos documentos há na seleção, só pelo cookie: é o número que a aba mostra."""
    texto = request.cookies.get(COOKIE_SELECAO) or ""
    return sum(1 for pedaco in texto.split(SEPARADOR_URL) if pedaco.strip())


templates.env.filters["marcas"] = _marcas
templates.env.filters["milhar"] = _milhar
templates.env.filters["id_seguro"] = _id_seguro
templates.env.globals["conta_selecao"] = conta_selecao


def selecao_do_pedido(request: Request, dados) -> Selecao:
    texto = request.query_params.get("s") or request.cookies.get(COOKIE_SELECAO)
    selecao, _ = Selecao.de_url(texto, dados)
    return selecao


def parse_consulta(request: Request) -> Consulta:
    params = {}
    for k, v in request.query_params.multi_items():
        # O formulário manda todos os campos, inclusive os vazios (ano_de="", etc.); vazio é
        # "sem filtro", e a Consulta recusa um número vazio, o que antes zerava a consulta toda.
        if not v.strip() or k == "fino":   # `fino` é do pedido em dois tempos, não da consulta
            continue
        if k in params:
            params[k] = f"{params[k]}|{v}"
        else:
            params[k] = v
    if "v" not in params:
        params["v"] = "2"
    try:
        return Consulta.de_url(params)
    except Exception:
        return Consulta()


def query_sem_pagina(request: Request) -> str:
    """A query do pedido sem os campos vazios nem a página: é o que "Carregar mais" repete."""
    pares = [(k, v) for k, v in request.query_params.multi_items()
             if v.strip() and k not in ("p", "fino")]
    return urlencode(pares)
