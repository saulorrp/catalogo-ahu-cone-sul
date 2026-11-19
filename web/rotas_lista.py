"""As rotas da Lista do eins zwei (ver `ahu/lista_eins.py`).

A Lista só existe no endereço privado do eins zwei. Duas variáveis de ambiente a ligam:
`EINS_LISTA_DIR` (a pasta dos registros) e `EINS_LISTA_TOKEN`. O proxy do endereço privado
(nginx, só na rede Tailscale) acrescenta a todo pedido o cabeçalho `X-Eins-Lista` com o token; o
endereço público nunca o manda. Sem as variáveis, as rotas respondem 404; sem o cabeçalho certo,
403, e o catálogo não mostra o botão "Σ Adicionar à lista".

Os pedidos que mudam a Lista exigem um cabeçalho que um formulário de outro site não consegue
mandar (`HX-Request` nos botões, `Content-Type: application/json` na API), para que uma página
qualquer aberta no navegador não escreva nela.
"""
import hmac
import json
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

import web.main as main_module
from ahu.busca import chave_cota, tabela_de_cotas
from ahu.lista_eins import FORMATO, Lista, cota_curta, cota_valida
from web.core import templates

router = APIRouter()

CABECALHO = "x-eins-lista"
MAX_CORPO = 8 * 1024 * 1024

_listas: dict[str, Lista] = {}


def _configurada() -> tuple[str, str] | None:
    pasta = os.environ.get("EINS_LISTA_DIR", "").strip()
    token = os.environ.get("EINS_LISTA_TOKEN", "").strip()
    return (pasta, token) if pasta and token else None


def lista_do_pedido(request: Request) -> Lista | None:
    """A Lista, se este pedido veio pelo endereço privado; None em qualquer outro caso."""
    config = _configurada()
    if not config:
        return None
    pasta, token = config
    enviado = request.headers.get(CABECALHO, "")
    if not enviado or not hmac.compare_digest(enviado.encode(), token.encode()):
        return None
    if pasta not in _listas:
        _listas[pasta] = Lista(pasta)
    return _listas[pasta]


def _exigir(request: Request) -> Lista:
    if not _configurada():
        raise HTTPException(status_code=404, detail="Lista desligada")
    lista = lista_do_pedido(request)
    if lista is None:
        raise HTTPException(status_code=403, detail="A Lista só abre no endereço privado")
    return lista


def estados_na_lista(request: Request) -> dict[str, str] | None:
    """Para os templates: doc_uid -> estado, ou None quando o botão da Lista não aparece."""
    if not hasattr(request.state, "estados_lista"):
        lista = lista_do_pedido(request)
        request.state.estados_lista = lista.estados_por_doc() if lista else None
    return request.state.estados_lista


templates.env.globals["estados_na_lista"] = estados_na_lista


def _ficha(dados, doc_uid: str) -> dict:
    from web.rotas_api import _serializar_ficha
    return _serializar_ficha(dados, doc_uid)


def _ficha_da_cota(dados, cota: str) -> dict | None:
    if dados is None:
        return None
    linhas = tabela_de_cotas(dados).get(chave_cota(f"PT/AHU/CU/{_referencia(cota)}") or "", [])
    if not linhas:
        return None
    return _ficha(dados, str(dados.documentos.iloc[linhas[0]]["doc_uid"]))


def _referencia(cota: str) -> str:
    serie, caixa, doc = cota.split("-")
    return f"{serie.replace('.', '-')}/{caixa}/{doc}"


def _cota_do_doc(dados, doc_uid: str) -> tuple[str | None, dict]:
    ficha = _ficha(dados, doc_uid)
    for rotulo in ("Código de referência", "Código atual"):
        cota = cota_curta(ficha["cotas"].get(rotulo))
        if cota:
            return cota, ficha
    return None, ficha


# ---- API (o eins zwei) ----------------------------------------------------------------------

@router.get("/api/lista", name="api_lista")
def api_lista(request: Request):
    lista = _exigir(request)
    return {"formato": FORMATO, "itens": lista.listar()}


@router.get("/api/lista/{cota}", name="api_lista_cota")
def api_lista_cota(request: Request, cota: str):
    lista = _exigir(request)
    registro = lista.ler(cota) if cota_valida(cota) else None
    if registro is None:
        raise HTTPException(status_code=404, detail="Processo fora da Lista")
    return registro


@router.put("/api/lista/{cota}", name="api_lista_atualizar")
async def api_lista_atualizar(request: Request, cota: str):
    lista = _exigir(request)
    if not cota_valida(cota):
        raise HTTPException(status_code=400, detail="Use a cota curta (089-0010-00955)")
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=415, detail="Mande JSON")
    corpo = await request.body()
    if len(corpo) > MAX_CORPO:
        raise HTTPException(status_code=413, detail="Registro grande demais")
    try:
        mudanca = json.loads(corpo or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON inválido")
    ficha = None
    if lista.ler(cota) is None and not isinstance(mudanca.get("ficha") if isinstance(mudanca, dict) else None, dict):
        try:
            ficha = _ficha_da_cota(main_module.dados_app, cota)
        except KeyError:
            ficha = None
    try:
        return lista.atualizar(cota, mudanca, ficha_nova=ficha)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/lista/{cota}", name="api_lista_remover")
def api_lista_remover(request: Request, cota: str):
    lista = _exigir(request)
    if not cota_valida(cota) or not lista.remover(cota):
        raise HTTPException(status_code=404, detail="Processo fora da Lista")
    return Response(status_code=204)


# ---- O botão "Σ Adicionar à lista" (o catálogo) -----------------------------------------------

def _botao(request: Request, lista: Lista, doc_uid: str, aviso: str = ""):
    request.state.estados_lista = lista.estados_por_doc()
    return templates.TemplateResponse(request, "_botao_lista.html", {"uid": doc_uid, "aviso_lista": aviso})


def _mudar(request: Request, doc_uid: str, adicionar: bool):
    lista = _exigir(request)
    if request.headers.get("hx-request") != "true":
        raise HTTPException(status_code=400, detail="Use o botão do catálogo")
    dados = main_module.dados_app
    try:
        cota, ficha = _cota_do_doc(dados, doc_uid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Documento não encontrado")
    if not cota:
        return _botao(request, lista, doc_uid, "Este documento não tem cota do AHU que a Lista reconheça.")
    if adicionar:
        lista.atualizar(cota, {"ficha": ficha, "evento": {"tipo": "catalogo", "acao": "adicionar"}})
    else:
        lista.remover(cota)
    return _botao(request, lista, doc_uid)


@router.post("/lista/adicionar/{doc_uid}", name="adicionar_lista")
def adicionar_lista(request: Request, doc_uid: str):
    return _mudar(request, doc_uid, True)


@router.post("/lista/remover/{doc_uid}", name="remover_lista")
def remover_lista(request: Request, doc_uid: str):
    return _mudar(request, doc_uid, False)

