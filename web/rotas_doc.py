from fastapi import APIRouter, HTTPException, Request

import web.main as main_module
from ahu.ficha import montar
from web.core import selecao_do_pedido, templates

router = APIRouter()

@router.get("/doc/{doc_uid}", name="doc")
def doc(request: Request, doc_uid: str, q: str = ""):
    dados = main_module.dados_app
    try:
        termos = q.split() if q else []
        ficha = montar(dados, doc_uid, termos)
    except KeyError:
        raise HTTPException(status_code=404, detail="Documento não encontrado")

    return templates.TemplateResponse(request, "doc.html", {
        "ficha": ficha,
        "marcado": doc_uid in selecao_do_pedido(request, dados),
    })
