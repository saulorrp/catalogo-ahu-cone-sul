import json

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

import web.main as main_module
from ahu.amostras import ErroDeAmostra, amostra_diacronica, amostra_estratificada, plano_de_coleta
from ahu.conversor import converter, converter_lista, tabela_da_cifra
from ahu.graficos import barras_por_decada, barras_por_pais
from ahu.lista_ahu import pdf_da_selecao
from ahu.pessoas import listar, listar_coletivos, painel_diatopico
from ahu.pessoas import pagina as pagina_pessoa
from ahu.selecao import Selecao
from web.core import templates

router = APIRouter()


@router.get("/pessoas", name="pessoas")
async def pessoas(request: Request, q: str = "", ordem: str = "documentos", tipo: str = "pessoas"):
    dados = main_module.dados_app
    if tipo == "pessoas":
        lista = listar(dados, texto=q, ordem=ordem)
    else:
        lista = listar_coletivos(dados, texto=q, ordem=ordem)
    return templates.TemplateResponse(request, "pessoas.html", {
        "lista": lista, "q": q, "ordem": ordem, "tipo": tipo
    })


@router.get("/pessoa/{pessoa_uid}", name="pessoa")
async def pessoa(request: Request, pessoa_uid: str):
    dados = main_module.dados_app
    try:
        pag = pagina_pessoa(dados, pessoa_uid)
    except KeyError:
        return HTMLResponse("Pessoa não encontrada", status_code=404)
    return templates.TemplateResponse(request, "pessoa.html", {"pagina": pag})


@router.get("/conversor", name="conversor")
async def conversor(request: Request, q: str = "", lote: str = ""):
    dados = main_module.dados_app
    conversoes = []
    if q:
        conversoes = converter(q, dados)
    elif lote:
        conversoes = converter_lista(lote, dados)
    cifra = tabela_da_cifra().to_dict("records")
    return templates.TemplateResponse(request, "conversor.html", {
        "q": q, "lote": lote, "conversoes": conversoes, "cifra": cifra
    })


def get_selecao(request: Request, dados) -> Selecao:
    s = request.query_params.get("s") or request.cookies.get("selecao")
    selecao, _ = Selecao.de_url(s, dados)
    return selecao


@router.get("/selecao", name="selecao")
async def selecao(request: Request):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    linhas = sel.linhas(dados)
    return templates.TemplateResponse(request, "selecao.html", {"linhas": linhas, "total": len(sel)})


@router.post("/selecao/adicionar/{doc_uid}", name="adicionar_selecao")
async def adicionar_selecao(request: Request, doc_uid: str):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    sel.adicionar(doc_uid)
    novo_s = sel.para_url(dados) or ""
    response = HTMLResponse("<button disabled>Na seleção</button>")
    response.set_cookie("selecao", novo_s, max_age=31536000, path="/")
    return response


@router.post("/selecao/remover/{doc_uid}", name="remover_selecao")
async def remover_selecao(request: Request, doc_uid: str):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    sel.remover(doc_uid)
    novo_s = sel.para_url(dados) or ""
    response = HTMLResponse("")
    response.set_cookie("selecao", novo_s, max_age=31536000, path="/")
    return response


@router.post("/selecao/limpar", name="limpar_selecao")
async def limpar_selecao(request: Request):
    response = RedirectResponse(url=request.url_for("selecao"), status_code=303)
    response.set_cookie("selecao", "", max_age=0, path="/")
    return response


@router.get("/selecao/pdf", name="selecao_pdf")
async def selecao_pdf(request: Request):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    pdf_bytes = pdf_da_selecao(dados, sel)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=lista_consulta_AHU.pdf"
    })


@router.get("/amostras", name="amostras")
async def amostras(
    request: Request,
    tipo: str = "diacronica",
    pessoa_uid: str = "",
    max_ano: int = 2,
    estratos: str = "decada",
    por_estrato: int = 3,
    semente: int = 42
):
    dados = main_module.dados_app
    amostra = None
    erro = None
    if tipo == "diacronica" and pessoa_uid:
        try:
            amostra = amostra_diacronica(
                dados, pessoa_uid, maximo_por_ano=max_ano if max_ano > 0 else None
            )
        except Exception as e:
            erro = str(e)
    elif tipo == "estratificada":
        from ahu.consulta import executar
        from web.core import parse_consulta
        consulta = parse_consulta(request)
        resposta = executar(
            dados, consulta, encoder=main_module.encoder_app, **main_module.extra_busca()
        )
        uids = [r.doc_uid for r in resposta.resultados]
        estratos_list = [e.strip() for e in estratos.split(",") if e.strip()]
        if not estratos_list:
            estratos_list = ["decada"]
        try:
            amostra = amostra_estratificada(dados, uids, estratos_list, por_estrato, semente)
        except Exception as e:
            erro = str(e)

    pessoas_opcoes = [{"id": p.pessoa_uid, "nome": p.nome} for p in listar(dados)]
    return templates.TemplateResponse(request, "amostras.html", {
        "tipo": tipo, "pessoa_uid": pessoa_uid, "max_ano": max_ano,
        "estratos": estratos, "por_estrato": por_estrato, "semente": semente,
        "amostra": amostra, "erro": erro, "pessoas_opcoes": pessoas_opcoes
    })


@router.get("/amostras/plano", name="amostras_plano")
async def amostras_plano(
    request: Request,
    tipo: str = "diacronica",
    pessoa_uid: str = "",
    max_ano: int = 2,
    estratos: str = "decada",
    por_estrato: int = 3,
    semente: int = 42
):
    dados = main_module.dados_app
    try:
        if tipo == "diacronica":
            amostra = amostra_diacronica(
                dados, pessoa_uid, maximo_por_ano=max_ano if max_ano > 0 else None
            )
        else:
            from ahu.consulta import executar
            from web.core import parse_consulta
            consulta = parse_consulta(request)
            resposta = executar(
                dados, consulta, encoder=main_module.encoder_app, **main_module.extra_busca()
            )
            uids = [r.doc_uid for r in resposta.resultados]
            estratos_list = [e.strip() for e in estratos.split(",") if e.strip()]
            amostra = amostra_estratificada(dados, uids, estratos_list, por_estrato, semente)
    except ErroDeAmostra as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    plano = plano_de_coleta(dados, amostra)
    return Response(
        content=json.dumps(plano, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=plano_coleta.eins.json"}
    )


@router.get("/painel", name="painel")
async def painel(request: Request, por: str = "documento"):
    dados = main_module.dados_app
    painel_data = painel_diatopico(dados, por)

    fig_pais = barras_por_pais(painel_data, "light")
    fig_decada = barras_por_decada(painel_data, "light")

    return templates.TemplateResponse(request, "painel.html", {
        "por": por,
        "painel": painel_data,
        "fig_pais_json": fig_pais.to_json(),
        "fig_decada_json": fig_decada.to_json(),
        "paises_records": painel_data.paises.to_dict("records"),
        "decadas_records": painel_data.decadas.to_dict("records")
    })


@router.get("/sobre", name="sobre")
async def sobre(request: Request):
    return templates.TemplateResponse(request, "sobre.html", {})


@router.get("/referencias", name="referencias")
async def referencias(request: Request):
    return templates.TemplateResponse(request, "referencias.html", {})
