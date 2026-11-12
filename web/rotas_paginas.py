import json

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

import web.main as main_module
from ahu.amostras import ErroDeAmostra, amostra_diacronica, amostra_estratificada, plano_de_coleta
from ahu.consulta import sem_duplicatas
from ahu.conversor import converter, converter_lista, tabela_da_cifra
from ahu.graficos import barras_por_decada, barras_por_pais
from ahu.lista_ahu import pdf_da_selecao
from ahu.pessoas import listar, listar_coletivos, painel_diatopico
from ahu.pessoas import pagina as pagina_pessoa
from ahu.selecao import Selecao, maximo_na_url
from web import cache
from web.core import COOKIE_SELECAO, selecao_do_pedido, templates

router = APIRouter()


def linha_do_tempo(dados) -> dict:
    """Documentos por década no catálogo inteiro, para o gráfico da aba Sobre.

    O eixo vai da primeira à última década com peso: as poucas datas muito fora (erros de
    leitura do ano, documentos soltos) entram na conta de `fora`, e não esticam o gráfico.
    """
    visiveis = sem_duplicatas(dados, None)
    docs = dados.documentos if visiveis is None else dados.documentos[visiveis]
    anos = pd.to_numeric(docs["ano_isolado"], errors="coerce")
    com_data = anos.dropna().astype(int)
    if com_data.empty:
        return {"decadas": [], "total": len(docs), "sem_data": int(anos.isna().sum()), "fora": 0,
                "maximo": 0, "teto": 0}
    inicio = int(com_data.quantile(0.002)) // 10 * 10
    fim = int(com_data.quantile(0.998)) // 10 * 10
    contagem = (com_data // 10 * 10).value_counts()
    decadas = [(d, int(contagem.get(d, 0))) for d in range(inicio, fim + 10, 10)]
    dentro = sum(n for _, n in decadas)
    maximo = max(n for _, n in decadas)
    # O teto do eixo: o primeiro número "redondo" (1, 2, 3, 4, 5 ou 8 vezes uma potência de dez)
    # acima da maior barra, para as quatro linhas de grade caírem em valores legíveis.
    potencia = 10 ** max(0, len(str(maximo)) - 1)
    teto = next(f * potencia for f in (1, 2, 3, 4, 5, 8, 10) if f * potencia >= maximo)
    return {"decadas": decadas, "total": len(docs), "sem_data": int(anos.isna().sum()),
            "fora": int(len(com_data) - dentro), "maximo": maximo, "teto": int(teto),
            "inicio": inicio, "fim": fim + 9}


@router.get("/pessoas", name="pessoas")
def pessoas(request: Request, q: str = "", ordem: str = "documentos", tipo: str = "pessoas"):
    dados = main_module.dados_app
    if tipo == "pessoas":
        lista = listar(dados, texto=q, ordem=ordem)
    else:
        lista = listar_coletivos(dados, texto=q, ordem=ordem)
    return templates.TemplateResponse(request, "pessoas.html", {
        "lista": lista, "q": q, "ordem": ordem, "tipo": tipo
    })


@router.get("/pessoa/{pessoa_uid}", name="pessoa")
def pessoa(request: Request, pessoa_uid: str):
    dados = main_module.dados_app
    try:
        pag = pagina_pessoa(dados, pessoa_uid)
    except KeyError:
        return HTMLResponse("Pessoa não encontrada", status_code=404)
    def limpo(valor):
        return None if valor is None or pd.isna(valor) else valor

    documentos = [{
        "uid": linha.doc_uid,
        "ano": int(linha.ano) if limpo(linha.ano) is not None else None,
        "tipologia": limpo(linha.tipologia),
        "destinatario": limpo(linha.destinatario),
        "autoria_verificada": bool(limpo(linha.autoria_verificada)),
    } for linha in pag.documentos.itertuples()]
    linha_do_tempo = [(int(r.ano), int(r.documentos)) for r in pag.linha_do_tempo.itertuples()]
    return templates.TemplateResponse(request, "pessoa.html", {
        "pagina": pag, "documentos": documentos, "linha_do_tempo": linha_do_tempo,
        "maximo_por_ano": max((n for _, n in linha_do_tempo), default=0),
    })


@router.get("/conversor", name="conversor")
def conversor(request: Request, q: str = "", lote: str = ""):
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
    return selecao_do_pedido(request, dados)


def _gravar_selecao(response, sel: Selecao, dados) -> None:
    response.set_cookie(COOKIE_SELECAO, sel.para_url(dados) or "", max_age=31536000, path="/")


def _resposta_da_selecao(request: Request, dados, sel: Selecao, doc_uid: str, aviso: str = ""):
    """O que muda na tela quando a seleção muda: o botão do documento, a gaveta e o número da aba.

    Os três vêm no mesmo pedaço de HTML; o HTMX troca o botão clicado e leva os outros dois ao
    lugar pelo `id` (hx-swap-oob).
    """
    response = templates.TemplateResponse(request, "_selecao_mudou.html", {
        "uid": doc_uid,
        "marcado": doc_uid in sel,
        "botao_fora": request.query_params.get("de") == "gaveta",
        "linhas_selecao": sel.linhas(dados),
        "total_selecao": len(sel),
        "aviso_selecao": aviso,
    })
    _gravar_selecao(response, sel, dados)
    return response


@router.get("/selecao", name="selecao")
def selecao(request: Request):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    linhas = sel.linhas(dados)
    return templates.TemplateResponse(request, "selecao.html", {
        "linhas": linhas, "total": len(sel), "maximo": maximo_na_url(dados)})


@router.post("/selecao/adicionar/{doc_uid}", name="adicionar_selecao")
def adicionar_selecao(request: Request, doc_uid: str):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    aviso = ""
    sobraram = sel.adicionar(doc_uid)
    if sobraram or sel.para_url(dados) is None:
        # A seleção viaja no cookie do navegador, e ele tem um teto. Passou do teto, o documento
        # não entra — antes isto apagava a seleção inteira sem avisar.
        sel.remover(doc_uid)
        aviso = (f"A seleção está cheia ({len(sel)} documentos). Baixe a lista em PDF e esvazie "
                 "a seleção para continuar.")
    return _resposta_da_selecao(request, dados, sel, doc_uid, aviso)


@router.post("/selecao/remover/{doc_uid}", name="remover_selecao")
def remover_selecao(request: Request, doc_uid: str):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    sel.remover(doc_uid)
    if request.query_params.get("volta") == "1":
        # Pedido da página Seleção, por formulário comum: volta a ela já sem o documento.
        response = RedirectResponse(url=request.url_for("selecao"), status_code=303)
        _gravar_selecao(response, sel, dados)
        return response
    return _resposta_da_selecao(request, dados, sel, doc_uid)


@router.post("/selecao/limpar", name="limpar_selecao")
def limpar_selecao(request: Request):
    destino = request.url_for("busca") if request.query_params.get("volta") == "busca" \
        else request.url_for("selecao")
    response = RedirectResponse(url=destino, status_code=303)
    response.set_cookie(COOKIE_SELECAO, "", max_age=0, path="/")
    return response


@router.get("/selecao/pdf", name="selecao_pdf")
def selecao_pdf(request: Request):
    dados = main_module.dados_app
    sel = get_selecao(request, dados)
    pdf_bytes = pdf_da_selecao(dados, sel)
    return Response(content=pdf_bytes, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=lista_consulta_AHU.pdf"
    })


@router.get("/amostras", name="amostras")
def amostras(
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
        # O reranker só muda a ordem; a amostra usa o conjunto, que é o mesmo sem ele.
        resposta = executar(dados, consulta, encoder=main_module.encoder_app)
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
def amostras_plano(
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
            resposta = executar(dados, consulta, encoder=main_module.encoder_app)
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
def painel(request: Request, por: str = "documento"):
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
def sobre(request: Request):
    dados = main_module.dados_app
    linha = cache.por_dados(dados, "linha_do_tempo", lambda: linha_do_tempo(dados))
    return templates.TemplateResponse(request, "sobre.html", {"linha": linha})


@router.get("/referencias", name="referencias")
def referencias(request: Request):
    return templates.TemplateResponse(request, "referencias.html", {})
