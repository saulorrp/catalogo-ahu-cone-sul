import os
import sqlite3
from datetime import datetime

from fastapi import APIRouter, Request

import web.main as main_module
from ahu.ficha import montar
from ahu.filtros import opcoes
from web import cache
from web.core import parse_consulta, query_sem_pagina, selecao_do_pedido, templates

router = APIRouter()

POR_PAGINA = 25


def log_busca(texto, num_resultados):
    log_dir = os.environ.get("AHU_LOG_DIR")
    if not log_dir:
        return
    db_path = os.path.join(log_dir, "buscas.sqlite")
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS buscas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                texto TEXT,
                resultados INTEGER,
                posicao_clicada INTEGER
            )
        """)
        conn.execute("INSERT INTO buscas (data, texto, resultados) VALUES (?, ?, ?)",
                     (datetime.utcnow().isoformat(), texto, num_resultados))
        conn.commit()
        conn.close()
    except Exception:
        pass


def opcoes_dos_filtros(dados) -> dict:
    def calcular():
        o = opcoes(dados)
        o["tem_destinatario"] = "recipient_name" in dados.documentos.columns
        o["nome_da_pessoa"] = {p["pessoa_uid"]: p["nome"] for p in o["pessoas"]}
        return o
    return cache.por_dados(dados, "opcoes", calcular)


def _contexto(request: Request, esperar: bool) -> dict:
    dados = main_module.dados_app
    consulta = parse_consulta(request)
    resposta, facetas, pendente = cache.buscar(
        dados, consulta, encoder=main_module.encoder_app, reranker=main_module.reranker_app,
        esperar=esperar or consulta.pagina > 1)
    fichas = [(r, montar(dados, r.doc_uid, resposta.termos_consulta))
              for r in resposta.pagina(consulta.pagina, POR_PAGINA)]
    return {
        "consulta": consulta,
        "resposta": resposta,
        "facetas": facetas,
        "pendente": pendente,
        "query": query_sem_pagina(request),
        "pagina_atual": consulta.pagina,
        "total_paginas": max(1, -(-resposta.total // POR_PAGINA)),
        "fichas": fichas,
        "na_selecao": set(selecao_do_pedido(request, dados).uids),
    }


@router.get("/", name="busca_raiz")
@router.get("/busca", name="busca")
def busca(request: Request):
    dados = main_module.dados_app
    contexto = _contexto(request, esperar=False)
    consulta = contexto["consulta"]
    if consulta.texto:
        log_busca(consulta.texto, contexto["resposta"].total)
    selecao = selecao_do_pedido(request, dados)
    contexto.update({
        "opcoes": opcoes_dos_filtros(dados),
        "linhas_selecao": selecao.linhas(dados),
        "inicio": not consulta.texto and consulta.filtros.vazio(),
    })
    return templates.TemplateResponse(request, "busca.html", contexto)


@router.get("/resultados", name="resultados")
def resultados(request: Request):
    """Um pedaço da lista: a página seguinte ("Carregar mais") ou a ordem fina do reranker."""
    return templates.TemplateResponse(request, "_resultados.html", _contexto(request, esperar=True))
