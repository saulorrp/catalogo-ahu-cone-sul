import os
import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Request

import web.main as main_module
from ahu.consulta import executar
from ahu.ficha import montar
from ahu.filtros import opcoes
from web.core import parse_consulta, templates

router = APIRouter()

def calcular_facetas(dados, resposta):
    if not resposta.resultados:
        return {}
    linhas = [r.linha for r in resposta.resultados]
    df = dados.documentos.iloc[linhas]

    facetas = {
        "colecao": df["colecao"].value_counts().to_dict(),
        "folder": df["folder"].value_counts().to_dict(),
        "tipologia": df["tipologia"].value_counts().to_dict(),
    }

    anos = pd.to_numeric(df["ano_isolado"], errors="coerce").dropna()
    if not anos.empty:
        decadas = (anos // 10 * 10).astype(int)
        facetas["decada"] = decadas.value_counts().to_dict()
        counts = anos.value_counts().to_dict()
        facetas["histograma"] = {int(k): int(v) for k, v in counts.items()}
    else:
        facetas["decada"] = {}
        facetas["histograma"] = {}

    return facetas

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

@router.get("/", name="busca_raiz")
@router.get("/busca", name="busca")
async def busca(request: Request):
    dados = main_module.dados_app
    consulta = parse_consulta(request)

    resposta = executar(dados, consulta, encoder=main_module.encoder_app, **main_module.extra_busca())

    if consulta.texto:
        log_busca(consulta.texto, resposta.total)

    facetas = calcular_facetas(dados, resposta)
    opcoes_filtros = opcoes(dados)
    opcoes_filtros["tem_destinatario"] = "recipient_name" in dados.documentos.columns

    fichas = []
    for r in resposta.pagina(consulta.pagina, 25):
        fichas.append((r, montar(dados, r.doc_uid, resposta.termos_consulta)))

    return templates.TemplateResponse(request, "busca.html", {
        "consulta": consulta,
        "resposta": resposta,
        "facetas": facetas,
        "opcoes": opcoes_filtros,
        "pagina_atual": consulta.pagina,
        "total_paginas": max(1, -(-resposta.total // 25)),
        "fichas": fichas
    })

@router.get("/resultados", name="resultados")
async def resultados(request: Request):
    dados = main_module.dados_app
    consulta = parse_consulta(request)
    resposta = executar(dados, consulta, encoder=main_module.encoder_app, **main_module.extra_busca())

    fichas = []
    for r in resposta.pagina(consulta.pagina, 25):
        fichas.append((r, montar(dados, r.doc_uid, resposta.termos_consulta)))

    return templates.TemplateResponse(request, "_resultados.html", {
        "consulta": consulta,
        "resposta": resposta,
        "pagina_atual": consulta.pagina,
        "total_paginas": max(1, -(-resposta.total // 25)),
        "fichas": fichas
    })
