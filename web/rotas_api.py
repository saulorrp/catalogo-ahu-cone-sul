import pandas as pd
from fastapi import APIRouter, HTTPException, Request

import web.main as main_module
from ahu.ficha import montar
from web import cache
from web.core import parse_consulta

router = APIRouter()

@router.get("/saude", name="saude")
def saude():
    dados = main_module.dados_app
    if not dados:
        raise HTTPException(status_code=503, detail="Dados não carregados")
    versao = dados.manifesto.get("versao", "desconhecida") if hasattr(dados, "manifesto") else "desconhecida"
    return {"status": "ok", "documentos": len(dados.documentos), "versao": versao}

@router.get("/api/busca", name="api_busca")
def api_busca(request: Request):
    dados = main_module.dados_app
    consulta = parse_consulta(request)
    resposta, _, _ = cache.buscar(dados, consulta, encoder=main_module.encoder_app,
                                  reranker=main_module.reranker_app)

    resultados = []
    for r in resposta.pagina(consulta.pagina, 25):
        linha = dados.documentos.iloc[r.linha]
        resultados.append({
            "doc_uid": r.doc_uid,
            "cota": str(linha.get("reference_code") or linha.get("new_code") or linha.get("old_code")),
            "ano": int(linha["ano_isolado"]) if pd.notna(linha.get("ano_isolado")) else None,
            "posicao": r.posicao
        })

    return {
        "total": resposta.total,
        "pagina": consulta.pagina,
        "resultados": resultados
    }

@router.get("/api/doc/{doc_uid}", name="api_doc")
def api_doc(doc_uid: str):
    dados = main_module.dados_app
    try:
        ficha = montar(dados, doc_uid)
    except KeyError:
        raise HTTPException(status_code=404, detail="Documento não encontrado")

    return {
        "doc_uid": ficha.doc_uid,
        "titulo": ficha.titulo,
        "resumo": ficha.resumo,
        "cotas": ficha.cotas,
        "pessoas": [{"nome": p.nome, "tipo": p.tipo} for p in ficha.pessoas]
    }

@router.get("/api/sugerir", name="api_sugerir")
def api_sugerir(campo: str, q: str = ""):
    dados = main_module.dados_app
    q = q.lower()
    sugestoes = []
    if campo == "remetente":
        from ahu.filtros import _opcoes_de_pessoas
        pessoas = _opcoes_de_pessoas(dados)
        for p in pessoas:
            if any(q in b.lower() for b in p["busca"]):
                sugestoes.append({"id": p["pessoa_uid"], "texto": p["nome"]})
                if len(sugestoes) >= 10:
                    break
    elif campo == "destinatario":
        if "recipient_name" in dados.documentos.columns:
            dest = dados.documentos["recipient_name"].dropna().unique()
            sugestoes = [{"id": d, "texto": d} for d in dest if q in d.lower()][:10]

    return sugestoes
