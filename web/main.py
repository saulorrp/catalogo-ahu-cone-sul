import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from ahu.dados import ErroDados, carregar
from web.rotas_api import router as router_api
from web.rotas_busca import router as router_busca
from web.rotas_doc import router as router_doc
from web.rotas_paginas import router as router_paginas

# Nomes de rota para url_for:
# "busca" (GET / e /busca)
# "resultados" (GET /resultados)
# "doc" (GET /doc/{doc_uid})
# "saude" (GET /saude)
# "api_busca" (GET /api/busca)
# "api_doc" (GET /api/doc/{doc_uid})
# "api_sugerir" (GET /api/sugerir)
# "pessoas" (GET /pessoas)
# "pessoa" (GET /pessoa/{pessoa_uid})
# "conversor" (GET /conversor)
# "selecao" (GET /selecao)
# "amostras" (GET /amostras)
# "painel" (GET /painel)
# "sobre" (GET /sobre)
# "referencias" (GET /referencias)

dados_app = None
encoder_app = None
reranker_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dados_app, encoder_app, reranker_app
    try:
        dados_app = carregar()
    except ErroDados as e:
        print(f"Erro ao carregar dados: {e}")

    sem_semantica = os.environ.get("AHU_SEM_SEMANTICA", "").strip().lower() in ("1", "sim", "true")
    if not sem_semantica and dados_app:
        from ahu.consulta_onnx import criar_encoder_de_dados
        cache = {}

        def encoder_preguicoso(textos):
            # Carrega o modelo na primeira busca e o reaproveita nas seguintes.
            if "enc" not in cache:
                cache["enc"] = criar_encoder_de_dados(dados_app)
            return cache["enc"](textos)
        encoder_app = encoder_preguicoso
    elif dados_app:
        from ahu.dados import APP_DADOS
        from ahu.estatico import EncoderEstatico, existe
        if existe(APP_DADOS):
            estatico = EncoderEstatico(APP_DADOS)
            encoder_app = estatico

    try:
        from ahu import reranker as mod_reranker
        if mod_reranker.ligado():
            reranker_app = mod_reranker.com_memoria(mod_reranker.criar_reranker())
    except ImportError:
        pass

    if dados_app and os.environ.get("AHU_SEM_AQUECIMENTO", "").strip().lower() not in ("1", "sim"):
        import threading
        threading.Thread(target=_aquecer, daemon=True).start()

    yield

    dados_app = None
    encoder_app = None
    reranker_app = None

def _aquecer():
    """Deixa prontos, logo que o servidor sobe, os modelos e a página inicial.

    Sem isto a primeira pessoa a buscar depois de uma atualização espera a carga do e5, do
    reranker e dos índices. Roda fora do caminho dos pedidos; se falhar, a busca carrega tudo na
    primeira consulta, como antes.
    """
    try:
        from ahu.consulta import Consulta
        from web import cache
        from web.rotas_busca import opcoes_dos_filtros
        opcoes_dos_filtros(dados_app)
        resposta, _, _ = cache.buscar(dados_app, Consulta(), encoder=encoder_app)
        if resposta.resultados:
            from ahu.ficha import montar
            montar(dados_app, resposta.resultados[0].doc_uid)   # monta o índice de pessoas
        cache.buscar(dados_app, Consulta(texto="carta do governador"), encoder=encoder_app,
                     reranker=reranker_app)
    except Exception as e:   # pragma: no cover - depende de rede e de modelo
        print(f"Aquecimento incompleto: {e}")


def criar_app(dados_injetados=None):
    global dados_app
    if dados_injetados:
        dados_app = dados_injetados

    dev_docs = os.environ.get("AHU_DEV_DOCS", "").strip().lower() in ("1", "sim", "true")
    docs_kwargs = {} if dev_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    inner_app = FastAPI(lifespan=lifespan if not dados_injetados else None, **docs_kwargs)
    # As páginas são HTML repetitivo (a de resultados passa de 100 kB): comprimidas, viajam em
    # um quinto do tamanho.
    inner_app.add_middleware(GZipMiddleware, minimum_size=1024)

    inner_app.include_router(router_busca)
    inner_app.include_router(router_doc)
    inner_app.include_router(router_api)
    inner_app.include_router(router_paginas)

    import pathlib
    static_dir = pathlib.Path(__file__).parent / "static"
    inner_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    root_path = os.environ.get("AHU_ROOT_PATH", "").rstrip("/")
    if root_path:
        # O Starlette não roda o lifespan de um app montado: o pai o assume.
        app = FastAPI(lifespan=lifespan if not dados_injetados else None, **docs_kwargs)
        app.mount(root_path, inner_app)
        return app
    return inner_app

app = criar_app()

