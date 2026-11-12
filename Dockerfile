# Imagem do app. O GitHub a publica em ghcr.io/saulorrp/catalogo-ahu a cada mudança no main
# (.github/workflows/imagem.yml); o servidor a roda com deploy/docker-compose.yml.
FROM python:3.11-slim

# Usuário sem root.
RUN useradd -m -u 1000 usuario
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=usuario . .
RUN mkdir -p /home/usuario/.cache/ahu-onnx && chown -R usuario /home/usuario/.cache
USER usuario

# O modelo ONNX da consulta é baixado na primeira busca semântica e fica aqui.
ENV AHU_ONNX_CACHE=/home/usuario/.cache/ahu-onnx \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
# AHU_FRONT=web (padrão): FastAPI + HTMX (web/); AHU_FRONT=streamlit: o app.py de antes.
ENV AHU_FRONT=web
CMD ["sh", "deploy/iniciar.sh"]
