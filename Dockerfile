# Imagem do Space do Hugging Face (tipo Docker). Também roda em qualquer serviço de contêiner.
FROM python:3.11-slim

# O Hugging Face roda o contêiner com o uid 1000.
RUN useradd -m -u 1000 usuario
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=usuario . .
USER usuario

# O modelo ONNX da consulta é baixado na primeira busca semântica e fica aqui.
ENV AHU_ONNX_CACHE=/home/usuario/.cache/ahu-onnx \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false"]
