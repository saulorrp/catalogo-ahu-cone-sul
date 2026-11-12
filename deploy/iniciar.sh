#!/bin/sh
# Escolhe a interface do container: AHU_FRONT=web (padrão) ou streamlit.
if [ "$AHU_FRONT" = "streamlit" ]; then
    exec streamlit run app.py --server.port=7860 --server.address=0.0.0.0 \
        --server.headless=true --server.enableXsrfProtection=false
fi
# O servidor já define STREAMLIT_SERVER_BASE_URL_PATH=ahu para o Funnel; vale também para o web.
if [ -z "$AHU_ROOT_PATH" ] && [ -n "$STREAMLIT_SERVER_BASE_URL_PATH" ]; then
    export AHU_ROOT_PATH="/${STREAMLIT_SERVER_BASE_URL_PATH#/}"
fi
exec uvicorn web.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*'
