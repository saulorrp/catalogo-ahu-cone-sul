#!/bin/sh
# Escolhe a interface do container: AHU_FRONT=web (padrão) ou streamlit.
if [ "$AHU_FRONT" = "streamlit" ]; then
    exec streamlit run app.py --server.port=7860 --server.address=0.0.0.0 \
        --server.headless=true --server.enableXsrfProtection=false
fi
exec uvicorn web.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*'
