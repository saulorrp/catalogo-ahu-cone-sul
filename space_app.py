"""space_app.py — entrada do Space do Hugging Face (SDK Gradio, plano gratuito).

O Space gratuito só aceita os SDKs Gradio e estático; o Docker é pago. O SDK Gradio roda
`python space_app.py` e espera um servidor na porta 7860: este arquivo só inicia nela o app
Streamlit de sempre (`app.py`). Fora do Space, use `streamlit run app.py`.
"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("AHU_ONNX_CACHE", os.path.expanduser("~/.cache/ahu-onnx"))
    os.execvp(sys.executable, [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true",
        "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false",
    ])
