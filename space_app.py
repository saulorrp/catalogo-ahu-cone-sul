"""space_app.py — entrada do Space do Hugging Face (SDK Gradio, plano gratuito).

O Space gratuito só aceita os SDKs Gradio e estático (o Docker é pago), e no plano grátis o
Gradio roda em ZeroGPU. Este arquivo cumpre o mínimo que o ZeroGPU exige (uma função com
`@spaces.GPU` na partida) e inicia na porta 7860 o app Streamlit de sempre (`app.py`). O app
não usa GPU. Fora do Space, use `streamlit run app.py`.
"""
import os
import subprocess
import sys

try:
    import spaces
except ImportError:  # fora do Space
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _sem_uso() -> None:
        """Só para o ZeroGPU reconhecer o Space; nunca é chamada."""


if __name__ == "__main__":
    os.environ.setdefault("AHU_ONNX_CACHE", os.path.expanduser("~/.cache/ahu-onnx"))
    processo = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true",
        "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false",
    ])
    sys.exit(processo.wait())
