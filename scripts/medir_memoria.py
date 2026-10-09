"""medir_memoria.py — WP 2.5: a memória e o tempo de carga que o A10 cobra.

O Streamlit Community Cloud gratuito não tem medidor: a única forma de saber se o app cabe é
medir a mesma carga aqui e comparar com o teto publicado. Este script mede, num processo novo e
em etapas, o RSS (memória residente) e o tempo de cada parte da carga fria:

  1. o interpretador com `streamlit`, `pandas` e `numpy` importados — o piso do processo;
  2. `ahu.dados.carregar()` — os artefatos de `app_dados/`;
  2b. com `--com-app`, o grafo de módulos que o `app.py` carrega (plotly, fpdf2, F3);
  3. a primeira consulta **léxica** (sem o modelo) — é o que abrir o app e buscar uma cota paga;
  4. `torch` e `sentence_transformers` importados — o custo da biblioteca, sem peso de modelo;
  5. o modelo de embeddings carregado e a primeira consulta semântica — o pico.

Rodar (com o modelo do manifesto):
    python scripts/medir_memoria.py

Com outro modelo, para decidir a troca do WP 2.5/4.2 sem reindexar nada:
    python scripts/medir_memoria.py --modelo intfloat/multilingual-e5-small --so-modelo

`--so-modelo` mede só as etapas 4 e 5: útil para comparar tamanhos de modelo depressa, já que a
etapa 5 não precisa que os vetores de `app_dados/` sejam daquele modelo (a dimensão só importa
para buscar de verdade, e aí o `ahu/dados.py` recusa iniciar — A4).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

MB = 1024 * 1024


def rss_mb() -> float:
    """RSS do processo, em MiB. Sem dependência nova: Windows pela API, Linux pelo /proc."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class Contagem(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wt.HANDLE
        # `K32GetProcessMemoryInfo` está no kernel32 desde o Windows 7; o `psapi.dll` é só o
        # apelido antigo. Sem `argtypes`, o handle de 64 bits chega truncado e a chamada falha
        # devolvendo zero — era por isso que a medição saía 0,0 MiB.
        funcao = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if funcao is None:
            funcao = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        funcao.argtypes = [wt.HANDLE, ctypes.POINTER(Contagem), wt.DWORD]
        funcao.restype = wt.BOOL

        contagem = Contagem()
        contagem.cb = ctypes.sizeof(Contagem)
        if not funcao(kernel32.GetCurrentProcess(), ctypes.byref(contagem), contagem.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo falhou")
        return contagem.WorkingSetSize / MB
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MB
    except OSError:
        import resource
        bruto = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return bruto / MB if sys.platform == "darwin" else bruto / 1024


class Medidor:
    def __init__(self) -> None:
        self.etapas: list[dict] = []
        self.inicio = time.perf_counter()

    def marcar(self, nome: str, desde: float) -> None:
        agora = time.perf_counter()
        self.etapas.append({
            "etapa": nome,
            "rss_mb": round(rss_mb(), 1),
            "segundos": round(agora - desde, 2),
            "acumulado_s": round(agora - self.inicio, 2),
        })
        print(f"  {nome:38} RSS {self.etapas[-1]['rss_mb']:8.1f} MiB   "
              f"+{self.etapas[-1]['segundos']:6.2f} s   "
              f"(total {self.etapas[-1]['acumulado_s']:6.2f} s)", flush=True)


def medir(modelo: str | None, so_modelo: bool, consulta: str,
          com_app: bool = False) -> dict:
    m = Medidor()
    t = time.perf_counter()
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import streamlit  # noqa: F401
    m.marcar("1. python + streamlit + pandas", t)

    dados = None
    if not so_modelo:
        t = time.perf_counter()
        from ahu.dados import carregar_sem_cache
        dados = carregar_sem_cache()
        m.marcar("2. artefatos de app_dados/", t)

        if com_app:
            # O grafo de módulos que o app de verdade carrega: plotly (painel), fpdf2 (PDFs) e
            # os módulos da F3. Sem esta etapa a medição ignora o que a F3 acrescentou.
            t = time.perf_counter()
            import app  # noqa: F401
            m.marcar("2b. módulos do app (plotly, fpdf2, F3)", t)

        t = time.perf_counter()
        from ahu.busca import buscar
        buscar(dados, "AHU_ACL_CU_017, Cx. 5, D. 487", modo="exata")
        m.marcar("3. primeira busca lexica (sem modelo)", t)
        modelo = modelo or dados.modelo_embeddings.get("nome")

    t = time.perf_counter()
    import sentence_transformers  # noqa: F401
    import torch  # noqa: F401
    m.marcar("4. torch + sentence_transformers", t)

    t = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    codificador = SentenceTransformer(modelo, device="cpu")
    m.marcar(f"5. modelo {modelo}", t)

    t = time.perf_counter()
    codificador.encode([f"query: {consulta}"], normalize_embeddings=True,
                       show_progress_bar=False)
    m.marcar("6. primeira consulta codificada", t)

    pico = max(e["rss_mb"] for e in m.etapas)
    saida = {
        "modelo": modelo,
        "plataforma": sys.platform,
        "python": sys.version.split()[0],
        "etapas": m.etapas,
        "pico_rss_mb": pico,
        "carga_fria_s": m.etapas[-1]["acumulado_s"],
    }
    if dados is not None:
        saida["documentos"] = int(len(dados.documentos))
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description="Mede RSS e tempo da carga fria do app (A10).")
    ap.add_argument("--modelo", default=None,
                    help="Modelo de embeddings a medir. Sem isto, o do manifesto.")
    ap.add_argument("--so-modelo", action="store_true",
                    help="Mede só torch e o modelo, sem carregar app_dados/.")
    ap.add_argument("--consulta", default="deserção de soldados")
    ap.add_argument("--com-app", action="store_true",
                    help="acrescenta uma etapa importando o grafo de módulos do app.py "
                         "(plotly, fpdf2 e os módulos da F3)")
    ap.add_argument("--saida", default=None, help="Grava o resultado em JSON.")
    args = ap.parse_args()

    print(f"Medindo a carga fria ({sys.platform}, python {sys.version.split()[0]}):")
    resultado = medir(args.modelo, args.so_modelo, args.consulta, args.com_app)
    print(f"\nPico de RSS: {resultado['pico_rss_mb']:.1f} MiB · "
          f"carga fria: {resultado['carga_fria_s']:.1f} s")
    if args.saida:
        Path(args.saida).write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        print(f"Gravado em {args.saida}")


if __name__ == "__main__":
    main()
