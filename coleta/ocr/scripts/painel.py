"""Painel de acompanhamento: redesenha a cada 10 s, sem piscar. Fechar a janela nao para nada.

Uso: clique duas vezes em painel.bat (ou, nesta pasta, python painel.py).
"""
import os, re, subprocess, sys, time
from datetime import datetime
import ocr_comum as c

VERMELHO, VERDE, AMARELO, CINZA, NEGRITO, FIM = "\033[91m", "\033[92m", "\033[93m", "\033[90m", "\033[1m", "\033[0m"
INTERVALO = 10
SAIDAS = {"ocr_api": "Batch", "ocr": "agy", "ocr_corrigido": "Tess+corr", "ocr_lite": "estudante"}
NOMES = {"86_bahia_v1": "Bahia 1", "87_bahia_v2": "Bahia 2", "75_sao_paulo_v1": "Sao Paulo 1",
         "76_sao_paulo_v2": "Sao Paulo 2", "91_minas_v1": "Minas 1", "92_minas_v2": "Minas 2", "93_minas_v3": "Minas 3",
         "79_para_v1": "Para 1", "80_para_v2": "Para 2", "81_para_v3": "Para 3", "pe_v1": "Pernambuco 1",
         "pe_v2": "Pernambuco 2", "pe_v3": "Pernambuco 3", "pe_v4": "Pernambuco 4", "maranhao": "Maranhao",
         "colonia": "Colonia Sacram.", "88_ultramar_brasil_geral": "Ultramar/Br.Geral",
         "84_codices_cu": "Codices CU", "89_cartografia": "Cartografia"}
# expected noise in the logs, never shown as an error
NORMAL = ("503", "high demand", "COTA agy", "RECITATION", "resposta vazia", "sem texto", "Estimating resolution")
_cache = {"baixando": (0, False)}


def pid_vivo(arq):
    p = c.KIT / arq
    if not p.exists():
        return None
    pid = p.read_text().strip()
    if not pid.isdigit():
        return None  # stopped on purpose ("parado")
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
    return pid in r.stdout


def baixando():
    """The download has no pid file; asking Windows is slow, so once a minute."""
    t, v = _cache["baixando"]
    if time.time() - t > 60:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                            "? CommandLine -match 'baixar_paginas' | Measure | % Count"], capture_output=True, text=True)
        v = r.stdout.strip() not in ("", "0")
        _cache["baixando"] = (time.time(), v)
    return v


def cauda(arquivo, n=3):
    p = c.KIT / arquivo
    if not p.exists():
        return []
    bruto = p.read_bytes()
    texto = bruto[-20000:].decode("utf-16-le" if bruto[:2] == b"\xff\xfe" else "utf-8", errors="replace")
    linhas = [l.strip("﻿\x00").rstrip() for l in texto.splitlines()]
    return [l for l in linhas if l.strip()][-n:]


def levantamento():
    """One listing per output folder: which pages are done, by which route, and when."""
    feitas, por_rota, recentes = set(), {}, {r: [] for r in SAIDAS.values()}
    agora = time.time()
    for pasta, rota in SAIDAS.items():
        d = c.KIT / pasta
        if not d.is_dir():
            continue
        for vol_dir in d.iterdir():
            if not vol_dir.is_dir():
                continue
            for e in os.scandir(vol_dir):
                if e.name[:4].isdigit() and e.name.endswith(".txt"):
                    pag = int(e.name[:4])
                    feitas.add((vol_dir.name, pag))
                    por_rota[(vol_dir.name, rota)] = por_rota.get((vol_dir.name, rota), 0) + 1
                    m = e.stat().st_mtime
                    if agora - m < 3600:
                        recentes[rota].append(m)
    return feitas, por_rota, recentes


def barra(feito, total, largura=20, cor=VERDE):
    n = min(largura, round(largura * feito / total)) if total else 0
    return cor + "■" * n + CINZA + "·" * (largura - n) + FIM


def ponto(estado):
    return {"ok": VERDE + "●" + FIM, "pausa": AMARELO + "●" + FIM, "erro": VERMELHO + "●" + FIM, "fim": CINZA + "●" + FIM}[estado]


def ha(ts):
    if not ts:
        return "nada ainda"
    m = (time.time() - ts) / 60
    return "agora" if m < 1 else f"ha {m:.0f} min"


def pausa_agy():
    for arq in ("ocr_fila.log", "corrigir.log"):
        cota = [l for l in cauda(arq, 60) if "COTA agy esgotada" in l and "pausa ate" in l]
        if cota:
            ate = cota[-1].split("pausa ate ")[1][:5]
            if datetime.now().strftime("%H:%M") < ate:
                return ate
    return None


def imagens(vols):
    """Downloaded pages per volume, by name only (no stat per file)."""
    r = {}
    for v in vols:
        d = c.PAGINAS / v
        r[v] = sum(1 for e in os.scandir(d) if e.name[:4].isdigit() and e.name[-4:] in (".jpg", ".png", "webp")) if d.is_dir() else 0
    return r


def custo():
    """The Batch horizon checks thousands of files: once a minute is enough."""
    t, v = _cache.get("custo", (0, None))
    if time.time() - t > 60 or v is None:
        v = c.horizonte()
        _cache["custo"] = (time.time(), v)
    return v


def tela():
    feitas, por_rota, recentes = levantamento()
    pend = {}
    for v, p in c.bloqueadas():
        if (v, p) not in feitas:
            pend[v] = pend.get(v, 0) + 1
    vols = list(c.DOCVIRT) + c.BN
    baix = imagens(vols)
    totais = {v: c.DOCVIRT.get(v) or baix[v] for v in vols}
    total = sum(totais.values())
    prontas = sum(1 for v, p in feitas if v in totais and p <= totais[v])
    hora = sum(len(x) for x in recentes.values())
    eta = f"~{(total - prontas) / hora:.1f} h no ritmo da ultima hora ({hora} pag/h)" if hora else "ritmo ainda sem dados"

    L = [f"{NEGRITO}RESGATE OCR{FIM}   {datetime.now():%d/%m %H:%M:%S}   atualiza a cada {INTERVALO} s · fechar esta janela nao para nada",
         "",
         f"  Paginas transcritas  {barra(prontas, total, 40)}  {NEGRITO}{prontas:,} / {total:,}{FIM} ({prontas / total:.0%})".replace(",", "."),
         f"  Falta: {total - prontas:,} paginas  ·  {eta}".replace(",", "."), "",
         f"  {NEGRITO}{'VOLUME':16} {'PROGRESSO':20}  {'TRANSCRITAS':>13}  {'BAIXADAS':>9}  {'RECUSADAS':>9}   FEITAS POR{FIM}",
         f"  {'':16} {'':20}  {'':>13}  {'':>9}  {'na fila':>9}   Batch / agy / Tess+corr / estudante"]
    for v in vols:
        tot = totais[v]
        n = sum(1 for p in range(1, tot + 1) if (v, p) in feitas)
        cor = VERDE if n == tot else (CINZA if n == 0 else AMARELO)
        rotas = " / ".join(str(por_rota.get((v, r), 0)) for r in SAIDAS.values())
        L.append(f"  {cor}{NOMES.get(v, v):16}{FIM} {barra(n, tot, 20, cor)}  {n:>5}/{tot:<4}{n / tot:>4.0%}  "
                 f"{baix[v]:>5}/{tot:<4} {pend.get(v, 0) or '':>9}   {CINZA}{rotas}{FIM}")

    # workers
    try:
        _, previstas, gasto, pendente, por_pagina = custo()
        dinheiro = (f"US$ {gasto + pendente:.2f} gastos (sem teto, recarga automatica)" if c.ORCAMENTO >= 1e5 else
                    f"US$ {gasto + pendente:.2f} de {c.ORCAMENTO:.2f} {barra(gasto + pendente, c.ORCAMENTO, 12, AMARELO)}") + f" · ainda ~{len(previstas)} pag"
    except Exception as e:
        dinheiro, previstas = f"(custo ilegivel: {e})", []
    ult = {r: max(x) if x else None for r, x in recentes.items()}
    completo = all(baix[v] >= t for v, t in c.DOCVIRT.items())
    est = []
    vivo = baixando()
    est.append((("ok" if vivo else ("fim" if completo else "erro")), "Download docvirt",
                "rodando" if vivo else ("terminou" if completo else "PAROU - avise o Claude"),
                f"{sum(baix[v] for v in c.DOCVIRT):,}/{sum(c.DOCVIRT.values()):,} pag".replace(",", ".")))
    b = pid_vivo("ocr_batch.pid")
    blog = cauda("ocr_batch.log", 1)
    bfase = "rodando"
    if blog:
        hora = blog[0][11:16]
        if "PAUSA" in blog[0]:
            bfase = "PAUSADO (catalogos em texto encontrados)"
        elif "preparando lote" in blog[0] or "prontas" in blog[0] and not blog[0].endswith("prontas 0"):
            bfase = f"montando lote desde {hora} (Tesseract das recusadas)"
        elif "enviado" in blog[0]:
            bfase = f"lote no Google desde {hora} (leva 10-30 min)"
    est.append(("ok" if b else "erro", "Batch (chave paga)", bfase if b else "PAROU - avise o Claude",
                f"{len(recentes['Batch'])} pag/h · ultima {ha(ult['Batch'])} · {dinheiro}"))
    a, pausa = pid_vivo("ocr_fila.pid"), pausa_agy()
    corr = pid_vivo("corrigir.pid")
    est.append(("pausa" if (a and pausa) else ("ok" if a else "erro"), "agy (AI Pro)",
                f"em pausa ate {pausa} (cota; volta sozinho)" if pausa else ("rodando" if a else "PAROU - avise o Claude"),
                f"{len(recentes['agy']) + len(recentes['Tess+corr'])} pag/h (direto + correcao) · ultima {ha(max(filter(None, [ult['agy'], ult['Tess+corr']]), default=None))}"))
    est.append(("fim" if corr is None else "pausa" if (corr and pausa) else ("ok" if corr else "erro"), "Correcao recusadas",
                "parada (volumes cobertos por catalogos em texto)" if corr is None else "rodando" if corr else "PAROU - avise o Claude",
                f"{sum(pend.values())} paginas na fila"))
    s = pid_vivo("ocr_estudante.pid")
    slog = cauda("ocr_estudante.log", 3)
    ocupado = bool(slog) and "503" in slog[-1]
    cota_dia = bool(slog) and "429" in slog[-1]
    elog = c.KIT / "ocr_estudante.log"
    # its refused pages are corrected into ocr_corrigido, so the log's time is its real last activity
    ativo = elog.stat().st_mtime if elog.exists() else None
    est.append(("pausa" if (s and (ocupado or cota_dia)) else ("ok" if s else "erro"), "Chave estudante",
                ("cota diaria do nivel gratuito esgotada" if cota_dia else
                 "Google ocupado (503), espera 5 min" if ocupado else "rodando") if s else "PAROU - avise o Claude",
                f"{len(recentes['estudante'])} pag/h direto · ultima atividade {ha(ativo)}"))
    L += ["", f"  {NEGRITO}TRABALHADORES{FIM}"]
    for e, nome, txt, extra in est:
        L.append(f"  {ponto(e)} {nome:20} {txt:42} {CINZA}{extra}{FIM}")

    # only real problems
    erros = []
    for arq in ("ocr_batch.err", "ocr_fila.err", "corrigir.err", "ocr_estudante.err"):
        erros += [f"{arq}: {l.strip()[:110]}" for l in cauda(arq, 2) if not any(k in l for k in NORMAL)]
    for arq in ("ocr_batch.log", "ocr_fila.log", "corrigir.log"):
        erros += [f"{arq}: {l.strip()[:110]}" for l in cauda(arq, 20)
                  if re.search(r"Traceback|erro ao|ERRO tesseract|FAILED|EXPIRED", l) and not any(k in l for k in NORMAL)]
    L += ["", f"  {NEGRITO}PROBLEMAS{FIM}"]
    L += [VERMELHO + "  " + x + FIM for x in erros[-3:]] or [CINZA + "  nenhum (503 e pausas de cota sao normais)" + FIM]
    sys.stdout.write("\033[H" + "\n".join(l + "\033[K" for l in L) + "\033[J")
    sys.stdout.flush()


if __name__ == "__main__":
    os.system("")  # turns on ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.system("cls")
    while True:
        t0 = time.time()
        try:
            tela()
        except Exception as e:
            print(VERMELHO + f"erro no painel: {e}" + FIM)
        time.sleep(max(1, INTERVALO - (time.time() - t0)))
