"""Varredura do DocReader do Projeto Resgate: pagina (pagfis) onde comeca cada documento.

Arquitetura com pool global de 8 trabalhadores, HTTP/1.1 Keep-Alive por thread,
cooldown global em caso de 403/rate-limit, gravacao atomica e telemetria em estado.json.

Saida nesta pasta:
- resgate_pagfis_<bib>.tsv (bib, caixa, doc, pagfis) quando a colecao termina;
- parcial_<bib>_w<n>.tsv e .ok enquanto roda;
- estado.json com o estado em tempo real do pool global para o painel.
"""
import http.client
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

PASTA = os.path.dirname(os.path.abspath(__file__))
FILA = [
    "017_RJ_AV", "023_SP_AV", "012_CSRP", "003_BG_AV", "011_MG",
    "015_PE", "005_BA_AV", "005_BA_CA", "005_BA_LF", "009_MA",
    "013_PA", "050_CS_AV", "084_OU_AV", "030_SP_AV", "089_UL_AV",
    "017-1_RJ", "023-1_SP_MG", "089_CU",
]
TRABALHADORES = 8
PARTES = 8
HOST = "resgate.bn.gov.br"
PATH_URL = "/docreader/DocReader.aspx?bib={bib}&hf=resgate.bn.gov.br&pagfis={p}"
FULL_URL = "https://" + HOST + PATH_URL
CAB = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Connection": "keep-alive",
}
RE_PAG = re.compile(r'id="hPagFis"[^>]*value="(\d+)"')
RE_LAB = re.compile(r'Cx\.\s*(\d+)\\+Doc\.\s*([^<"]*)')

_parar = threading.Event()
_log_trava = threading.Lock()
_agendador_trava = threading.Lock()
_prep_trava = threading.Lock()
_cooldown_trava = threading.Lock()
_tls = threading.local()

_inicio_sessao = time.time()
_cooldown_until = 0.0
_cont = {"req": 0, "docs": 0}
_slots = [
    {
        "tid": i,
        "bib": None,
        "shard": None,
        "start_pag": None,
        "end_pag": None,
        "cur_pag": None,
        "docs_sessao": 0,
        "state": "idle",
        "last_error": "",
    }
    for i in range(TRABALHADORES)
]
_colecoes = {}  # bib -> {"mx": int, "pending": list[int], "running": set[int], "done": set[int], "finalized": bool, "docs_sessao": int}
_fila_pendente = []
_erros_recentes = []


class _PararExecucao(Exception):
    pass


def log(msg):
    with _log_trava:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def registrar_erro(msg):
    with _log_trava:
        _erros_recentes.append(f"{time.strftime('%H:%M:%S')} {msg}")
        if len(_erros_recentes) > 10:
            del _erros_recentes[:-10]


def _get_conn():
    conn = getattr(_tls, "conn", None)
    if conn is None:
        conn = http.client.HTTPSConnection(HOST, timeout=45)
        _tls.conn = conn
    return conn


def _reset_conn():
    conn = getattr(_tls, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _tls.conn = None


def _esperar_cooldown():
    with _cooldown_trava:
        rem = _cooldown_until - time.time()
    if rem > 0:
        if _parar.wait(rem):
            raise _PararExecucao()


def _definir_cooldown(segundos):
    with _cooldown_trava:
        global _cooldown_until
        alvo = time.time() + segundos
        if alvo > _cooldown_until:
            _cooldown_until = alvo


def _http_get(bib, p):
    path = PATH_URL.format(bib=bib, p=p)
    conn = _get_conn()
    conn.request("GET", path, headers=CAB)
    resp = conn.getresponse()
    body = resp.read()
    status = resp.status
    if status in (301, 302, 303, 307, 308):
        loc = resp.getheader("Location")
        if loc:
            req = urllib.request.Request(
                urllib.parse.urljoin(FULL_URL.format(bib=bib, p=p), loc), headers=CAB
            )
            with urllib.request.urlopen(req, timeout=45) as r2:
                return r2.status, r2.read().decode("utf-8", "ignore")
    return status, body.decode("utf-8", "ignore")


def rotulo(bib, p, tid=None, shard=None):
    """(ok, chave, caixa, doc) da pagina p. Reutiliza Keep-Alive e aplica backoff global em bloqueios."""
    espera = 5
    tag = f"[{bib} w{shard}]" if shard is not None else f"[{bib}]"
    while True:
        if _parar.is_set():
            raise _PararExecucao()
        _esperar_cooldown()
        try:
            status, h = _http_get(bib, p)
            if status != 200:
                _reset_conn()
                if status in (403, 429, 503):
                    msg = f"{tag} HTTP {status} da BN ({'VPN desligada?' if status == 403 else 'sobrecarga'}); aguardo {espera}s."
                    log(msg)
                    registrar_erro(msg)
                    _definir_cooldown(espera)
                else:
                    msg = f"{tag} HTTP {status}; tento de novo em {espera}s."
                    log(msg)
                    registrar_erro(msg)
                if tid is not None:
                    with _agendador_trava:
                        _slots[tid]["state"] = "backoff"
                        _slots[tid]["last_error"] = f"HTTP {status}"
                if _parar.wait(espera):
                    raise _PararExecucao()
                espera = min(espera * 2, 120)
                continue

            with _agendador_trava:
                _cont["req"] += 1
                if tid is not None:
                    _slots[tid]["state"] = "running"
                    _slots[tid]["last_error"] = ""

            mp, ml = RE_PAG.search(h), RE_LAB.search(h)
            ok = bool(mp and ml and int(mp.group(1)) == p)
            if not ml:
                return False, "", "", ""
            cx, doc = ml.group(1), ml.group(2).strip()
            return ok, cx + "|" + doc, cx, doc
        except _PararExecucao:
            raise
        except Exception as e:
            _reset_conn()
            msg = f"{tag} erro de rede ({type(e).__name__}); tento de novo em {espera}s."
            log(msg)
            registrar_erro(msg)
            if tid is not None:
                with _agendador_trava:
                    _slots[tid]["state"] = "backoff"
                    _slots[tid]["last_error"] = type(e).__name__
            if _parar.wait(espera):
                raise _PararExecucao()
            espera = min(espera * 2, 120)


def max_pag(bib, tid=None):
    arq = os.path.join(PASTA, f"max_{bib}.txt")
    if os.path.exists(arq):
        return int(open(arq, encoding="utf-8").read().strip())
    lo, hi = 1, 1024
    while rotulo(bib, hi, tid=tid)[0]:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        m = (lo + hi) // 2
        if rotulo(bib, m, tid=tid)[0]:
            lo = m
        else:
            hi = m
    tmp = arq + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(lo))
    os.replace(tmp, arq)
    return lo


def proximo(bib, P, chave, mx, passo, tid=None, shard=None):
    """Primeira pagina depois de P com rotulo diferente: (pagina, chave, caixa, doc) ou None no fim."""
    lo, s, hi, achado = P, passo, None, None
    while True:
        q = lo + s
        if q > mx:
            hi = mx + 1
            break
        r = rotulo(bib, q, tid=tid, shard=shard)
        if r[1] == chave:
            lo, s = q, s * 2
        else:
            hi, achado = q, r
            break
    while hi - lo > 1:
        m = (lo + hi) // 2
        r = rotulo(bib, m, tid=tid, shard=shard)
        if r[1] == chave:
            lo = m
        else:
            hi, achado = m, r
    if hi > mx or not achado or not achado[0]:
        return None
    return hi, achado[1], achado[2], achado[3]


def trabalhador(tid, bib, i, a, b, mx):
    arq = os.path.join(PASTA, f"parcial_{bib}_w{i}.tsv")
    feito = os.path.join(PASTA, f"parcial_{bib}_w{i}.ok")
    if os.path.exists(feito):
        return True

    linhas = [l.rstrip("\n").split("\t") for l in open(arq, encoding="utf-8") if l.strip()] if os.path.exists(arq) else []
    f = open(arq, "a", encoding="utf-8")
    try:
        passo = 3
        if linhas:
            cx, doc, p = linhas[-1]
            P, chave = int(p), cx + "|" + doc
            with _agendador_trava:
                _slots[tid]["cur_pag"] = P
            r = proximo(bib, P, chave, mx, passo, tid=tid, shard=i)
            if not r:
                open(feito, "w").close()
                return True
            P, chave, cx, doc = r
        else:
            ok, chave, cx, doc = rotulo(bib, a, tid=tid, shard=i)
            P = a
            if a > 1 and rotulo(bib, a - 1, tid=tid, shard=i)[1] == chave:
                r = proximo(bib, a, chave, mx, passo, tid=tid, shard=i)
                if not r:
                    open(feito, "w").close()
                    return True
                P, chave, cx, doc = r

        while P <= b:
            if _parar.is_set():
                return False
            f.write(f"{cx}\t{doc}\t{P}\n")
            f.flush()
            with _agendador_trava:
                _cont["docs"] += 1
                _slots[tid]["cur_pag"] = P
                _slots[tid]["docs_sessao"] += 1
                if bib in _colecoes:
                    _colecoes[bib]["docs_sessao"] += 1
            r = proximo(bib, P, chave, mx, passo, tid=tid, shard=i)
            if not r:
                break
            passo = max(2, min(8, round((r[0] - P) * 0.8)))
            P, chave, cx, doc = r

        open(feito, "w").close()
        return True
    except _PararExecucao:
        return False
    finally:
        f.close()


def finalizar_colecao(bib):
    final = os.path.join(PASTA, f"resgate_pagfis_{bib}.tsv")
    if os.path.exists(final):
        return
    linhas = []
    for i in range(PARTES):
        arq = os.path.join(PASTA, f"parcial_{bib}_w{i}.tsv")
        if os.path.exists(arq):
            with open(arq, encoding="utf-8") as fh:
                linhas.extend(l.rstrip("\n").split("\t") for l in fh if l.strip())
    vistos, saida = set(), []
    for cx, doc, p in sorted(linhas, key=lambda x: int(x[2])):
        if p not in vistos:
            vistos.add(p)
            saida.append(f"{bib}\t{cx}\t{doc}\t{p}")
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("bib\tcaixa\tdoc\tpagfis\n" + "\n".join(saida) + "\n")
    os.replace(tmp, final)
    for i in range(PARTES):
        for ext in ("tsv", "ok"):
            try:
                os.remove(os.path.join(PASTA, f"parcial_{bib}_w{i}.{ext}"))
            except FileNotFoundError:
                pass
    log(f"[{bib}] PRONTO, {len(saida)} documentos -> {os.path.basename(final)}")


def _preparar_proxima_colecao(tid):
    """Inicializa a proxima colecao da fila pendente (fora de _agendador_trava durante max_pag)."""
    with _prep_trava:
        while True:
            if _parar.is_set():
                return None
            with _agendador_trava:
                # Ja existe tarefa pendente preparada por outra thread?
                for b, st in _colecoes.items():
                    if st["pending"]:
                        return b
                if not _fila_pendente:
                    return None
                bib = _fila_pendente.pop(0)
                _slots[tid]["bib"] = bib
                _slots[tid]["shard"] = "init"
                _slots[tid]["state"] = "init"

            final = os.path.join(PASTA, f"resgate_pagfis_{bib}.tsv")
            if os.path.exists(final):
                log(f"[{bib}] ja feito, pulo.")
                continue

            mx = max_pag(bib, tid=tid)
            tam = -(-mx // PARTES)
            done = set()
            pending = []
            for i in range(PARTES):
                a = 1 + i * tam
                if a > mx:
                    done.add(i)
                elif os.path.exists(os.path.join(PASTA, f"parcial_{bib}_w{i}.ok")):
                    done.add(i)
                else:
                    pending.append(i)

            if not pending and len(done) == PARTES:
                finalizar_colecao(bib)
                continue

            log(f"[{bib}] {mx} paginas; iniciando ({len(pending)} fatias pendentes).")
            with _agendador_trava:
                _colecoes[bib] = {
                    "mx": mx,
                    "tam": tam,
                    "pending": pending,
                    "running": set(),
                    "done": done,
                    "finalized": False,
                    "docs_sessao": 0,
                }
            return bib


def pegar_tarefa(tid):
    while not _parar.is_set():
        with _agendador_trava:
            for bib, st in _colecoes.items():
                if st["pending"]:
                    shard = st["pending"].pop(0)
                    st["running"].add(shard)
                    mx = st["mx"]
                    tam = st["tam"]
                    a = 1 + shard * tam
                    b = min(mx, (shard + 1) * tam)
                    _slots[tid].update({
                        "bib": bib,
                        "shard": shard,
                        "start_pag": a,
                        "end_pag": b,
                        "cur_pag": None,
                        "state": "running",
                        "last_error": "",
                    })
                    return bib, shard, a, b, mx
        if _preparar_proxima_colecao(tid) is None:
            return None
    return None


def concluir_tarefa(tid, bib, shard, concluido):
    deve_finalizar = False
    with _agendador_trava:
        st = _colecoes.get(bib)
        if st:
            st["running"].discard(shard)
            if concluido:
                st["done"].add(shard)
                if len(st["done"]) == PARTES and not st["finalized"]:
                    st["finalized"] = True
                    deve_finalizar = True
            else:
                st["pending"].insert(0, shard)
        _slots[tid].update({
            "bib": None,
            "shard": None,
            "start_pag": None,
            "end_pag": None,
            "cur_pag": None,
            "state": "stopping" if _parar.is_set() else "idle",
        })
    if deve_finalizar:
        finalizar_colecao(bib)
        with _agendador_trava:
            _colecoes.pop(bib, None)


def worker_loop(tid):
    try:
        while not _parar.is_set():
            tarefa = pegar_tarefa(tid)
            if tarefa is None:
                break
            bib, shard, a, b, mx = tarefa
            ok = trabalhador(tid, bib, shard, a, b, mx)
            concluir_tarefa(tid, bib, shard, ok)
    except _PararExecucao:
        pass
    finally:
        _reset_conn()
        with _agendador_trava:
            _slots[tid]["state"] = "stopped"
            _slots[tid]["bib"] = None
            _slots[tid]["shard"] = None


def gravar_estado():
    agora = time.time()
    elapsed = max(0.01, (agora - _inicio_sessao) / 60.0)
    with _agendador_trava:
        docs_s = _cont["docs"]
        req_s = _cont["req"]
        slots = [dict(s) for s in _slots]
        ativas = {
            bib: {
                "max_pages": st["mx"],
                "docs_sessao": st["docs_sessao"],
                "pending": list(st["pending"]),
                "running": sorted(st["running"]),
                "done": sorted(st["done"]),
            }
            for bib, st in _colecoes.items()
        }
    with _log_trava:
        erros = list(_erros_recentes)

    payload = {
        "updated_at": agora,
        "started_at": _inicio_sessao,
        "pid": os.getpid(),
        "docs_sessao": docs_s,
        "req_sessao": req_s,
        "docs_per_min": round(docs_s / elapsed, 1),
        "req_per_min": round(req_s / elapsed, 1),
        "slots": slots,
        "active_collections": ativas,
        "recent_errors": erros,
    }
    alvo = os.path.join(PASTA, "estado.json")
    tmp = alvo + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, alvo)
    except Exception:
        pass


def reporter_loop():
    ultimo_log = time.time()
    while not _parar.wait(5):
        gravar_estado()
        agora = time.time()
        if agora - ultimo_log >= 60:
            ultimo_log = agora
            with _agendador_trava:
                ativas = list(_colecoes.keys())
                docs_s = _cont["docs"]
                req_s = _cont["req"]
            elapsed = max(0.01, (agora - _inicio_sessao) / 60.0)
            nomes = "+".join(ativas) if ativas else "fila"
            log(
                f"[{nomes}] {docs_s} documentos, {req_s} paginas lidas nesta sessao "
                f"({docs_s / elapsed:.1f} docs/min)"
            )
    gravar_estado()


def _on_signal(signum, frame):
    if not _parar.is_set():
        log(f"Sinal {signum} recebido; encerrando trabalhadores com seguranca...")
        _parar.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        status, h = _http_get("021_SC", 37)
        if status != 200:
            chave = f"HTTP {status}"
        else:
            ml = RE_LAB.search(h)
            chave = (ml.group(1) + "|" + ml.group(2).strip()) if ml else ""
    except urllib.error.HTTPError as e:
        chave = f"HTTP {e.code}"
    except Exception as e:
        chave = type(e).__name__
    finally:
        _reset_conn()

    if chave != "1|4":
        log(
            f'TESTE FALHOU ({chave or "pagina sem rotulo"}). Se a VPN esta ligada, a BN esta barrando '
            "o Python."
        )
        if sys.stdin.isatty():
            input("Enter para fechar.")
        sys.exit(1)

    log("Teste OK (SC Cx. 1 Doc. 4 = pagina 37). Iniciando pool global de 8 trabalhadores.")
    _fila_pendente = list(sys.argv[1:] or FILA)

    rep = threading.Thread(target=reporter_loop, daemon=True)
    rep.start()

    threads = [threading.Thread(target=worker_loop, args=(tid,), daemon=False) for tid in range(TRABALHADORES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _parar.set()
    rep.join(timeout=2)
    gravar_estado()
    if not any(_colecoes.values()):
        log("Fila inteira concluida.")
