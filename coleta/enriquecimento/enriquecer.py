"""Roda o enriquecimento com o Gemini sobre a fila de entrada/lotes.jsonl.gz, sem pressa.

Duas rotas, as duas sem custo:
  agy  o Gemini da assinatura AI Pro, pela linha de comando `agy` (cota em janelas de 5 h);
  api  a API do Gemini com a chave da conta AI Plus, no nível gratuito (variável
       GEMINI_API_KEY_ESTUDANTE). A chave paga (GEMINI_API_KEY) é recusada de propósito.

Quando uma rota esgota a cota, os trabalhadores dela esperam a cota voltar e continuam. Cada lote
pronto vira saida/<lote>.json, com o modelo e a rota que o produziram; um lote já feito não roda
de novo, então o script pode ser parado e retomado a qualquer hora.

Uso (no Hermerico, dentro desta pasta):
  python enriquecer.py                      # 2 trabalhadores agy + 1 api
  python enriquecer.py --agy 1 --api 0      # só a assinatura Pro
  python enriquecer.py --so avaliacao       # só os lotes de avaliação
"""
import argparse, gzip, json, os, re, subprocess, sys, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).parent
ENTRADA = AQUI / "entrada" / "lotes.jsonl.gz"
SAIDA = AQUI / "saida"
FALHAS = AQUI / "saida" / "_falhas.jsonl"
PROMPT = (AQUI / "PROMPT.md").read_text(encoding="utf-8")
API = "https://generativelanguage.googleapis.com/v1beta"
MODELO_AGY = "gemini-3.8-flash-medium"
MODELO_API = "gemini-3.1-flash-lite"
CHAVE_PAGA = "GEMINI_API_KEY"
MENOR_LOTE = 5

_trava = threading.Lock()
_pausa = {"agy": 0.0, "api": 0.0}


def agora():
    return time.strftime("%d/%m %H:%M")


class CotaEsgotada(Exception):
    def __init__(self, segundos, texto):
        super().__init__(texto)
        self.segundos = segundos


def espera(rota, e):
    """Todos os trabalhadores da rota esperam a cota voltar (mais um minuto), uma vez só."""
    with _trava:
        if time.time() >= _pausa[rota]:
            _pausa[rota] = time.time() + e.segundos + 60
            print(f"{agora()} [{rota}] cota esgotada; pausa até "
                  f"{time.strftime('%d/%m %H:%M', time.localtime(_pausa[rota]))}: {e}", flush=True)
    time.sleep(max(0, _pausa[rota] - time.time()))


def texto_do_pedido(docs):
    linhas = [f"[{i}] {d['texto']}" for i, d in enumerate(docs, 1)]
    return PROMPT.rstrip() + "\n\n" + "\n".join(linhas) + "\n"


def chama_agy(pedido, modelo):
    proc = subprocess.run(["agy", "-p", pedido, "--model", modelo, "--mode", "plan",
                           "--output-format", "json", "--print-timeout", "900s"],
                          cwd=AQUI, capture_output=True)
    bruto = proc.stdout.decode("utf-8", errors="replace")
    erro = proc.stderr.decode("utf-8", errors="replace")
    try:
        dados = json.loads(bruto[bruto.index("{"):])
    except ValueError:
        dados = {"response": "", "error": (bruto[:200] + " " + erro[:200]).strip()}
    msg = f"{dados.get('error') or ''} {erro}"
    if "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
        m = re.search(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", msg)
        seg = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)) if m else 1800
        raise CotaEsgotada(max(seg, 60), msg.strip()[:200])
    if dados.get("error"):
        raise RuntimeError(str(dados["error"])[:300])
    return dados.get("response") or ""


def chama_api(pedido, modelo, chave):
    corpo = {"contents": [{"parts": [{"text": pedido}]}],
             "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"{API}/models/{modelo}:generateContent",
                                 data=json.dumps(corpo).encode(),
                                 headers={"x-goog-api-key": chave, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            dados = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            depois = e.headers.get("Retry-After")
            raise CotaEsgotada(int(depois) if depois and depois.isdigit() else 3600, "HTTP 429")
        if e.code in (500, 502, 503, 504):
            raise CotaEsgotada(300, f"HTTP {e.code}")
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
    partes = dados.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in partes)


def le_resposta(texto, docs):
    """O array JSON da resposta, conferido contra os verbetes pedidos. Levanta ValueError."""
    t = texto.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    itens = json.loads(t[t.index("["):t.rindex("]") + 1])
    if len(itens) != len(docs) or [i.get("n") for i in itens] != list(range(1, len(docs) + 1)):
        raise ValueError(f"esperava {len(docs)} itens numerados, vieram {len(itens)}")
    saida = []
    for d, i in zip(docs, itens):
        saida.append({"doc_uid": d["doc_uid"],
                      "rem": pessoa(i.get("rem")), "dest": pessoa(i.get("dest")),
                      "mencionados": lista(i.get("mencionados")), "lugares": lista(i.get("lugares")),
                      "assunto": lista(i.get("assunto"))[:3]})
    return saida


def pessoa(p):
    if not isinstance(p, dict) or not (p.get("nome") or p.get("cargo")):
        return None
    tipo = p.get("tipo") if p.get("tipo") in ("pessoa", "instituicao", "coletivo") else None
    return {"nome": str(p.get("nome") or "").strip(), "cargo": str(p.get("cargo") or "").strip(),
            "tipo": tipo}


def lista(v):
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def processa(docs, rota, args):
    """Os resultados de uma lista de verbetes. Se a resposta não fecha, tenta de novo e depois
    divide ao meio, até MENOR_LOTE; o que ainda falhar vai para _falhas.jsonl."""
    modelo = args.modelo_agy if rota == "agy" else args.modelo_api
    for tentativa in range(2):
        try:
            texto = (chama_agy(texto_do_pedido(docs), modelo) if rota == "agy"
                     else chama_api(texto_do_pedido(docs), modelo, args.chave_valor))
            return le_resposta(texto, docs)
        except CotaEsgotada:
            raise
        except (ValueError, RuntimeError, OSError) as e:
            erro = str(e)[:200]
    if len(docs) > MENOR_LOTE:
        meio = len(docs) // 2
        return processa(docs[:meio], rota, args) + processa(docs[meio:], rota, args)
    with _trava, open(FALHAS, "a", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"doc_uid": d["doc_uid"], "rota": rota, "erro": erro},
                               ensure_ascii=False) + "\n")
    return []


def grava(lote, rota, modelo, resultados):
    final = SAIDA / f"{lote}.json"
    tmp = final.with_suffix(".tmp")
    tmp.write_text(json.dumps({"lote": lote, "rota": rota, "modelo": modelo,
                               "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "docs": resultados}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agy", type=int, default=2, help="trabalhadores na assinatura Pro (agy)")
    ap.add_argument("--api", type=int, default=1, help="trabalhadores na chave Plus (API grátis)")
    ap.add_argument("--chave", default="GEMINI_API_KEY_ESTUDANTE", help="variável com a chave")
    ap.add_argument("--modelo-agy", default=MODELO_AGY)
    ap.add_argument("--modelo-api", default=MODELO_API)
    ap.add_argument("--so", default="", help="só os lotes cujo nome começa assim (ex.: avaliacao)")
    args = ap.parse_args()
    if args.api and args.chave == CHAVE_PAGA:
        sys.exit(f"{CHAVE_PAGA} é a chave paga; este script só usa a rota gratuita.")
    args.chave_valor = os.environ.get(args.chave, "")
    if args.api and not args.chave_valor:
        print(f"{args.chave} não definida: seguindo só com agy.", flush=True)
        args.api = 0

    SAIDA.mkdir(exist_ok=True)
    with gzip.open(ENTRADA, "rt", encoding="utf-8") as f:
        fila = [json.loads(l) for l in f]
    fila = [l for l in fila if l["lote"].startswith(args.so)
            and not (SAIDA / f"{l['lote']}.json").exists()]
    total = len(fila)
    print(f"{agora()} {total} lotes por fazer; agy {args.agy}, api {args.api}", flush=True)
    feitos = [0]

    def trabalhador(rota):
        modelo = args.modelo_agy if rota == "agy" else args.modelo_api
        while True:
            with _trava:
                if not fila:
                    return
                lote = fila.pop(0)
            while True:
                try:
                    resultados = processa(lote["docs"], rota, args)
                    break
                except CotaEsgotada as e:
                    espera(rota, e)
            grava(lote["lote"], rota, modelo, resultados)
            with _trava:
                feitos[0] += 1
                if feitos[0] % 10 == 0 or feitos[0] == total:
                    print(f"{agora()} {feitos[0]}/{total} lotes ({lote['lote']}, {rota})", flush=True)

    fios = [threading.Thread(target=trabalhador, args=(r,), daemon=True)
            for r, n in (("agy", args.agy), ("api", args.api)) for _ in range(n)]
    for t in fios:
        t.start()
    for t in fios:
        t.join()
    print(f"{agora()} fim: {feitos[0]} lotes nesta rodada.", flush=True)


if __name__ == "__main__":
    main()
