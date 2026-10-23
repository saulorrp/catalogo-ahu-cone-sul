"""Transcreve as paginas baixadas com o Gemini Flash, pagina por pagina.

Dois caminhos:
- agy (padrao): Google Antigravity CLI, com a conta Google do dono. Cada pagina e
  uma chamada `agy -p ... --mode plan` (o agy so le a imagem e devolve o texto;
  quem grava o .txt e este script). Sem custo por pagina, mas lento. Saida em ocr/.
- --api: a API Gemini direta (chave em GEMINI_API_KEY), preco cheio. Saida em ocr_api/.
O Batch API (metade do preco) e ocr_batch.py.

Uso (na pasta deste kit):
    python ocr_gemini.py 86_bahia_v1 --de 41 --ate 60                      # teste, agy
    python ocr_gemini.py 86_bahia_v1 --de 41 --ate 60 --api --saida ocr_api # teste, API
    python ocr_gemini.py --api --listar-modelos                             # modelos da API
    python ocr_gemini.py --fila --paralelo 3                                # rota gratuita
--fila: o agy pega o que o Batch nao vai pegar: primeiro as paginas do docvirt alem do
orcamento (ocr_comum.horizonte), depois os PDFs da BN. Uma pagina recusada e tentada de
novo uma vez; na segunda vai para ocr_api/_bloqueadas.csv. Pula paginas ja feitas.
"""
import argparse, base64, json, os, subprocess, sys, threading, time, urllib.error
from concurrent.futures import ThreadPoolExecutor
import ocr_comum as c

MODELO_AGY = "gemini-3.8-flash-medium"  # matches flash-high on 86_bahia_v1 41-60, 1.6x faster
TEMPERATURA = float(os.environ.get("OCR_TEMPERATURA", "0"))
_trava = threading.Lock()
_falhas = {}
_recusas = {}  # volume -> refusals of the direct transcription in a row
SO_VOLUMES = set()  # --volumes: the queue works only on these


def prompt_agy(imagem_rel):
    return (
        f"Read the image file {imagem_rel} in this folder with your file viewing tool "
        "(do not run any command or script; just open the image and look at it).\n"
        "Tarefa: transcrever essa pagina escaneada de um catalogo do Projeto Resgate (AHU), "
        "seguindo estas regras:\n\n"
        f"{c.regras()}\n\n{c.FIM} Nao crie nem edite arquivos."
    )


def chama_agy(imagem, modelo):
    dados = c.roda_agy(prompt_agy(imagem.relative_to(c.KIT).as_posix()), modelo)
    if dados.get("status") != "SUCCESS" or not dados.get("response", "").strip():
        raise RuntimeError(f"{dados.get('status')}, resposta vazia, negado: {dados.get('denied_actions')}")
    return dados["response"], dados.get("usage", {})


def chama_api(imagem, modelo, pensamento):
    corpo = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": c.MIME[imagem.suffix],
                             "data": base64.b64encode(imagem.read_bytes()).decode()}},
            {"text": c.prompt_api()},
        ]}],
        "generationConfig": {"temperature": TEMPERATURA},
    }
    if pensamento:
        corpo["generationConfig"]["thinkingConfig"] = {"thinkingLevel": pensamento}
    for tentativa in range(5):
        try:
            dados = c.pede(f"{c.API}/models/{modelo}:generateContent", corpo)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or tentativa == 4:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
            time.sleep(10 * 2 ** tentativa)
    cand = (dados.get("candidates") or [{}])[0]
    partes = cand.get("content", {}).get("parts", [])
    texto = "".join(p.get("text", "") for p in partes if not p.get("thought"))
    if not texto.strip():
        raise RuntimeError(f"resposta vazia, finishReason={cand.get('finishReason')}")
    return texto, dados.get("usageMetadata", {})


def transcreve(volume, pag, args, saida_dir, log):
    imagem = c.imagem(volume, pag)
    saida = saida_dir / volume / f"{pag:04d}.txt"
    if imagem is None or saida.exists():
        return
    trava = c.OCR_AGY / volume / f"{pag:04d}.lock"
    if args.fila:
        trava.parent.mkdir(parents=True, exist_ok=True)
        trava.touch()
    t0 = time.time()
    try:
        # a volume Gemini keeps refusing goes straight to Tesseract, with a direct try every 20 pages
        # BN catalogs are refused about half the time, and a refused call still spends AI Pro quota:
        # agy corrects the Tesseract text there from the start (it passes the RECITATION filter)
        if args.fila and not args.api and volume in c.BN:
            return via_tesseract(volume, pag, "BN: direto para correcao", args)
        if args.fila and _recusas.get(volume, 0) >= 5 and pag % 20:
            return via_tesseract(volume, pag, "volume recusado", args)
        if args.api:
            texto, uso = chama_api(imagem, args.modelo or c.MODELO_API, args.pensamento)
        else:
            texto, uso = chama_agy(imagem, args.modelo or MODELO_AGY)
    except c.CotaEsgotada as e:
        c.espera_cota(e)  # the page stays in the queue, nothing counted against it
        return
    except Exception as e:
        print(f"  ERRO {volume} {pag}: {str(e)[:160]}", flush=True)
        if args.fila and args.api and ("HTTP 503" in str(e) or "HTTP 429" in str(e)):
            time.sleep(300)  # free tier busy or out of quota: wait, the page stays in the queue
            return
        if args.fila:
            _recusas[volume] = _recusas.get(volume, 0) + 1
            via_tesseract(volume, pag, str(e)[:120], args)
        return
    finally:
        trava.unlink(missing_ok=True)
    _recusas[volume] = 0
    motivo = c.suspeita(volume, pag, texto)
    if motivo:
        print(f"  SUSPEITA {volume} {pag}: {motivo} - nao gravada", flush=True)
        if args.fila:
            via_tesseract(volume, pag, f"suspeita: {motivo}", args)
        return
    duracao = time.time() - t0
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(c.limpa(texto), encoding="utf-8")
    with _trava:
        log.write(json.dumps({"volume": volume, "pagina": pag, "api": args.api,
                              "modelo": args.modelo or (c.MODELO_API if args.api else MODELO_AGY),
                              "segundos": round(duracao, 1), "usage": uso}) + "\n")
        log.flush()
        print(f"  {volume} {pag:04d}: {duracao:.0f} s, {json.dumps(uso)}", flush=True)


def via_tesseract(volume, pag, motivo, args):
    """A page Gemini refuses to transcribe: Tesseract reads it and Gemini corrects it (corrigir_tesseract.py).

    The agy queue corrects through agy; an API queue (the student key) through its own key and model,
    so it never spends the AI Pro quota."""
    import corrigir_tesseract as ct
    try:
        if args.api:
            ct.corrige(volume, pag, "texto", ct.c.KIT / "ocr_corrigido", modelo=args.modelo)
        else:
            ct.corrige(volume, pag, "agy", ct.c.KIT / "ocr_corrigido")
    except c.CotaEsgotada as e:
        c.espera_cota(e)
        return
    except urllib.error.HTTPError as e:
        if e.code in (429, 503):
            time.sleep(300)  # free tier busy: not the page's fault
            return
        print(f"  ERRO tesseract {volume} {pag}: HTTP {e.code}", flush=True)
    except Exception as e:
        print(f"  ERRO tesseract {volume} {pag}: {e}", flush=True)
    if not c.feito(volume, pag):
        chave = c.nome(volume, pag)
        _falhas[chave] = _falhas.get(chave, 0) + 1
        if _falhas[chave] >= 2:
            c.bloqueia(volume, pag, "agy+tesseract", motivo)


def proxima(ocupadas, de_tras=False):
    """Next page for a free queue, or None when nothing is left for it right now.

    de_tras: from the last BN page backwards, so a second queue (the student key) meets
    the agy queue at the end instead of competing with it."""
    em_voo, previstas, *_ = c.horizonte()
    sem_teto = c.ORCAMENTO >= 1e5
    # with no cap the Batch would reserve everything: the free queues then take pages from the end
    # (the Batch takes from the front and skips pages they have locked)
    do_batch = em_voo if sem_teto else em_voo | set(previstas)
    bloq = c.bloqueadas()
    if sem_teto:
        ordem = [(vol, pag) for vol in reversed(list(c.DOCVIRT) + c.BN) for pag in reversed(c.paginas(vol))]
    elif de_tras:
        ordem = [(vol, pag) for vol in reversed(c.BN) for pag in reversed(c.paginas(vol))]
    else:
        ordem = [(vol, pag) for vol in list(c.DOCVIRT) + c.BN for pag in c.paginas(vol)]
    for vol, pag in ordem:
        if SO_VOLUMES and vol not in SO_VOLUMES:
            continue
        k = c.nome(vol, pag)
        if (k in do_batch or k in ocupadas or (vol, pag) in bloq or c.feito(vol, pag)
                or _falhas.get(k, 0) >= 2 or (c.OCR_AGY / vol / f"{pag:04d}.lock").exists()):
            continue
        return vol, pag
    return None


def fila(args, log, saida_dir):
    ocupadas = set()

    def trabalhador():
        while True:
            with _trava:
                alvo = proxima(ocupadas, args.de_tras)
                if alvo is None:
                    return
                ocupadas.add(c.nome(*alvo))
            transcreve(*alvo, args, saida_dir, log)
            with _trava:
                ocupadas.discard(c.nome(*alvo))

    with ThreadPoolExecutor(args.paralelo) as pool:
        for _ in range(args.paralelo):
            pool.submit(trabalhador)
    print("fila vazia", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", nargs="?", help="pasta em paginas/, ex.: 86_bahia_v1")
    ap.add_argument("--todos", action="store_true")
    ap.add_argument("--fila", action="store_true", help="rota gratuita (agy) sobre o que o Batch nao cobre")
    ap.add_argument("--de", type=int, default=1)
    ap.add_argument("--ate", type=int, default=9999)
    ap.add_argument("--api", action="store_true", help="usar a API Gemini (GEMINI_API_KEY)")
    ap.add_argument("--modelo", help=f"padrao: {MODELO_AGY} (agy) ou {c.MODELO_API} (API)")
    ap.add_argument("--pensamento", default="low", help="thinkingLevel na API (low, medium, high; '' = padrao)")
    ap.add_argument("--paralelo", type=int, default=1)
    ap.add_argument("--saida", default="ocr", help="pasta de saida (padrao: ocr)")
    ap.add_argument("--listar-modelos", action="store_true")
    ap.add_argument("--de-tras", action="store_true", help="com --fila: os PDFs da BN do fim para o comeco")
    ap.add_argument("--chave", default="GEMINI_API_KEY",
                    help="variavel com a chave da API (ex.: GEMINI_API_KEY_ESTUDANTE, nivel gratuito)")
    ap.add_argument("--volumes", default="", help="com --fila: so estes volumes, separados por virgula")
    args = ap.parse_args()
    SO_VOLUMES.update(v for v in args.volumes.split(",") if v)
    if args.chave != "GEMINI_API_KEY":
        os.environ["GEMINI_API_KEY"] = os.environ.get(args.chave, "")
    if args.listar_modelos:
        for m in c.pede(f"{c.API}/models?pageSize=200").get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].removeprefix("models/"), "-", m.get("displayName"))
        return
    saida_dir = c.KIT / args.saida
    saida_dir.mkdir(exist_ok=True)
    with open(saida_dir / "_log.jsonl", "a", encoding="utf-8") as log:
        if args.fila:
            return fila(args, log, saida_dir)
        volumes = sorted(p.name for p in c.PAGINAS.iterdir() if p.is_dir()) if args.todos else [args.volume]
        if volumes == [None]:
            ap.error("diga o volume, --todos ou --fila")
        for volume in volumes:
            alvo = [p for p in c.paginas(volume) if args.de <= p <= args.ate]
            with ThreadPoolExecutor(args.paralelo) as pool:
                list(pool.map(lambda p: transcreve(volume, p, args, saida_dir, log), alvo))


if __name__ == "__main__":
    sys.exit(main())
