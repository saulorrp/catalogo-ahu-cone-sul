"""O que ocr_gemini.py (agy e API direta) e ocr_batch.py (Batch API) compartilham.

Saidas: ocr_api/<volume>/NNNN.txt (API e Batch) e ocr/<volume>/NNNN.txt (agy).
Uma pagina esta feita se tiver .txt em qualquer das duas.
"""
import csv, json, os, sys, threading, time, urllib.request
from pathlib import Path

KIT = Path(__file__).parent
PAGINAS = KIT / "paginas"
OCR_AGY = KIT / "ocr"
OCR_API = KIT / "ocr_api"
BLOQUEADAS = OCR_API / "_bloqueadas.csv"
LOTES = OCR_API / "_lotes.json"
API = "https://generativelanguage.googleapis.com/v1beta"
MODELO_API = "gemini-3.8-flash"

# docvirt, na ordem de baixar_paginas.py (e da prioridade do Batch), com o total de paginas
DOCVIRT = {"86_bahia_v1": 708, "87_bahia_v2": 478, "75_sao_paulo_v1": 311, "76_sao_paulo_v2": 804,
           "91_minas_v1": 358, "92_minas_v2": 374, "93_minas_v3": 360,
           "79_para_v1": 451, "80_para_v2": 490, "81_para_v3": 262,
           # added 2026-10-22: Ultramar/Brasil-Geral, codices of the CU, cartography (in this priority)
           "88_ultramar_brasil_geral": 622, "84_codices_cu": 179, "89_cartografia": 429}
# PDFs da Biblioteca Nacional, extraidos por extrair_bn.py
BN = ["pe_v1", "pe_v2", "pe_v3", "pe_v4", "maranhao", "colonia"]
BN_BATCH = ["maranhao", "colonia", "pe_v2", "pe_v3", "pe_v4"]
# volumes left out of every queue (owner, 2026-10-22 23:45: finish the main catalogues first)
EM_ESPERA = {"88_ultramar_brasil_geral", "84_codices_cu", "89_cartografia"}
MIME = {".webp": "image/webp", ".jpg": "image/jpeg", ".png": "image/png"}

# US$ por milhao de tokens, gemini-3.8-flash, ate 31/12/2026 (ai.google.dev/gemini-api/docs/pricing)
PRECO = {"padrao": (0.75, 3.75), "batch": (0.375, 1.875)}
ORCAMENTO = 1e6  # no cap: the owner turned auto-reload on for the paid key (2026-10-22 21:42)
ESTIMATIVA_BATCH = 0.005  # US$ por pagina antes de haver lotes concluidos (medido: ~0.0044)

_trava = threading.Lock()


class CotaEsgotada(Exception):
    """agy said the AI Pro quota is used up; `segundos` until it resets. Not the page's fault."""

    def __init__(self, segundos, texto):
        super().__init__(texto)
        self.segundos = segundos


def roda_agy(prompt, modelo):
    """One `agy -p` call in plan mode; returns its JSON. Raises CotaEsgotada on a quota error."""
    import re, subprocess
    proc = subprocess.run(["agy", "-p", prompt, "--model", modelo, "--mode", "plan", "--output-format", "json",
                           "--print-timeout", "900s"], cwd=KIT, capture_output=True)
    bruto = proc.stdout.decode("utf-8", errors="replace")
    erro = proc.stderr.decode("utf-8", errors="replace")
    try:
        dados = json.loads(bruto[bruto.index("{"):])
    except ValueError:
        dados = {"status": "SEM_JSON", "response": "", "error": (bruto[:200] + " " + erro[:200]).strip()}
    msg = f"{dados.get('error') or ''} {erro}"
    if "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
        m = re.search(r"Resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", msg)
        segundos = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + int(m.group(3) or 0)) if m else 1800
        raise CotaEsgotada(max(segundos, 60), msg.strip()[:200])
    return dados


_pausa_ate = 0.0


def espera_cota(e):
    """Every worker that hits the quota waits until it resets (plus a minute), once."""
    global _pausa_ate
    with _trava:
        if time.time() >= _pausa_ate:
            _pausa_ate = time.time() + e.segundos + 60
            print(f"{time.strftime('%H:%M')} COTA agy esgotada; pausa ate "
                  f"{time.strftime('%H:%M', time.localtime(_pausa_ate))}: {e}", flush=True)
    time.sleep(max(0, _pausa_ate - time.time()))


def nome(volume, pag):
    return f"{volume}/{pag:04d}"


def imagem(volume, pag):
    for ext in MIME:
        p = PAGINAS / volume / f"{pag:04d}{ext}"
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def paginas(volume):
    pasta = PAGINAS / volume
    if not pasta.is_dir():
        return []
    return sorted(int(p.stem) for p in pasta.iterdir()
                  if p.suffix in MIME and p.stem.isdigit() and p.stat().st_size > 1000)


def feito(volume, pag):
    return any((d / volume / f"{pag:04d}.txt").exists() for d in (OCR_AGY, OCR_API, KIT / "ocr_corrigido", KIT / "ocr_lite"))


def regras():
    """The rules and the example from PROMPT_OCR.md, without the batch instructions."""
    texto = (KIT / "PROMPT_OCR.md").read_text(encoding="utf-8")
    return texto[texto.index("## Regras de transcricao"):texto.index("## Ao terminar cada volume")].strip()


FIM = ("IMPORTANTE: responda SOMENTE com a transcricao da pagina, em texto puro, sem "
       "comentarios, sem introducao, sem cercas ``` e sem links. Se a pagina estiver em branco, ou so "
       "tiver texto transparecendo do verso, responda apenas `# SEM VERBETES`; nunca repita o exemplo.")

# AHU_ACL_CU code of each catalogue; a page full of another catalogue's code was invented
# (seen once: flash-lite wrote 46 Bahia entries for a blank Colonia page)
CODIGO = {"86_bahia_v1": "005", "87_bahia_v2": "005", "75_sao_paulo_v1": "023", "76_sao_paulo_v2": "023",
          "91_minas_v1": "011", "92_minas_v2": "011", "93_minas_v3": "011", "79_para_v1": "013",
          "80_para_v2": "013", "81_para_v3": "013", "pe_v1": "015", "pe_v2": "015", "pe_v3": "015",
          "pe_v4": "015", "maranhao": "009", "colonia": "012"}


def suspeita(volume, pag, texto):
    """Why a transcription looks invented, or None."""
    import re
    cs = re.findall(r"AHU[\s_]*ACL[\s_]*CU[\s_]*(\d{3})", texto)
    outros = set(CODIGO.values()) - {CODIGO.get(volume)}
    # catalogues that span captaincies (Brasil-Geral, codices, cartography) have no single code
    if volume in CODIGO and len(cs) >= 3 and sum(x in outros for x in cs) > len(cs) / 2:
        return f"{len(cs)} cotas de outra capitania ({cs[0]})"
    cru = KIT / "tesseract" / volume / f"{pag:04d}.txt"
    if len(cs) >= 5 and cru.exists() and not re.search(r"AH[UIl1]|ACL|[CcEe]x\.", cru.read_text(encoding="utf-8", errors="replace")):
        return f"{len(cs)} cotas, mas o Tesseract nao ve nenhuma"
    return None


def prompt_api():
    return ("Tarefa: transcrever a pagina escaneada anexa, de um catalogo do Projeto Resgate "
            f"(AHU), seguindo estas regras:\n\n{regras()}\n\n{FIM}")


def limpa(resposta):
    texto = resposta.strip()
    if texto.startswith("```") and texto.endswith("```"):
        texto = texto.split("\n", 1)[1].rsplit("```", 1)[0].strip() if "\n" in texto else ""
    return texto + "\n"


def custo(uso, tarifa):
    entrada, saida = PRECO[tarifa]
    return (uso.get("promptTokenCount", 0) * entrada
            + (uso.get("candidatesTokenCount", 0) + uso.get("thoughtsTokenCount", 0)) * saida) / 1e6


def bloqueadas():
    if not BLOQUEADAS.exists():
        return set()
    with open(BLOQUEADAS, encoding="utf-8") as f:
        return {(r["volume"], int(r["pagina"])) for r in csv.DictReader(f)}


def bloqueia(volume, pag, rota, motivo):
    with _trava:
        novo = not BLOQUEADAS.exists()
        BLOQUEADAS.parent.mkdir(exist_ok=True)
        with open(BLOQUEADAS, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if novo:
                w.writerow(["volume", "pagina", "rota", "motivo"])
            w.writerow([volume, pag, rota, motivo])


def le_lotes():
    if not LOTES.exists():
        return {"lotes": [], "tentativas": {}}
    for _ in range(5):
        try:
            return json.loads(LOTES.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            time.sleep(1)  # being rewritten by ocr_batch.py
    raise RuntimeError("nao consegui ler _lotes.json")


def grava_lotes(estado):
    tmp = LOTES.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=1), encoding="utf-8")
    os.replace(tmp, LOTES)


def gasto_teste():
    """What the standard-price API test (ocr_gemini.py --api) has spent."""
    log = OCR_API / "_log.jsonl"
    if not log.exists():
        return 0.0
    return sum(custo(json.loads(l)["usage"], "padrao") for l in log.read_text(encoding="utf-8").splitlines() if l)


def horizonte(estado=None):
    """The docvirt pages the Batch still means to take: (in flight, still affordable ones).

    The affordable ones are the first pages, in DOCVIRT order, that are neither done nor
    blocked nor in a batch, as many as the money left pays for. The agy route leaves them alone.
    """
    estado = estado or le_lotes()
    em_voo, gasto, pendente = set(), gasto_teste(), 0.0
    concluidas = [l for l in estado["lotes"] if l.get("custo_real") is not None]
    por_pagina = (sum(l["custo_real"] for l in concluidas) / max(1, sum(l["n"] for l in concluidas))
                  if concluidas else ESTIMATIVA_BATCH) * 1.1
    for l in estado["lotes"]:
        if l.get("custo_real") is not None:
            gasto += l["custo_real"]
        elif not l.get("liberado"):
            em_voo.update(l["chaves"])
            pendente += l["n"] * por_pagina
    cabe = max(0, int((ORCAMENTO - gasto - pendente) / por_pagina))
    bloq = bloqueadas()
    previstas = []
    # docvirt first; then the BN catalogues the Batch is least refused on (Pernambuco is refused ~50%)
    ordem = [(v, t) for v, t in DOCVIRT.items() if v not in EM_ESPERA] + [(v, len(paginas(v))) for v in BN_BATCH]
    for vol, total in ordem:
        for pag in range(1, total + 1):
            if len(previstas) >= cabe:
                break
            if nome(vol, pag) not in em_voo and (vol, pag) not in bloq and not feito(vol, pag):
                previstas.append(nome(vol, pag))
    return em_voo, previstas, gasto, pendente, por_pagina


def pede(url, corpo=None, metodo=None, cabecalhos=None, dados=None, bruto=False):
    chave = os.environ.get("GEMINI_API_KEY")
    if not chave:
        sys.exit("GEMINI_API_KEY nao definida")
    h = {"x-goog-api-key": chave, "Content-Type": "application/json", **(cabecalhos or {})}
    if corpo is not None:
        dados = json.dumps(corpo).encode()
    req = urllib.request.Request(url, data=dados, headers=h, method=metodo)
    with urllib.request.urlopen(req, timeout=900) as r:
        conteudo = r.read()
        return (conteudo, r.headers) if bruto else json.loads(conteudo)
