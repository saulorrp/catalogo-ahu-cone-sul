"""Rota das paginas que o Gemini recusa (RECITATION): Tesseract le a pagina, o Gemini corrige.

Para cada pagina: tesseract/<volume>/NNNN.txt (texto cru do Tesseract, guardado) e
ocr_corrigido/<volume>/NNNN.txt (corrigido). Variantes de correcao:
  agy       - (padrao, gratis) como "texto", pelo agy com flash-medium
  texto     - imagem + texto do Tesseract; o Gemini devolve o texto corrigido no formato
              de PROMPT_OCR.md
  correcoes - imagem + texto do Tesseract; o Gemini devolve so uma lista JSON de trocas
              {"errado": ..., "certo": ...}, aplicadas aqui; o formato e feito aqui
Uso (na pasta deste kit, com GEMINI_API_KEY):
    python corrigir_tesseract.py 86_bahia_v1 42 47 60 --variante texto --saida teste_correcao
    python corrigir_tesseract.py --bloqueadas --laco   # tudo de ocr_api/_bloqueadas.csv, pelo agy
"""
import argparse, base64, csv, io, json, os, re, subprocess, sys, time, urllib.error
from concurrent.futures import ThreadPoolExecutor
import ocr_comum as c

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA = c.KIT / "tessdata"
CRU = c.KIT / "tesseract"
MODELO_AGY = "gemini-3.8-flash-medium"


def tesseract(volume, pag):
    destino = CRU / volume / f"{pag:04d}.txt"
    if not destino.exists():
        from PIL import Image
        buf = io.BytesIO()
        Image.open(c.imagem(volume, pag)).save(buf, "PNG")
        proc = subprocess.run([TESSERACT, "stdin", "stdout", "--tessdata-dir", str(TESSDATA),
                               "-l", "por", "--psm", "1"], input=buf.getvalue(), capture_output=True)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(proc.stdout)
    return destino.read_text(encoding="utf-8")


def gemini(imagem, texto, modelo=None):
    corpo = {"contents": [{"parts": [
        {"inline_data": {"mime_type": c.MIME[imagem.suffix], "data": base64.b64encode(imagem.read_bytes()).decode()}},
        {"text": texto}]}],
        "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingLevel": "low"}}}
    for tentativa in range(4):
        try:
            d = c.pede(f"{c.API}/models/{modelo or c.MODELO_API}:generateContent", corpo)
            break
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or tentativa == 3:
                raise
            time.sleep(10 * 2 ** tentativa)
    cand = (d.get("candidates") or [{}])[0]
    saida = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return saida, cand.get("finishReason"), c.custo(d.get("usageMetadata", {}), "padrao")


def pedido_texto(cru, onde="anexa"):
    return ("Abaixo esta o texto que um OCR (Tesseract) leu da pagina " + onde + ", de um catalogo do Projeto "
            "Resgate (AHU). O OCR erra letras. Corrija os erros comparando com a imagem e devolva o texto "
            f"corrigido, organizado segundo estas regras:\n\n{c.regras()}\n\n{c.FIM}\n\n"
            f"TEXTO DO OCR:\n{cru}")


def pede_texto(imagem, cru, modelo=None):
    return gemini(imagem, pedido_texto(cru), modelo)


def pede_texto_agy(imagem, cru, modelo):
    """The same request through agy (free); the image is read from disk by agy."""
    pedido = (f"Read the image file {imagem.relative_to(c.KIT).as_posix()} in this folder with your file "
              "viewing tool (do not run any command or script; just open the image and look at it). "
              "Nao crie nem edite arquivos.\n\n" + pedido_texto(cru, "nessa imagem"))
    d = c.roda_agy(pedido, modelo)  # raises CotaEsgotada on a quota error
    texto = d.get("response", "") if d.get("status") == "SUCCESS" else ""  # anything else is a failure
    return texto, f"agy:{d.get('status')} {str(d.get('error') or '')[:80]}".strip(), 0.0


def pede_correcoes(imagem, cru):
    pedido = ("Abaixo esta o texto que um OCR (Tesseract) leu da pagina anexa, de um catalogo do Projeto "
              "Resgate (AHU). Compare com a imagem e liste SO os erros de leitura do OCR, um por linha, "
              'cada um como um objeto JSON {"errado": "<trecho exatamente como esta no OCR>", "certo": '
              '"<o mesmo trecho como impresso>"}. Use trechos curtos (uma a tres palavras), mas unicos no '
              "texto. Nao devolva o texto inteiro, nem nada alem das linhas JSON.\n\n"
              f"TEXTO DO OCR:\n{cru}")
    saida, motivo, custo = gemini(imagem, pedido)
    texto, aplicadas = cru, 0
    for linha in saida.splitlines():
        linha = linha.strip().rstrip(",")
        if not linha.startswith("{"):
            continue
        try:
            t = json.loads(linha)
        except ValueError:
            continue
        if t.get("errado") and t["errado"] in texto:
            texto = texto.replace(t["errado"], t.get("certo", ""), 1)
            aplicadas += 1
    return formata(texto), motivo, custo, aplicadas


COTA_NOVA = re.compile(r"^AHU[ _]ACL[ _]CU")
DATA = re.compile(r"^(\d+\.\s*)?\[?(ant\.|post\.|c\.)?\s*\d{4}\s*[,\]]")


def formata(texto):
    """Lay raw OCR text out as PROMPT_OCR.md asks: drop the running head, one line per part."""
    blocos, atual = [], []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        if DATA.match(linha) and atual and any(COTA_NOVA.match(l) for l in atual):
            blocos.append(atual)
            atual = []
        atual.append(linha)
    if atual:
        blocos.append(atual)
    saida = []
    for bloco in blocos:
        partes, resumo = [], ""
        for linha in bloco:
            if DATA.match(linha) or re.match(r"^(Anexos?:|Obs\.:|AHU)", linha):
                if resumo:
                    partes.append(resumo); resumo = ""
                partes.append(linha)
            elif partes and not resumo and partes[-1].startswith(("Anexo", "Obs.")) and not COTA_NOVA.match(linha):
                partes[-1] += " " + linha
            else:
                resumo = resumo[:-1] + linha if resumo.endswith("-") else (resumo + " " + linha).strip()
        if resumo:
            partes.append(resumo)
        if any(DATA.match(p) for p in partes):
            saida.append("\n".join(partes))
    return "\n\n".join(saida) + "\n"


def quase_branca(cru):
    """Tesseract sees fewer than 30 words and no cota: a blank page or show-through only."""
    return len(re.findall(r"[a-zà-ÿ]{3,}", cru)) < 30 and not re.search(r"AH[UIl1]|ACL", cru)


def corrige(volume, pag, variante, saida_dir, modelo=None):
    imagem = c.imagem(volume, pag)
    cru = tesseract(volume, pag)
    # a (near-)blank page: models invent entries for it (flash-lite wrote 46 Bahia entries for colonia/0370)
    if quase_branca(cru):
        destino = saida_dir / volume / f"{pag:04d}.txt"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text("# SEM VERBETES\n", encoding="utf-8")
        print(f"{volume} {pag:04d}: pagina quase em branco no Tesseract -> # SEM VERBETES", flush=True)
        return 0.0
    if variante == "agy":
        texto, motivo, custo = pede_texto_agy(imagem, cru, MODELO_AGY)
        aplicadas = None
        if not texto.strip():
            print(f"{volume} {pag:04d} agy: {motivo}, sem texto", flush=True)
            return 0.0
    elif variante == "texto":
        texto, motivo, custo = pede_texto(imagem, cru, modelo)
        aplicadas = None
        if not texto.strip() and modelo:
            print(f"{volume} {pag:04d} texto ({modelo}): {motivo}, sem texto", flush=True)
            return custo
        if not texto.strip():
            # blocked again: fall back to the corrections list
            texto, motivo2, custo2, aplicadas = pede_correcoes(imagem, cru)
            motivo, custo = f"{motivo}->correcoes:{motivo2}", custo + custo2
    else:
        texto, motivo, custo, aplicadas = pede_correcoes(imagem, cru)
    suspeita = c.suspeita(volume, pag, texto)
    if suspeita:
        print(f"{volume} {pag:04d} {variante}: SUSPEITA {suspeita} - nao gravada", flush=True)
        return custo
    destino = saida_dir / volume / f"{pag:04d}.txt"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(c.limpa(texto), encoding="utf-8")
    with open(saida_dir / "_log_tesseract.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps({"volume": volume, "pagina": pag, "variante": variante, "motivo": motivo,
                              "custo": round(custo, 5), "correcoes": aplicadas}) + "\n")
    print(f"{volume} {pag:04d} {variante}: {motivo}, US$ {custo:.4f}, correcoes {aplicadas}", flush=True)
    return custo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("volume", nargs="?")
    ap.add_argument("paginas", nargs="*", type=int)
    ap.add_argument("--bloqueadas", action="store_true")
    ap.add_argument("--variante", default="agy", choices=["agy", "texto", "correcoes"])
    ap.add_argument("--saida", default="ocr_corrigido")
    ap.add_argument("--laco", action="store_true", help="com --bloqueadas: repete a cada 10 min")
    ap.add_argument("--paralelo", type=int, default=1)
    args = ap.parse_args()
    saida_dir = c.KIT / args.saida
    tentativas = {}
    while True:
        if args.bloqueadas:
            # the agy queue already tried Tesseract twice on its "agy+tesseract" pages
            with open(c.BLOQUEADAS, encoding="utf-8") if c.BLOQUEADAS.exists() else open(os.devnull) as f:
                alvo = sorted({(r["volume"], int(r["pagina"])) for r in csv.DictReader(f)
                               if r.get("rota") != "agy+tesseract"})
        else:
            alvo = [(args.volume, p) for p in args.paginas]
        fazer = [(v, p) for v, p in alvo
                 if not (saida_dir / v / f"{p:04d}.txt").exists() and tentativas.get((v, p), 0) < 2]
        def uma(vp):
            tentativas[vp] = tentativas.get(vp, 0) + 1
            try:
                corrige(*vp, args.variante, saida_dir)
            except c.CotaEsgotada as e:
                tentativas[vp] -= 1  # not the page's fault
                c.espera_cota(e)

        with ThreadPoolExecutor(args.paralelo) as pool:
            list(pool.map(uma, fazer))
        if not args.laco:
            return
        time.sleep(600)


if __name__ == "__main__":
    sys.exit(main())
