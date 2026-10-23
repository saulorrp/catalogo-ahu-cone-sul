"""Teste DeepSeek (deepseek-flash, com imagem) nas paginas do Opus. Saida em ocr_deepseek/ (so avaliacao).

(a) transcricao direta da imagem com as regras de PROMPT_OCR.md; (b) --texto: correcao so do texto
do Tesseract (tesseract_amostra). Para se o saldo cair mais que LIMITE. Uso: python teste_deepseek.py [--texto]
"""
import base64, io, json, os, statistics, sys, time, urllib.request
import ocr_comum as c
import qualidade_tesseract as q
import teste_correcao_texto as tt

URL = "https://api.deepseek.com"
LIMITE = 0.70


def pede(caminho, corpo=None):
    h = {"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Content-Type": "application/json"}
    req = urllib.request.Request(URL + caminho, data=corpo and json.dumps(corpo).encode(), headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def saldo():
    return float(pede("/user/balance")["balance_infos"][0]["total_balance"])


def chama(conteudo):
    corpo = {"model": "deepseek-flash", "temperature": 0, "reasoning_effort": "low",
             "messages": [{"role": "user", "content": conteudo}]}
    t0 = time.time()
    d = pede("/chat/completions", corpo)
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {}), time.time() - t0


def main():
    so_texto = "--texto" in sys.argv
    pasta = c.KIT / ("ocr_deepseek_texto" if so_texto else "ocr_deepseek")
    inicio = saldo()
    linhas, segs = [], []
    for f in sorted((c.KIT / "ocr_opus").glob("*/[0-9]*.txt")):
        v, p = f.parent.name, int(f.stem)
        destino = pasta / v / f.name
        if not destino.exists():
            if inicio - saldo() > LIMITE:
                print("limite de gasto atingido"); break
            if so_texto:
                cru = (c.KIT / "tesseract_amostra" / v / f.name).read_text(encoding="utf-8", errors="replace")
                conteudo = tt.PEDIDO.format(regras=c.regras(), fim=c.FIM, cru=cru)
            else:
                from PIL import Image
                buf = io.BytesIO()
                Image.open(c.imagem(v, p)).convert("RGB").save(buf, "PNG")
                conteudo = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                               base64.b64encode(buf.getvalue()).decode()}},
                            {"type": "text", "text": c.prompt_api()}]
            try:
                texto, uso, s = chama(conteudo)
            except urllib.error.HTTPError as e:
                print(v, p, "HTTP", e.code, e.read()[:300]); continue
            segs.append(s)
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(c.limpa(texto) if texto.strip() else "", encoding="utf-8")
            print(f"{v}/{f.stem}: {s:.0f} s, {uso.get('prompt_tokens')} in / {uso.get('completion_tokens')} out", flush=True)
        o, d = f.read_text(encoding="utf-8"), destino.read_text(encoding="utf-8")
        g = q.referencia(v, p)[1]
        linhas.append((f"{v}/{f.stem}", q.wer(o, d)[0], q.wer(o, g)[0], len(q.cotas(o) & q.cotas(d)),
                       len(q.cotas(o) & q.cotas(g)), len(q.cotas(o)), q.datas(o) == q.datas(d), c.suspeita(v, p, d)))
    gasto = inicio - saldo()
    n = len(linhas)
    print(f"\n{'texto-so' if so_texto else 'imagem'}: {n} paginas, gasto US$ {gasto:.3f} (~{gasto / max(n, 1):.4f}/pag), "
          f"{statistics.median(segs) if segs else 0:.0f} s/pag")
    print(f"WER vs Opus: DeepSeek mediana {statistics.median(x[1] for x in linhas):.1%}, max {max(x[1] for x in linhas):.1%}; "
          f"Gemini mediana {statistics.median(x[2] for x in linhas):.1%}")
    print(f"D. iguais ao Opus: DeepSeek {sum(x[3] for x in linhas)}/{sum(x[5] for x in linhas)}, "
          f"Gemini {sum(x[4] for x in linhas)}/{sum(x[5] for x in linhas)}; anos iguais {sum(x[6] for x in linhas)}/{n}; "
          f"vazias {sum(1 for x in linhas if x[1] >= 0.99)}; suspeitas {[x[0] for x in linhas if x[7]]}")
    for x in sorted(linhas, key=lambda x: -x[1])[:4]:
        print("  pior:", x[0], f"WER {x[1]:.1%}, D. {x[3]}/{x[5]}")


if __name__ == "__main__":
    main()
