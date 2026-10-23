"""Mede a qualidade do Tesseract cru contra a transcricao direta do Gemini, numa amostra.

Amostra: paginas com transcricao direta do Gemini (ocr_api, ocr ou ocr_lite; nunca as de
ocr_corrigido, que partem do proprio Tesseract). O Tesseract delas e rodado numa pasta a parte
(tesseract_amostra/), com as mesmas opcoes do estoque. Grava qualidade_tesseract.md.

Uso (na pasta deste kit):  python qualidade_tesseract.py [--por-volume 10]
"""
import argparse, difflib, io, os, random, re, statistics, subprocess
import ocr_comum as c

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
AMOSTRA = c.KIT / "tesseract_amostra"
GRUPOS = {"Bahia": ["86_bahia_v1", "87_bahia_v2"], "Sao Paulo": ["75_sao_paulo_v1", "76_sao_paulo_v2"],
          "Minas": ["91_minas_v1", "92_minas_v2"], "Pernambuco v1": ["pe_v1"], "Colonia": ["colonia"],
          "Maranhao": ["maranhao"]}
# new-style AHU quotas: AHU_ACL_CU_005, Cx. 6, D. 502 (OCR spaces/underscores and Cx/cx vary)
RE_COTA = re.compile(r"AHU[\s_.,]*ACL[\s_.,]*CU[\s_.,]*(\d{3}(?:-\d{2})?)\s*,?\s*[Cc]x\.?\s*(\d+)\s*,?\s*D\.?\s*(\d+)")
RE_DATA = re.compile(r"^\s*(?:\d+\.\s*)?\[?(?:ant\.|post\.|c\.)?\s*(1[5-8]\d\d)\s*[,\]]", re.M)


def referencia(vol, pag):
    for d in ("ocr_api", "ocr"):  # not ocr_lite: flash-lite invented a page once
        f = c.KIT / d / vol / f"{pag:04d}.txt"
        if f.exists():
            return d, f.read_text(encoding="utf-8-sig")
    return None, None


def tesseract(vol, pag):
    destino = AMOSTRA / vol / f"{pag:04d}.txt"
    if not destino.exists():
        from PIL import Image
        buf = io.BytesIO()
        Image.open(c.imagem(vol, pag)).save(buf, "PNG")
        r = subprocess.run([TESSERACT, "stdin", "stdout", "--tessdata-dir", str(c.KIT / "tessdata"), "-l", "por",
                            "--psm", "1"], input=buf.getvalue(), capture_output=True)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(r.stdout)
    return destino.read_text(encoding="utf-8", errors="replace")


def palavras(t):
    return re.findall(r"\w+|[^\w\s]", re.sub(r"-\n(\w)", r"\1", t))


def wer(ref, hip):
    a, b = palavras(ref), palavras(hip)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    erros = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return erros / max(len(a), 1), sm


def chars(ref, hip):
    n = lambda t: re.sub(r"\s+", " ", re.sub(r"-\n(\w)", r"\1", t)).strip()
    return difflib.SequenceMatcher(None, n(ref), n(hip), autojunk=False).ratio()


def cotas(t):
    """Document numbers of the new-style cotas: `D. 502` (AHU_ACL_CU) or `Doc.: 47` (Minas).

    Only the number is compared, so a Tesseract `ex` for `Cx` does not count as a miss."""
    return set(re.findall(r"(?:\bD\.|\bDoc\.\s*:?)\s*(\d{1,6})\b", t))


def datas(t):
    return [m for m in RE_DATA.findall(t)]


def opus():
    """Gemini and Tesseract against the Opus transcriptions (ocr_opus/, made from the image only)."""
    linhas = ["", "## Contra a transcricao do Opus (referencia independente)", "",
              "| pagina | Gemini: similaridade / WER | Tesseract: similaridade / WER | D. Opus | D. Gemini iguais | "
              "D. Tesseract iguais | datas Opus/Gemini/Tess. |", "|---|---|---|---|---|---|---|"]
    divergencias = []
    for f in sorted((c.KIT / "ocr_opus").glob("*/[0-9]*.txt")):
        v, p = f.parent.name, int(f.stem)
        o = f.read_text(encoding="utf-8")
        _, g = referencia(v, p)
        t = (AMOSTRA / v / f.name).read_text(encoding="utf-8", errors="replace") if (AMOSTRA / v / f.name).exists() else None
        if g is None or t is None:
            continue
        co, cg, ct = cotas(o), cotas(g), cotas(t)
        linhas.append(f"| {v}/{f.stem} | {chars(o, g):.3f} / {wer(o, g)[0]:.1%} | {chars(o, t):.3f} / {wer(o, t)[0]:.1%} "
                      f"| {len(co)} | {len(co & cg)} | {len(co & ct)} | {len(datas(o))}/{len(datas(g))}/{len(datas(t))} |")
        for x in sorted(co ^ cg, key=int):
            divergencias.append(f"{v}/{f.stem}: D. {x} so no {'Opus' if x in co else 'Gemini'}"
                                f"{' (Tesseract tem)' if x in ct else ''}")
        if datas(o) != datas(g):
            a, b = datas(o), datas(g)
            sm = difflib.SequenceMatcher(None, a, b)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    divergencias.append(f"{v}/{f.stem}: anos Opus {a[i1:i2]} x Gemini {b[j1:j2]}")
    linhas += ["", "Divergencias de numero de documento ou ano entre Opus e Gemini:", ""]
    linhas += [f"- {d}" for d in divergencias] or ["- nenhuma"]
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-volume", type=int, default=10)
    args = ap.parse_args()
    random.seed(7)
    linhas, exemplos, piores = [], {}, []
    tabela = ["| grupo | pag. | ref. | similaridade (caract.) | WER | cotas Gemini | cotas Tesseract iguais | inicios (datas) G/T |",
              "|---|---|---|---|---|---|---|---|"]
    for grupo, vols in GRUPOS.items():
        # keep the pages already read into tesseract_amostra (the Opus sample uses them)
        amostra = [(v, int(f.stem)) for v in vols for f in sorted((AMOSTRA / v).glob("[0-9]*.txt"))
                   if referencia(v, int(f.stem))[1] and len(cotas(referencia(v, int(f.stem))[1])) >= 3]
        if len(amostra) < args.por_volume:
            candidatas = [(v, p) for v in vols for p in c.paginas(v)
                          if (v, p) not in amostra and referencia(v, p)[1]]
            candidatas = [x for x in candidatas if len(cotas(referencia(*x)[1])) >= 3]  # pages with entries
            amostra += random.sample(candidatas, min(args.por_volume - len(amostra), len(candidatas)))
        if not amostra:
            continue
        sims, wers, ct_g, ct_ok, ini_g, ini_t, refs = [], [], 0, 0, 0, 0, set()
        for v, p in amostra:
            d, ref = referencia(v, p)
            refs.add(d)
            hip = tesseract(v, p)
            s = chars(ref, hip); w, sm = wer(ref, hip)
            sims.append(s); wers.append(w)
            cg, ch = cotas(ref), cotas(hip)
            ct_g += len(cg); ct_ok += len(cg & ch)
            ini_g += len(RE_DATA.findall(ref)); ini_t += len(RE_DATA.findall(hip))
            piores.append((w, f"{v}/{p:04d}", s))
            a, b = palavras(ref), palavras(hip)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "replace" and i2 - i1 <= 3 and j2 - j1 <= 3:
                    exemplos.setdefault(grupo, []).append((" ".join(b[j1:j2]), " ".join(a[i1:i2]), f"{v}/{p:04d}"))
        tabela.append(f"| {grupo} | {len(amostra)} | {', '.join(sorted(refs))} | {statistics.median(sims):.3f} "
                      f"(min {min(sims):.3f}) | {statistics.median(wers):.1%} | {ct_g} | {ct_ok} ({ct_ok / max(ct_g, 1):.0%}) "
                      f"| {ini_g}/{ini_t} |")
    linhas += ["# Qualidade do Tesseract cru (amostra)", "",
               "Referencia: a transcricao direta do Gemini da mesma pagina (Batch `ocr_api`, agy `ocr` ou "
               "chave de estudante `ocr_lite`), nunca as de `ocr_corrigido`, que partem do proprio Tesseract. "
               "Tesseract 5.4, `-l por --psm 1`, modelo tessdata_best. WER por palavra/pontuacao. Cotas: "
               "`AHU_ACL_CU_NNN, Cx. N, D. N` normalizadas (codigo, caixa, documento).", ""] + tabela
    linhas += opus()
    linhas += ["", "## Exemplos de erro do Tesseract (Tesseract -> Gemini)", ""]
    for grupo, ex in exemplos.items():
        amostra = random.sample(ex, min(8, len(ex)))
        linhas.append(f"- **{grupo}**: " + "; ".join(f"`{t}` -> `{g}` ({pg})" for t, g, pg in amostra))
    linhas += ["", "## Paginas onde o Tesseract foi pior", ""]
    for w, pg, s in sorted(piores, reverse=True)[:6]:
        linhas.append(f"- {pg}: WER {w:.1%}, similaridade {s:.3f}")
    (c.KIT / "qualidade_tesseract.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
