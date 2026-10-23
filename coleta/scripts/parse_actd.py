"""ACTD/IICT text-PDF catalogues (actd.iict.pt, CU-<Nome>.pdf): strip page headers and front matter,
then reuse parse_texto. Usage: python parse_actd.py IN.txt SERIE URL OUT.jsonl
IN.txt comes from pdf_para_texto.py; SERIE is e.g. 015 or 005-01 (used only for reporting)."""
import re, sys, json
import parse_texto

HDR = re.compile(r"^(CAT[ÁA]LOGO\b|INVENT[ÁA]RIO\b|EXISTENTES NO ARQUIVO|.*\bEXISTENTES NO ARQUIVO HIST|AHU_(ACL_)?CU_[\w\-]+\s*$|"
                 r"P(ÁG|AG)?\.?\s*-?\s*\d+\s*(/\s*\d+|-)\s*$)", re.I)
START = re.compile(r"^\s*\d{1,5}[A-Z]?\s*[-–]\s*$")
DATA = re.compile(r"^\s*(\[|\d{3,4}\b|s\.\s*d|ca\.|post\.|ant\.)", re.I)

def limpa(texto):
    # some catalogues (ES) number entries "1." and restart per section: normalise to "1-"
    # MT prints "338. 1751, Junho, 9, ..." on one line; LF/Ceará print a bare "671" line before the date
    linhas = [re.sub(r"^\s*(\d{1,5})\.\s*$", r"\1-", l) for l in texto.splitlines() if not HDR.match(l.strip())]
    linhas = [re.sub(r"^\s*(\d{1,5})\.\s+(?=[\[\d])", r"\1- ", l) for l in linhas]
    for i, l in enumerate(linhas):
        if re.fullmatch(r"\s*\d{1,5}\s*", l):
            nxt = next((x for x in linhas[i + 1:i + 3] if x.strip()), "")
            if DATA.match(nxt): linhas[i] = l.strip() + "-"
    # front matter: skip until the first entry number followed by a date line
    for i, l in enumerate(linhas):
        if START.match(l) or re.match(r"^\s*\d{1,5}-\s+[\[\d]", l):
            nxt = next((x for x in linhas[i + 1:i + 4] if x.strip()), "")
            if DATA.match(nxt): break
    else:
        i = 0
    # keep the last page marker before the first entry so page numbers stay right
    marca = next((linhas[j] for j in range(i, -1, -1) if parse_texto.PAG.match(linhas[j].strip())), None)
    corpo = linhas[i:]
    return "\n".join(([marca] if marca else []) + corpo)

if __name__ == "__main__":
    src, serie, url, out = sys.argv[1:5]
    recs = parse_texto.parse(limpa(open(src, encoding="utf-8").read()), url, "--sem-numero" in sys.argv)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            r["serie_actd"] = serie
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    com = sum(1 for r in recs if r["chave_cota"])
    print(f"{serie}: {len(recs)} verbetes, {com} com cota")
