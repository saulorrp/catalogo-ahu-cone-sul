"""Parse Projeto Resgate verbete pages (gov.br HTML tables) into JSONL records.

Usage: python parse_verbetes.py PAGE.html URL OUT.jsonl
Each verbete row: <td>numero (link to docreader)</td><td><p>data_local</p><p>resumo</p>
<p>Anexo...</p><p>old cota</p><p>new cota</p></td>
"""
import json, re, sys
from bs4 import BeautifulSoup

from parse_texto import norm_cota
RE_NEW = re.compile(r"^\W{0,2}AHU[_ .]+ACL[_ .]+\S*.*?(Cx\.|D\.|doc\.|Códice|Cod\.)", re.I)
RE_FLS = re.compile(r"^\(?\s*[\d+ ]+\s*fls?\.?\s*\)?\.?$", re.I)
RE_OBS = re.compile(r"^Obs\.?\s*:?", re.I)
RE_OLD = re.compile(r"^(AHU[-–\s]|[A-ZÁÉÍÓÚÂÊÔÃÕÇ ]{4,}, cx\.)", re.I)
RE_ANEXO = re.compile(r"^Anexos?\s*[:;]", re.I)
RE_DATA = re.compile(r"^\[?\s*(ant\.|post\.|ca?\.|s\.\s*d|s/d|\d{3,4})", re.I)
TIPO = re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\- ]{2,})\b")

def limpa(t):
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()

def parse(html, url):
    soup = BeautifulSoup(html, "lxml")
    core = soup.select_one("#content-core") or soup
    out = []
    for tr in core.find_all("tr"):
        tds = [td for td in tr.find_all("td", recursive=False) if limpa(td.get_text(" "))]
        if len(tds) != 2: continue
        num = limpa(tds[0].get_text(" "))
        if not re.fullmatch(r"\d+[A-Z]?\.?", num): continue
        num = num.rstrip(".")
        a = tds[0].find("a", href=True)
        partes = [limpa(p.get_text(" ")) for p in tds[1].find_all(["p", "div"], recursive=True) if not p.find(["p", "div"])]
        if not partes:
            partes = [limpa(x) for x in tds[1].get_text("\n").split("\n")]
        partes = [p for p in partes if p]
        rec = {"numero": num, "data_local": None, "description": None, "anexos": None,
               "old_code": None, "new_code": None, "observacoes": [], "folhas": None, "outros": [], "url_imagem": a["href"] if a else None,
               "caminho_resgate": url}
        resto = []
        for i, p in enumerate(partes):
            if i == 0 and RE_DATA.match(p): rec["data_local"] = p
            elif RE_NEW.search(p) and not rec["new_code"]: rec["new_code"] = p
            elif RE_OBS.match(p): rec["observacoes"].append(p)
            elif RE_FLS.match(p): rec["folhas"] = p
            elif RE_ANEXO.match(p): rec["anexos"] = RE_ANEXO.sub("", p).strip()
            elif RE_OLD.match(p) and not rec["old_code"]: rec["old_code"] = p
            elif p in ("\\",): continue
            else: resto.append(p)
        if resto:
            rec["description"] = resto[0]; rec["outros"] = resto[1:]
        rec["chave_cota"] = norm_cota(rec["new_code"])
        m = TIPO.match(rec["description"] or "")
        rec["tipologia_lite"] = m.group(1).strip() if m else None
        out.append(rec)
    return out

if __name__ == "__main__":
    recs = parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[2])
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(recs), "verbetes")
