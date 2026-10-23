"""Parse verbetes from plain text (PDF text layer or OCR) into JSONL records.

Usage: python parse_texto.py TEXT.txt CAMINHO_URL OUT.jsonl
Text has verbetes like:
  144- 1645, Agosto, 19, Lisboa
  CARTA RÉGIA ... (several lines)
  Anexo: lembrete.
  AHU-Rio de Janeiro, cx. 2, doc. 66.
  AHU_ACL_CU_017, Cx. 2, D. 144.
Page markers '\f' or lines '=== PAGINA n ===' carry the page number.
"""
import json, re, sys

CAB = re.compile(r"^(CATÁLOGO DE DOCUMENTOS|EXISTENTES NO ARQUIVO|AHU_ACL_CU_BRASIL|PÁG\.\s*-)", re.I)
INICIO = re.compile(r"^\s*(\d{1,5}[A-Z]?)\s*(?:[-–]\s*|\s+(?=\[|\d{3,4},|ca\.|s\.\s*d))(.*)$")
PAG = re.compile(r"^=== PAGINA (\d+) ===$")
RE_NEW = re.compile(r"^\W{0,2}AHU[_ .]+(ACL[_ .]+)?CU", re.I)
RE_FIM = re.compile(r"(D(oc)?[.:]*|pasta|fls?\.?)\s*\d+", re.I)
RE_OLD = re.compile(r"^AHU\s*[-–]", re.I)
RE_ANEXO = re.compile(r"^Anexos?\s*[:;]?", re.I)
RE_OBS = re.compile(r"^(Obs\.?\s*:?|Anexa? ao n)", re.I)
RE_DATA = re.compile(r"^\[?\s*(ant\.|post\.|ca?\.|s\.\s*d|s/d|\d{3,4})", re.I)
TIPO = re.compile(r"^([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ\- ]{2,})\b")

def norm_cota(c):
    m = re.search(r"AHU[_ ]+CU[_ ]+Reino[,.\s]+Cx\.?\s*(\d+(?:\s*[-–]?\s*[A-Z](?![a-z]))?)[,.\s]+pasta\s*(\d+)", c or "", re.I)
    if m: return f"076|{re.sub(r'[^0-9A-Z]', '', m.group(1))}|p{int(m.group(2))}"
    # 'ACL' optional (Minas contracts print AHU_CU_011); 'ex.'/'Gx.' are typos of 'Cx.' seen in the thematic pages.
    # Series with a suffix keep it as digits: 005-01 -> '00501', 005-02 -> '00502', 017-01 -> '01701'.
    m = re.search(r"AHU[_ .]+(?:ACL[_ .]+)?CU[_ .]*([\d\- ]+?)[,.\s]*[C€(EG][xX]\.?\s*(\d*)[A-Z]?[,.\s]+D(?:oc)?[.:]*\s*(\d{1,3}(?:\.\d{3})+(?!\d)|\d+)", c or "", re.I)
    if not m: return None
    cx = int(m.group(2)) if m.group(2) else 0
    return f"{re.sub(r'\D', '', m.group(1)).zfill(3)}|{cx}|{int(m.group(3).replace('.', ''))}"  # 'D. 20.367' = 20367

def junta(linhas):
    out = ""
    for l in linhas:
        l = l.strip()
        if not l: continue
        if out.endswith("-") and not out.endswith(" -"): out = out + l
        else: out = (out + " " + l).strip()
    return re.sub(r"\s+", " ", out)

def blocos(texto):
    pagina, atual, ultimo = None, None, 0
    for linha in texto.splitlines():
        s = linha.strip()
        mp = PAG.match(s)
        if mp: pagina = int(mp.group(1)); continue
        if not s or CAB.match(s): continue
        m = INICIO.match(s)
        num = int(re.sub(r"\D", "", m.group(1))) if m else None
        plausivel = m and 0 < num - ultimo <= 60 and (not m.group(2).strip() or RE_DATA.match(m.group(2).strip()))
        if m and (atual is None or atual["fechado"] or plausivel):
            if atual:
                if atual["fechado"]: ultimo = int(re.sub(r"\D", "", atual["numero"]))
                yield atual
            atual = {"numero": m.group(1), "linhas": [m.group(2)] if m.group(2).strip() else [], "pagina": pagina, "fechado": False}
            continue
        if atual is None: continue
        atual["linhas"].append(s)
        ult = atual["linhas"][-3:]
        k = max((i for i, l in enumerate(ult) if RE_NEW.match(l)), default=None)
        if k is not None and RE_FIM.search(junta(ult[k:])): atual["fechado"] = True
    if atual: yield atual

def blocos_por_cota(texto):
    """For catalogs without verbete numbers: each block ends at its new-cota line."""
    pagina, linhas, pag_ini = None, [], None
    for linha in texto.splitlines():
        s = linha.strip()
        mp = PAG.match(s)
        if mp: pagina = int(mp.group(1)); continue
        if not s or CAB.match(s): continue
        if not linhas: pag_ini = pagina
        linhas.append(s)
        if RE_NEW.match(s) and RE_FIM.search(s):
            m = re.search(r"D\.?\s*(\d+[\d\-, ]*)", s)
            yield {"numero": m.group(1).strip(" ,.") if m else "", "linhas": linhas, "pagina": pag_ini}
            linhas = []

def parse(texto, url, sem_numero=False):
    out = []
    for b in (blocos_por_cota(texto) if sem_numero else blocos(texto)):
        L = [l for l in b["linhas"] if l.strip()]
        if not any(RE_NEW.match(l) for l in L): continue
        rec = {"numero": b["numero"], "data_local": None, "description": None, "anexos": None,
               "old_code": None, "new_code": None, "observacoes": [], "pagina": b["pagina"],
               "caminho_resgate": url}
        # group lines into fields: a field starts at a line matching a marker
        campos, cur = [], None
        for i, l in enumerate(L):
            tipo = ("new" if RE_NEW.match(l) else "old" if RE_OLD.match(l) else
                    "anexo" if RE_ANEXO.match(l) else "obs" if RE_OBS.match(l) else
                    "data" if i == 0 and RE_DATA.match(l) else None)
            if tipo or cur is None:
                cur = [tipo or "desc", [l]]; campos.append(cur)
            elif cur[0] == "data" and len(cur[1]) == 1 and not cur[1][0].rstrip().endswith(("]", ")")) and "[" in cur[1][0]:
                cur[1].append(l)  # date wrapped across lines
            elif cur[0] == "data":
                cur = ["desc", [l]]; campos.append(cur)
            else:
                cur[1].append(l)
        for tipo, ls in campos:
            t = junta(ls)
            if tipo == "data": rec["data_local"] = t
            elif tipo == "desc": rec["description"] = (rec["description"] + " " + t) if rec["description"] else t
            elif tipo == "anexo": rec["anexos"] = RE_ANEXO.sub("", t).strip()
            elif tipo == "obs": rec["observacoes"].append(t)
            elif tipo == "old": rec["old_code"] = t
            elif tipo == "new" and not rec["new_code"]: rec["new_code"] = t
        rec["chave_cota"] = norm_cota(rec["new_code"])
        m = TIPO.match(rec["description"] or "")
        rec["tipologia_lite"] = m.group(1).strip() if m else None
        out.append(rec)
    return out

if __name__ == "__main__":
    recs = parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[2], "--sem-numero" in sys.argv)
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(recs), "verbetes")
