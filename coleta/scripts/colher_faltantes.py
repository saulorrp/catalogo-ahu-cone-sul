"""Parse the pages saved by mapear_faltantes.py (raw/<path below projeto-resgate>.html) into verbetes/.
 - contratos-capitanias/*   -> verbetes/contratos-<pagina>.jsonl          fonte_tipo='contratos', tema='contratos'
 - catalogos-tematicos/*/*  -> verbetes/tematico-<tema>-<pagina>.jsonl    fonte_tipo='tematico',  tema=<tema>
 - Coleção Pombalina (BNP)  -> colecoes_outras/bnp-pombalina.jsonl        colecao='bnp-pombalina' (NOT in verbetes/)
 - Série Reino / Códices under the new branch: skipped when the text is identical to the page already harvested.
Writes relatorio_faltantes.json. Existing parsers are used unchanged.
Usage: python colher_faltantes.py /mnt/project-files/resgate"""
import json, re, sys, hashlib
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).parent))
import parse_verbetes, parse_texto, parse_sergipe

PR = "https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate"
R = Path(sys.argv[1])
novo = json.loads((R / "resgate_arvore_faltantes.json").read_text("utf-8"))
velho = json.loads((R / "resgate_arvore.json").read_text("utf-8"))
(R / "verbetes").mkdir(exist_ok=True); (R / "colecoes_outras").mkdir(exist_ok=True)

TEMAS = {"copy_of_catalogo-de-documentos-sobre-mulheres": "mulheres", "escravidao-negra": "escravidao",
         "fauna-flora": "semiarido", "irmandades": "irmandades", "historia-dos-povos-indigenas": "povos-indigenas",
         "caminhos": "caminhos-rio-da-prata"}
RE_OLD_FIM = re.compile(r"(?:^|[\s.;])(AHU\s*[-–][^.]*?(?:cx\.|doc\.|n\.v\.|pasta)[^.]*\.?)\s*$", re.I)

def core_html(path):
    return BeautifulSoup(path.read_text("utf-8"), "lxml").select_one("#content-core")
def sig(core): return hashlib.sha1(" ".join(core.get_text(" ", strip=True).split()).encode()).hexdigest()

ja = {}
for n in velho:
    if n.get("arquivo") and (R / "raw" / n["arquivo"]).exists():
        ja[sig(core_html(R / "raw" / n["arquivo"]))] = n["arquivo"]

def old_code_do_anexo(r):
    """Pages put 'Anexo: x. AHU-ALAGOAS, cx. 1, doc. 7.' on one line: move the old cota out."""
    if r.get("old_code"): return
    for campo in ("anexos",):
        v = r.get(campo)
        if v and (m := RE_OLD_FIM.search(v)):
            r["old_code"] = m.group(1).strip(); r[campo] = v[:m.start(1)].strip(" .") or None; return
    obs = r.get("observacoes") or []
    for i, v in enumerate(obs):
        if m := RE_OLD_FIM.search(v):
            r["old_code"] = m.group(1).strip(); obs[i] = v[:m.start(1)].strip() ; return

BLOCOS = ["p", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "section", "blockquote"]
def texto_blocos(core):
    """One line per leaf block element (<br> splits lines); inline tags are joined with spaces. Used when
    get_text('\\n') shatters a verbete into one token per line (mulheres: Minas, RJ...)."""
    core = BeautifulSoup(str(core), "lxml")
    for br in core.find_all("br"): br.replace_with("\n")
    linhas = []
    for el in core.find_all(BLOCOS):
        if el.find(BLOCOS): continue
        for parte in el.get_text(" ").split("\n"):
            parte = " ".join(parte.split())
            if parte: linhas.append(parte)
    return "\n".join(linhas)

def normaliza_numeracao(texto):
    """'12.   1700, Maio, 3' and a bare '57.' -> '12 - 1700, ...' / '57 -' (what parse_texto.INICIO expects)."""
    return re.sub(r"(?m)^\s*(\d{1,5}[A-Z]?)\s*\.(?!\d)\s*", r"\1 - ", texto)

RE_ACL = re.compile(r"AHU[_ .]+(?:ACL[_ .]+)?CU[_ .]*[0-9OIlS\-]+[,.\s]*[CEG€(][xX].{0,25}?D(?:oc)?\.?\s*\d+[^\s]*", re.I)
def corrige_serie(texto):
    """OCR typos in the cota: series AHU_ACL_CU_0ll / 00S-0l -> 011 / 005-01; 'Cx..' -> 'Cx.';
    'Caixa: 10 Doc.: 1004' (Maranhão) -> 'Cx. 10, D. 1004'."""
    tr = str.maketrans("OIlS", "0115")
    def f(m):
        if not any(ch.isdigit() for ch in m.group(2)): return m.group(0)
        return m.group(1) + m.group(2).translate(tr) + (("-" + m.group(3).translate(tr)) if m.group(3) else "")
    texto = re.sub(r"(AHU[_ .]+(?:ACL[_ .]+)?CU[_ .]*)([0-9OIlS]{3})(?:-([0-9OIlS]{2}))?(?![0-9A-Za-z])", f, texto)
    texto = re.sub(r"(?m)^(.*AHU[_ .]+(?:ACL[_ .]+)?CU.*?)Caixa\s*:?\s*(\d+)\s*,?\s*Doc(?:umento)?\.?\s*:?\s*(\d)", r"\1Cx. \2, D. \3", texto)
    return re.sub(r"(AHU[_ .]+(?:ACL[_ .]+)?CU[^\n]*?[CEGce][xX])\.\.", r"\1.", texto)

def separa_cota(texto):
    """'Em anexo: 1 documento. AHU_ACL_CU_009, Cx. 1 Doc.: 70' -> cota moved to its own line."""
    return re.sub(r"(?m)^(?!\W{0,2}AHU)(\S.*?)[\s.]+(AHU[_ .]+(?:ACL[_ .]+)?CU[_ .]*[0-9OIlS\-]{3,6}\s*,\s*[CEG€(][xX].*)$", r"\1.\n\2", texto)

def recupera_cota(r):
    """Table rows sometimes run old and new cota together ('AHU-CEARÁ, cx.1, doc. 71. AHU_ACL_CU_006, Cx. 1, D. 55.')
    or put the new cota inside anexos/observacoes: lift it into new_code."""
    if r.get("chave_cota"): return
    campos = [("old_code", r.get("old_code")), ("anexos", r.get("anexos")), ("description", r.get("description"))]
    campos += [("observacoes", o) for o in (r.get("observacoes") or [])] + [("outros", o) for o in (r.get("outros") or [])]
    for nome, v in campos:
        if not v: continue
        m = RE_ACL.search(corrige_serie(v))
        if not m: continue
        cota = m.group(0).strip(" ,;")
        if not parse_texto.norm_cota(cota): continue
        r["new_code"] = cota if cota.endswith(".") else cota + "."
        r["chave_cota"] = parse_texto.norm_cota(cota)
        resto = corrige_serie(v).replace(m.group(0), "").strip(" ,;.") or None
        if nome == "old_code": r["old_code"] = resto
        elif nome == "anexos": r["anexos"] = resto
        elif nome == "description": r["description"] = resto
        elif nome == "observacoes": r["observacoes"] = [x if x != v else (resto or "") for x in r["observacoes"]]
        elif nome == "outros": r["outros"] = [x for x in r["outros"] if x != v] + ([resto] if resto else [])
        return

RE_COD = re.compile(r"AHU_ACL_CU,?\s*COD\.?,?\s*(\d+),?\s*fls?\.?\s*(.+?)\.?\s*$", re.I)
def parse_codices_tematico(texto, url):
    """Thematic 'Códices' pages: 'N. date' / summary / 'AHU_ACL_CU, COD, 258, fls. 41v.'"""
    out, atual = [], None
    for l in normaliza_numeracao(texto).splitlines():
        l = l.strip()
        m = re.match(r"^(\d{1,5})\s*-\s*(.*)$", l)
        if m and (atual is None or atual["fechado"]):
            atual = {"numero": m.group(1), "data": m.group(2).strip() or None, "desc": [], "fechado": False}; continue
        if atual is None or atual["fechado"] or not l: continue
        c = RE_COD.search(l)
        if c:
            fl = re.sub(r"\s+", "", c.group(2)).rstrip(".")
            out.append({"numero": atual["numero"], "data_local": atual["data"], "description": " ".join(atual["desc"]) or None,
                        "anexos": None, "old_code": None, "new_code": f"AHU_ACL_CU, Cód. {c.group(1)}, fl. {c.group(2).rstrip('.')}.",
                        "observacoes": [], "codice": c.group(1), "caminho_resgate": url,
                        "chave_cota": f"cod{c.group(1)}|fl{fl}",
                        "tipologia_lite": (atual["desc"][0].split()[0].strip(",.").upper() if atual["desc"] else None)})
            atual["fechado"] = True
        else: atual["desc"].append(l)
    return out

RE_ANTIGO = re.compile(r"^AHU\s*[–\-]\s*Cons\.?\s*Ultram\.?\s*[–\-]\s*Brasil/(\w+)\s*[–\-]\s*Cx\.?:?\s*(\d+)[,\s]*Doc\.?:?\s*(\d+)", re.I)
def parse_antigo(texto, url):
    """Mulheres/Minas Gerais: 'N- 1715, 11, 28 – Lisboa' / summary / 'AHU – Cons. Ultram. – Brasil/MG – Cx.: 1, Doc.: 48'.
    Only the OLD cota is printed (no AHU_ACL_CU code), so chave_cota stays None; chave_antiga = 'MG|cx|doc'."""
    out, atual = [], None
    for l in texto.splitlines():
        l = l.strip()
        m = re.match(r"^(\d{1,5})-\s*(.*)$", l)
        if m and (atual is None or atual["fechado"]):
            atual = {"numero": m.group(1), "data": m.group(2).strip(" –-") or None, "desc": [], "fechado": False}; continue
        if atual is None or atual["fechado"] or not l: continue
        a = RE_ANTIGO.match(l)
        if a:
            desc = " ".join(atual["desc"]).strip()
            data = atual["data"]
            if desc.startswith(("–", "-")) and not data:
                pass
            desc = desc.lstrip("–- ").strip()
            data = re.sub(r"^A(\d{3})", r"1\1", data) if data else data
            out.append({"numero": atual["numero"], "data_local": data, "description": desc or None, "anexos": None,
                        "old_code": l, "new_code": None, "observacoes": [], "caminho_resgate": url, "chave_cota": None,
                        "chave_antiga": f"{a.group(1).upper()}|{int(a.group(2))}|{int(a.group(3))}",
                        "tipologia_lite": (desc.split()[0].strip(",.").upper() if desc else None)})
            atual["fechado"] = True
        else: atual["desc"].append(l)
    return out

def parse_catalogo(html, url, core, slug=""):
    if slug == "codices":
        recs = parse_codices_tematico(core.get_text("\n", strip=True), url)
        if recs: return recs, "codices-tematico"
    recs = parse_verbetes.parse(html, url); modo = "tabela"
    for r in recs: recupera_cota(r)
    if sum(1 for r in recs if r["chave_cota"]) < 5:
        melhor, modo = [], "texto"
        for nome, base in (("texto", core.get_text("\n", strip=True)), ("texto-blocos", texto_blocos(core))):
            for norm_num in (False, True):  # try with and without 'N.' -> 'N -' renumbering; keep the best
                texto = separa_cota(corrige_serie(normaliza_numeracao(base) if norm_num else base))
                rs = [r for r in parse_texto.parse(texto, url) if r["chave_cota"]]
                if len(rs) > len(melhor): melhor, modo = rs, nome + ("+num" if norm_num else "")
        recs = melhor
        for r in recs: old_code_do_anexo(r); recupera_cota(r)
        if len(recs) < 5:
            r2 = parse_sergipe.parse(core.get_text("\n", strip=True), url)
            if len(r2) > len(recs): recs, modo = r2, "caixa-documento"
        if len(recs) < 5:
            r3 = parse_antigo(texto_blocos(core), url)
            if len(r3) > len(recs): recs, modo = r3, "cota-antiga"
    return recs, modo

RE_PBA = re.compile(r"^PBA\s*(\d+)(?:\s*(?:a|e)\s*PBA\s*(\d+))?\s*[–\-]?\s*(.*)$")
def parse_pombalina(core, url):
    linhas = [l.strip() for l in core.get_text("\n", strip=True).splitlines() if l.strip()]
    j = []
    for k, l in enumerate(linhas):  # 'PBA' / '626 -' split over two lines
        if j and j[-1] == "PBA" and re.match(r"^\d+\s*[–\-]?$", l): j[-1] = "PBA " + l
        else: j.append(l)
    blocos, atual = [], None
    for l in j:
        m = RE_PBA.match(l)
        if m and not re.match(r"^Tomo\b", l):
            atual = {"n": m.group(1), "fim": m.group(2), "l": [m.group(3).strip()] if m.group(3).strip() else []}; blocos.append(atual)
        elif atual is not None:
            if l == "-" and not atual["l"]: continue
            atual["l"].append(l)
    out = []
    for b in blocos:
        L = b["l"]
        desc = L[0] if L else None
        cota = next((l for l in L if re.search(r"Cotas?\s*:", l)), None)
        ano = re.findall(r"\b(1[5-8]\d\d)\b", " ".join(L[:2]))
        num = b["n"] + (f"-{b['fim']}" if b["fim"] else "")
        out.append({"numero": num, "data_local": f"{ano[0]}-{ano[-1]}" if ano else None, "description": desc,
                    "anexos": None, "old_code": cota, "new_code": f"BNP, Coleção Pombalina, PBA {num}",
                    "observacoes": [], "outros": L[1:], "digitalizado": any("digitalizado" in l.lower() and "microfil" not in l.lower() for l in L),
                    "caminho_resgate": url, "chave_cota": f"bnp-pombalina|{num}", "colecao": "bnp-pombalina",
                    "tipologia_lite": "CODICE"})
    return out

relatorio, vistos_sig = [], {}
for n in novo:
    if not n.get("arquivo"): continue
    url, rel = n["url"], n["url"][len(PR) + 1:]
    partes = rel.split("/"); f = R / "raw" / n["arquivo"]
    html = f.read_text("utf-8"); core = BeautifulSoup(html, "lxml").select_one("#content-core")
    if core is None: continue
    reg = {"pagina": rel, "titulo": n.get("titulo")}
    if sig(core) in ja:
        reg.update(tipo="duplicado", igual_a=ja[sig(core)], verbetes=0)
        relatorio.append(reg); continue
    if len(core.get_text(strip=True)) > 1500 and sig(core) in vistos_sig:
        reg.update(tipo="duplicado-interno", igual_a=vistos_sig[sig(core)], verbetes=0); relatorio.append(reg); continue
    vistos_sig[sig(core)] = rel
    slug = partes[-1]
    if partes[0] == "catalogo-de-verbetes-codices-avulsos-e-series" and len(partes) >= 3 and partes[1] == "contratos-capitanias":
        fonte, tema, nome = "contratos", "contratos", f"contratos-{slug}"
    elif partes[0] == "catalogos-tematicos" and len(partes) >= 3:
        tema = next((v for k, v in TEMAS.items() if partes[1].startswith(k)), partes[1])
        fonte, nome = "tematico", f"tematico-{tema}-{slug}"
    elif partes[0].startswith("catalogo-de-verbetes-codices") and len(partes) >= 3 and partes[1].startswith("brasil-na-colecao-pombalina"):
        if "codices-com-documentos" in slug:
            recs = parse_pombalina(core, url)
            for r in recs: r.update(titulo_pagina=n.get("titulo"), formato_fonte="html", capitania="Coleção Pombalina (BNP)")
            with open(R / "colecoes_outras" / "bnp-pombalina.jsonl", "w", encoding="utf-8") as fo:
                for r in recs: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            reg.update(tipo="bnp-pombalina", modo="codices PBA", verbetes=len(recs)); relatorio.append(reg)
        else:
            reg.update(tipo="texto-introdutorio", verbetes=0); relatorio.append(reg)
        continue
    else:
        reg.update(tipo="indice/nao-catalogo", verbetes=0); relatorio.append(reg); continue
    recs, modo = parse_catalogo(html, url, core, slug)
    if not recs:
        reg.update(tipo="sem-verbetes", verbetes=0, caracteres=len(core.get_text())); relatorio.append(reg); continue
    cap = n.get("titulo") or slug
    with open(R / "verbetes" / f"{nome}.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            r.update(capitania=cap, titulo_pagina=n.get("titulo"), formato_fonte="html", fonte_tipo=fonte, tema=tema)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    nums = [int(re.sub(r"\D", "", r["numero"]) or 0) for r in recs if re.match(r"\d", r["numero"] or "")]
    faltam = sorted(set(range(min(nums), max(nums) + 1)) - set(nums)) if nums else []
    reg.update(tipo=fonte, tema=tema, arquivo=nome + ".jsonl", modo=modo, verbetes=len(recs),
               com_cota=sum(1 for r in recs if r["chave_cota"]), com_data=sum(1 for r in recs if r["data_local"]),
               numeros=[min(nums), max(nums)] if nums else None, lacunas=len(faltam), lacunas_ex=faltam[:15])
    relatorio.append(reg); print(json.dumps(reg, ensure_ascii=False))
(R / "relatorio_faltantes.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), "utf-8")
