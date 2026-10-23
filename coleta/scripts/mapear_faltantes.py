"""Crawl the extra Projeto Resgate branches (catalogo-de-verbetes-codices-avulsos-e-series, catalogos-tematicos,
guia-de-fontes-sobre-o-brasil-no-exterior, mapas-e-iconografia, acervo). Follows <a href>, iframe src and
data-url attributes (tab content). Saves raw HTML as raw/<path-below-projeto-resgate with __>.html, linked
PDFs to fontes/, and writes resgate_arvore_faltantes.json (redirects to docvirt are recorded, not followed).
Usage: python mapear_faltantes.py /mnt/project-files/resgate   (resumable; 1 request/second)"""
import json, re, sys, time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urldefrag, urlparse, parse_qs
import requests
from bs4 import BeautifulSoup

PR = "https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate"
RAIZES = ["catalogo-de-verbetes-codices-avulsos-e-series", "catalogos-tematicos",
          "guia-de-fontes-sobre-o-brasil-no-exterior", "mapas-e-iconografia", "acervo"]
OUT = Path(sys.argv[1]); RAW = OUT / "raw"; FONTES = OUT / "fontes"
RAW.mkdir(exist_ok=True); FONTES.mkdir(exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0 (pesquisa academica; catalogo AHU)"
arq = OUT / "resgate_arvore_faltantes.json"
nodes = {n["url"]: n for n in json.loads(arq.read_text("utf-8"))} if arq.exists() else {}

def norm(u): return urldefrag(u)[0].rstrip("/")
def fname(u): return RAW / (u[len(PR):].strip("/").replace("/", "__") + ".html")
def get(url):
    for t in range(4):
        try:
            r = S.get(url, timeout=90, allow_redirects=False); time.sleep(1.0); return r
        except Exception as e:
            print("erro", url, e, file=sys.stderr); time.sleep(2 ** t)
    return None

queue, seen = deque(PR + "/" + r for r in RAIZES), set()
while queue:
    url = queue.popleft()
    if url in seen: continue
    seen.add(url)
    f = fname(url)
    if url in nodes and nodes[url].get("status") and (f.exists() or nodes[url].get("redirect") or url.lower().endswith(".pdf")):
        n = nodes[url]
        html = f.read_text("utf-8") if f.exists() else ""
    else:
        r = get(url)
        if r is None: nodes[url] = {"url": url, "erro": True}; continue
        n = {"url": url, "status": r.status_code}
        html = ""
        if r.status_code in (301, 302, 303, 307, 308):
            loc = urljoin(url, r.headers.get("Location", "")); n["redirect"] = loc
            if loc.startswith(PR) and norm(loc) != url: queue.append(norm(loc))
            nodes[url] = n; print(r.status_code, url[len(PR):], "->", loc[:120], flush=True); continue
        ct = r.headers.get("Content-Type", "")
        if "pdf" in ct or url.lower().endswith(".pdf"):
            nome = FONTES / re.sub(r"[^\w.\-]+", "_", url[len(PR):].strip("/").replace("/", "__"))
            nome.write_bytes(r.content); n.update(pdf=nome.name, bytes=len(r.content), formato="pdf")
            nodes[url] = n; print("PDF", url[len(PR):], len(r.content), flush=True); continue
        html = r.text; n["bytes"] = len(html)
        if r.status_code == 200: f.write_text(html, "utf-8"); n["arquivo"] = f.name
    if not html: nodes[url] = n; continue
    soup = BeautifulSoup(html, "lxml")
    t = soup.find("h1") or soup.title
    n["titulo"] = t.get_text(" ", strip=True) if t else ""
    core = soup.select_one("#content-core") or soup.select_one("#content") or soup
    filhos, externos, pdfs = set(), set(), set()
    cand = [(a["href"], a.get_text(" ", strip=True)) for a in core.find_all("a", href=True)]
    cand += [(e["data-url"], "") for e in core.find_all(attrs={"data-url": True})]
    cand += [(e["src"], "") for e in core.find_all("iframe", src=True)]
    for h, txt in cand:
        h = norm(urljoin(url + "/", h)) if not h.startswith("http") else norm(h)
        if h.startswith(PR + "/") and h != url:
            if re.search(r"\.pdf(/|$)|/@@download|/view$", h, re.I) and not h.endswith("/view"): pdfs.add(h)
            if re.search(r"/@@images|\.(jpg|png|jpeg|gif)$|\?", h, re.I): continue
            if any(h.startswith(PR + "/" + r) for r in RAIZES):
                filhos.add(h)
                if h not in seen: queue.append(h)
        elif h.startswith("http") and "gov.br" not in h:
            externos.add(h)
    n.update(filhos=sorted(filhos), externos=sorted(externos), pdfs=sorted(pdfs))
    nodes[url] = n
    print(len(nodes), n["status"], len(html), url[len(PR):][:120], flush=True)
    arq.write_text(json.dumps(list(nodes.values()), ensure_ascii=False, indent=1), "utf-8")
arq.write_text(json.dumps(list(nodes.values()), ensure_ascii=False, indent=1), "utf-8")
