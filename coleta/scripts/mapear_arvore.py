"""Crawl the Projeto Resgate 'novos-instrumentos-de-pesquisa' tree, saving raw HTML and a tree map."""
import json, hashlib, sys, time, re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urldefrag
import requests
from bs4 import BeautifulSoup

BASE = "https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/novos-instrumentos-de-pesquisa"
OUT = Path(sys.argv[1])
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (pesquisa academica; catalogo AHU)"

def norm(u):
    u = urldefrag(u)[0].rstrip("/")
    for suf in ("/view", "/@@images"):
        pass
    return u

def fname(u):
    rel = u[len(BASE):].strip("/") or "_index"
    return RAW / (rel.replace("/", "__") + ".html")

seen, queue, nodes = set(), deque([BASE]), []
while queue:
    url = queue.popleft()
    if url in seen: continue
    seen.add(url)
    f = fname(url)
    if f.exists():
        html, final, status = f.read_text("utf-8"), url, 200
    else:
        for tent in range(4):
            try:
                r = S.get(url, timeout=90, allow_redirects=True); break
            except Exception as e:
                print("erro", url, e, file=sys.stderr); time.sleep(2 ** tent)
        else:
            nodes.append({"url": url, "erro": True}); continue
        final, status, html = r.url, r.status_code, r.text
        time.sleep(0.7)
        if status == 200 and final.startswith(BASE):
            f.write_text(html, "utf-8")
    soup = BeautifulSoup(html, "lxml")
    title = (soup.find("h1") or soup.title)
    title = title.get_text(" ", strip=True) if title else ""
    core = soup.select_one("#content-core") or soup.select_one("#content") or soup
    filhos, externos = [], []
    for a in core.find_all("a", href=True):
        h = norm(urljoin(url, a["href"]))
        if h.startswith(BASE + "/") and h != url and not re.search(r"\.(pdf|jpg|png)$|/@@|/view$|\?", h):
            filhos.append(h)
            if h not in seen: queue.append(h)
        elif h.startswith("http") and "gov.br" not in h:
            externos.append(h)
    nodes.append({"url": url, "final": final, "status": status, "titulo": title,
                  "bytes": len(html), "arquivo": f.name if f.exists() else None,
                  "filhos": sorted(set(filhos)), "externos": sorted(set(externos))})
    print(len(nodes), status, len(html), url[len(BASE):][:110], flush=True)

(OUT / "resgate_arvore.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=1), "utf-8")
