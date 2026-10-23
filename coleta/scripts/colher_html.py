"""Parse every saved gov.br verbete page listed in resgate_arvore.json into verbetes/<slug>.jsonl.

Usage: python colher_html.py /mnt/project-files/resgate
Index pages (onomástico, toponímico, assuntos...) are kept raw but not parsed.
"""
import json, re, sys
from pathlib import Path
from bs4 import BeautifulSoup
sys.path.insert(0, str(Path(__file__).parent))
import parse_verbetes, parse_texto, parse_sergipe, parse_codices

BASE = "https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/novos-instrumentos-de-pesquisa"
R = Path(sys.argv[1])
arvore = json.loads((R / "resgate_arvore.json").read_text("utf-8"))
por_url = {n["url"]: n for n in arvore}
pai = {f: n["url"] for n in arvore for f in n.get("filhos", [])}
(R / "verbetes").mkdir(exist_ok=True)

def capitania(url):
    """Title of the top-level catalog page under BASE, e.g. 'Goiás' catalog."""
    rel = url[len(BASE):].strip("/").split("/")
    topo = por_url.get(BASE + "/" + rel[0], {})
    return topo.get("titulo") or rel[0]

relatorio = []
for n in arvore:
    if not n.get("arquivo") or n["url"] == BASE: continue
    if re.search(r"indice|índice", n["url"] + " " + n.get("titulo", ""), re.I): continue
    html = (R / "raw" / n["arquivo"]).read_text("utf-8")
    recs = parse_verbetes.parse(html, n["url"])
    modo = "tabela"
    if "codice" in n["url"]:
        recs = parse_codices.parse(html, n["url"]); modo = "codice"
    if modo != "codice" and sum(1 for r in recs if r["chave_cota"]) < 5:
        core = BeautifulSoup(html, "lxml").select_one("#content-core")
        texto = core.get_text("\n", strip=True) if core else ""
        recs = [r for r in parse_texto.parse(texto, n["url"]) if r["chave_cota"]]
        modo = "texto"
        if len(recs) < 5:
            recs = parse_sergipe.parse(texto, n["url"]); modo = "caixa-documento"
    if not recs: continue
    cap = capitania(n["url"])
    slug = n["url"][len(BASE):].strip("/").split("/")[-1]
    with open(R / "verbetes" / f"{slug}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            r.update(capitania=cap, titulo_pagina=n.get("titulo"), formato_fonte="html")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nums = [int(re.sub(r"\D", "", r["numero"]) or 0) for r in recs if re.match(r"\d", r["numero"] or "")]
    faltam = sorted(set(range(min(nums), max(nums) + 1)) - set(nums)) if nums else []
    relatorio.append({"pagina": slug, "capitania": cap, "modo": modo, "verbetes": len(recs),
                      "com_cota": sum(1 for r in recs if r["chave_cota"]),
                      "com_data": sum(1 for r in recs if r["data_local"]),
                      "numeros": [min(nums), max(nums)] if nums else None,
                      "lacunas": len(faltam), "lacunas_ex": faltam[:15]})
    print(json.dumps(relatorio[-1], ensure_ascii=False))
(R / "relatorio_html.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), "utf-8")
