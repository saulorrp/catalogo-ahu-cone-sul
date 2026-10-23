"""Códices pages: table rows [Códice, Fls., resumo, Dia, Mês, Ano]. One record per registered entry."""
import re
from bs4 import BeautifulSoup

def parse(html, url):
    core = BeautifulSoup(html, "lxml").select_one("#content-core")
    out, titulo = [], None
    for tr in core.find_all("tr"):
        c = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(c) >= 6 and c[0] == "Códice": titulo = c[2]; continue
        if len(c) < 6 or not re.fullmatch(r"\d+", c[0]) or not re.match(r"\d", c[1]): continue
        cod, fl, resumo, dia, mes, ano = c[:6]
        out.append({"numero": f"{cod}-{fl}", "data_local": ", ".join(x for x in (ano, mes, dia) if x),
                    "description": resumo, "anexos": None, "old_code": None,
                    "new_code": f"AHU_ACL_CU, Cód. {cod}, fl. {fl}.", "observacoes": [],
                    "codice": cod, "titulo_codice": titulo, "caminho_resgate": url,
                    "chave_cota": f"cod{cod}|fl{fl.replace(' ', '')}",
                    "tipologia_lite": resumo.split()[0].strip(",.").upper() if resumo else None})
    return out
