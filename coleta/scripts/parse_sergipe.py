"""Sergipe catalog layout: number line, 'Caixa: 01, Documento Nº: 86, 1700, junho, 16, Lisboa', summary.
The page carries no AHU_ACL cota per verbete, so new_code is rebuilt from caixa + documento
(series 022) and flagged with cota_reconstruida=True."""
import re

CAB = re.compile(r"^Caixa\s*:?\s*(\d+)\s*,\s*Documento\s*N[º°o]?\s*:?\s*(\d+)\s*(?:-\s*([A-Z]))?\s*,?\s*(.*)$", re.I)

def parse(texto, url, serie="022"):
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    out = []
    for i, l in enumerate(linhas):
        m = CAB.match(l)
        if not m: continue
        num = linhas[i - 1] if i and re.fullmatch(r"\d+[A-Z]?", linhas[i - 1]) else None
        desc = []
        for nl in linhas[i + 1:]:
            if CAB.match(nl) or re.fullmatch(r"\d+[A-Z]?", nl): break
            desc.append(nl)
        d = " ".join(desc)
        anexo = re.search(r"\((Anexos?\s*:?\s*[^)]*)\)\.?\s*$", d)
        cx, doc, suf = int(m.group(1)), int(m.group(2)), m.group(3) or ""
        out.append({"numero": num, "data_local": m.group(4).strip() or None, "description": d,
                    "anexos": re.sub(r"^Anexos?\s*:?\s*", "", anexo.group(1)) if anexo else None,
                    "old_code": None, "new_code": f"AHU_ACL_CU_{serie}, Cx. {cx}, D. {doc}{suf}.",
                    "cota_reconstruida": True, "observacoes": [], "caminho_resgate": url,
                    "chave_cota": f"{serie}|{cx}|{doc}{suf}",
                    "tipologia_lite": (d.split()[0].strip(",.").upper() if d else None)})
    return out
