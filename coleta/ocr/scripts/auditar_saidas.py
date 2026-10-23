"""Procura transcricoes inventadas: cotas novas cujo codigo nao e o do volume (ex.: _005 da Bahia
num volume de Pernambuco) ou o texto do exemplo de PROMPT_OCR.md fora da pagina de onde veio.

Uso (na pasta deste kit):  python auditar_saidas.py   -> lista em _suspeitas.csv
"""
import collections, csv, glob, os, re

# AHU_ACL_CU code of each captaincy
CODIGO = {"86_bahia_v1": "005", "87_bahia_v2": "005", "75_sao_paulo_v1": "023", "76_sao_paulo_v2": "023",
          "91_minas_v1": "011", "92_minas_v2": "011", "93_minas_v3": "011", "79_para_v1": "013",
          "80_para_v2": "013", "81_para_v3": "013", "pe_v1": "015", "pe_v2": "015", "pe_v3": "015",
          "pe_v4": "015", "maranhao": "009", "colonia": "012"}
RE = re.compile(r"AHU[\s_]*ACL[\s_]*CU[\s_]*(\d{3})")
EXEMPLO = "Miguel Manso Preto"
PASTAS = ["ocr_api", "ocr", "ocr_corrigido", "ocr_lite"]


def main():
    suspeitas, codigos = [], collections.defaultdict(collections.Counter)
    for d in PASTAS:
        for f in glob.glob(f"{d}/*/[0-9]*.txt"):
            vol = f.split(os.sep)[-2]
            texto = open(f, encoding="utf-8-sig", errors="replace").read()
            cs = RE.findall(texto)
            codigos[vol].update(cs)
            alheios = sum(x != CODIGO.get(vol) for x in cs)
            if cs and alheios > len(cs) / 2:
                suspeitas.append((d, vol, os.path.basename(f), f"{alheios}/{len(cs)} cotas de outro codigo"))
            elif EXEMPLO in texto and not (vol == "86_bahia_v1" and f.endswith("0050.txt")):
                suspeitas.append((d, vol, os.path.basename(f), "texto do exemplo do prompt"))
    with open("_suspeitas.csv", "w", encoding="utf-8", newline="") as s:
        w = csv.writer(s)
        w.writerow(["pasta", "volume", "arquivo", "motivo"])
        w.writerows(suspeitas)
    for d in PASTAS:
        n = len(glob.glob(f"{d}/*/[0-9]*.txt"))
        print(d, n, "arquivos;", sum(1 for x in suspeitas if x[0] == d), "suspeitos")
    for x in suspeitas[:40]:
        print("  ", *x)
    print("codigos por volume:", {v: dict(c.most_common(3)) for v, c in sorted(codigos.items())})


if __name__ == "__main__":
    main()
