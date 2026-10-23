"""Cross-check the OCR of the scanned Resgate catalogues against the ACTD text, by cota.

For each volume, takes the best transcription of each page (Gemini corrected > Gemini batch >
Gemini free > Gemini lite > raw Tesseract), parses the verbetes with parse_texto (blocks end at the
new-cota line), and compares each one with the record of the same chave_cota in ahu_brasil_lite.jsonl.

Writes qc/conferencia_ocr.jsonl (one line per OCR verbete matched or not) and
qc/RELATORIO_CONFERENCIA_OCR.md. Nothing in the collection is changed.

Usage: python conferir_ocr.py R OCR_DIR
  R        working folder with ahu_brasil_lite.jsonl and qc/qc_flags.jsonl
  OCR_DIR  folder with ocr_corrigido/, ocr_api/, ocr/, ocr_lite/, ocr_agy_medium/, tesseract/
"""
import collections, difflib, json, re, statistics, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import parse_texto  # noqa: E402

PRIORIDADE = ["ocr_corrigido", "ocr_api", "ocr", "ocr_lite", "ocr_agy_medium", "tesseract"]
LIMIAR = 0.9


def norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\W+", " ", t).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def ano(t):
    m = re.search(r"\b1[4-8]\d\d\b", t or "")
    return m.group(0) if m else None


def ano_ocr(o):
    """The year of an OCR verbete. When the verbete number sits on its own line before the date,
    the parser folds the date into the start of the description."""
    return ano(o["data_local"]) or ano((o["description"] or "")[:60])


def uniformiza(texto):
    """Maranhão prints the cota as 'AHU-ACL-CU-009 Caixa: 97 Doc.: 07893'."""
    texto = re.sub(r"AHU-ACL-CU-(\d+)\s+Caixa:\s*(\d+)\s+Doc\.?:\s*0*(\d+)",
                   r"AHU_ACL_CU_\1, Cx. \2, D. \3.", texto)
    return texto


def paginas(ocr_dir):
    """volume -> {page number: (source folder, text)}, best source first."""
    vols = collections.defaultdict(dict)
    for fonte in PRIORIDADE:
        for arq in sorted((ocr_dir / fonte).glob("*/*.txt")):
            m = re.match(r"(\d+)", arq.stem)
            if not m: continue
            vol, pag = arq.parent.name, int(m.group(1))
            if pag not in vols[vol]:
                vols[vol][pag] = (fonte, arq.read_text(encoding="utf-8", errors="replace"))
    return vols


def main():
    R, ocr_dir = Path(sys.argv[1]), Path(sys.argv[2])
    por_chave = collections.defaultdict(list)
    for linha in open(R / "ahu_brasil_lite.jsonl", encoding="utf-8"):
        r = json.loads(linha)
        if r.get("chave_cota"): por_chave[r["chave_cota"]].append(r)
    flags = {}
    for linha in open(R / "qc" / "qc_flags.jsonl", encoding="utf-8"):
        f = json.loads(linha); flags[f["doc_uid"]] = f["qc"]

    saida, resumo = [], {}
    for vol, pags in sorted(paginas(ocr_dir).items()):
        texto = uniformiza("\n".join(f"=== PAGINA {p} ===\n{pags[p][1]}" for p in sorted(pags)))
        recs = parse_texto.parse(texto, vol, sem_numero=True)
        fontes = collections.Counter(f for f, _ in pags.values())
        sims, sem_par = [], 0
        for o in recs:
            cands = por_chave.get(o["chave_cota"] or "", [])
            linha = {"volume": vol, "pagina": o["pagina"], "chave_cota": o["chave_cota"],
                     "new_code_ocr": o["new_code"], "data_ocr": o["data_local"],
                     "description_ocr": o["description"]}
            if not cands:
                sem_par += 1; linha["situacao"] = "sem_par_no_actd"; saida.append(linha); continue
            melhor = max(cands, key=lambda c: sim(c.get("description"), o["description"]))
            s = sim(melhor.get("description"), o["description"])
            sims.append(s)
            linha.update({"doc_uid": melhor["doc_uid"], "similaridade": round(s, 3),
                          "qc": flags.get(melhor["doc_uid"], []),
                          "data_actd": melhor.get("data_local"),
                          "ano_ocr": ano_ocr(o),
                          "ano_confere": ano(melhor.get("data_local")) == ano_ocr(o),
                          "situacao": "confere" if s >= LIMIAR else "diverge"})
            saida.append(linha)
        resumo[vol] = {"paginas": len(pags), "fontes": dict(fontes), "verbetes": len(recs),
                       "sem_cota": sum(1 for o in recs if not o["chave_cota"]),
                       "sem_par": sem_par, "com_par": len(sims),
                       "mediana": round(statistics.median(sims), 3) if sims else None,
                       "confere": sum(s >= LIMIAR for s in sims)}

    (R / "qc").mkdir(exist_ok=True)
    with open(R / "qc" / "conferencia_ocr.jsonl", "w", encoding="utf-8") as f:
        for linha in saida: f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    pares = [x for x in saida if "doc_uid" in x]
    marcados = [x for x in pares if x["qc"]]
    por_marca = collections.defaultdict(lambda: [0, 0])
    for x in marcados:
        for m in x["qc"]:
            por_marca[m][0] += 1; por_marca[m][1] += x["situacao"] == "confere"
    sem_data = [x for x in marcados if "sem_data" in x["qc"]]
    sem_data_resolvida = sum(1 for x in sem_data if x["ano_ocr"])

    L = ["# Conferência do OCR com o texto do ACTD", "",
         "Cada verbete transcrito por OCR dos catálogos escaneados do Projeto Resgate foi comparado, "
         "pela cota, com o verbete do mesmo documento no banco (texto do ACTD). Similaridade de "
         f"texto normalizado (sem acentos, caixa e pontuação); confere a partir de {LIMIAR}.", "",
         f"Verbetes do OCR com cota no banco: {len(pares)}. Conferem: "
         f"{sum(x['situacao'] == 'confere' for x in pares)} "
         f"({100 * sum(x['situacao'] == 'confere' for x in pares) / max(len(pares), 1):.1f}%). "
         f"Ano igual: {sum(x['ano_confere'] for x in pares)}.", "",
         "## Por volume", "",
         "| Volume | Páginas | Verbetes | Sem cota | Sem par | Com par | Mediana | Conferem |",
         "|---|---|---|---|---|---|---|---|"]
    for v, r in resumo.items():
        L.append(f"| {v} | {r['paginas']} | {r['verbetes']} | {r['sem_cota']} | {r['sem_par']} | "
                 f"{r['com_par']} | {r['mediana']} | {r['confere']} |")
    L += ["", "## Verbetes marcados pelo QC que o OCR alcança", "",
          f"{len(marcados)} verbetes marcados têm par no OCR.", "",
          "| Marca | Com par no OCR | Texto confere |", "|---|---|---|"]
    for m, (n, ok) in sorted(por_marca.items()):
        L.append(f"| {m} | {n} | {ok} |")
    L += ["", f"Dos {len(sem_data)} marcados `sem_data` com par, o OCR traz um ano em "
          f"{sem_data_resolvida}.", "",
          "## Leitura", "",
          "- Volumes com 0 verbetes são índices (81_para_v3, pe_v4), ainda sem OCR completo "
          "(89_cartografia) ou com cota em outro formato (Minas Gerais, que imprime a cota antiga).",
          "- Os Códices (84_codices_cu) não estão no ACTD: o OCR é a única fonte de texto deles, "
          "e por isso não há com que comparar.",
          "- O catálogo do Maranhão impresso abrevia o texto (REQ., cap.), enquanto o ACTD o "
          "escreve por extenso. A similaridade baixa ali mede a diferença de edição, não erro.",
          "- O ano do OCR é o primeiro número entre 1400 e 1899 no início do verbete. Quando o "
          "número do verbete cai nessa faixa ele é lido como ano, então a taxa de ano igual é um "
          "piso.", "",
          "Fontes de cada página, por ordem de preferência: " + ", ".join(PRIORIDADE) + ".",
          "Detalhe por verbete: `conferencia_ocr.jsonl`."]
    (R / "qc" / "RELATORIO_CONFERENCIA_OCR.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
