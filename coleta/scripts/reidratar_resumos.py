"""Reidrata resumos dos documentos a partir dos PDFs do ACTD.

Uso: python reidratar_resumos.py ENTRADA.parquet PASTA_PDFS [--saida SAIDA.parquet] [--checksums ../fontes_checksums.tsv] [--falhas FALHAS.csv]
"""
import sys, os, re, json, hashlib, difflib, argparse, collections
from pathlib import Path
import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
import parse_actd, parse_texto

TEM_ANO = re.compile(r"\b1[4-8]\d\d\b|s\.\s*d", re.I)
DATA_NA_DESC = re.compile(r"^(\[?\s*(?:ant\.|post\.|ca\.)?\s*1[4-8]\d\d\b[^\n]{0,80}?)\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{3,}\b)")

def separa_data(r):
    if TEM_ANO.search(r.get("data_local") or ""): return
    m = DATA_NA_DESC.match(r.get("description") or "")
    if m:
        r["data_local"] = m.group(1).strip(" ,")
        r["description"] = r["description"][m.end():].strip()

def nome_local(url):
    """Converte URL do ACTD no nome de arquivo local <id>__<arquivo>."""
    if not url: return ""
    m = re.search(r"actd:([^/]+)/([^/\s]+)", url)
    if m:
        return f"{m.group(1)}__{m.group(2)}"
    u = url.strip()
    parts = [p for p in u.split("/") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    return parts[-1] if parts else ""

def extrair_candidatos_pdf(pdf_path, url):
    """Extrai e processa verbetes de um PDF, retornando candidatos organizados por página."""
    doc = pymupdf.open(pdf_path)
    texto = ""
    for i, page in enumerate(doc, 1):
        texto += f"=== PAGINA {i} ===\n{page.get_text()}\n"
    
    limpo = parse_actd.limpa(texto)
    recs = parse_texto.parse(limpo, url, sem_numero=False) + parse_texto.parse(limpo, url, sem_numero=True)
    
    page_map = collections.defaultdict(list)
    all_cands = []
    vistos_todas = set()

    for r in recs:
        for c, v in list(r.items()):
            if isinstance(v, str):
                r[c] = re.sub(r"[\x00-\x08\x0b-\x1f]", "", v).strip()
        separa_data(r)
        desc = r.get("description")
        if desc:
            h = hashlib.sha1(desc.encode("utf-8")).hexdigest()
            pg = r.get("pagina")
            if h not in vistos_todas:
                vistos_todas.add(h)
                all_cands.append((h, desc, pg))
            if pg is not None:
                try:
                    pg_int = int(pg)
                    if not any(item[0] == h for item in page_map[pg_int]):
                        page_map[pg_int].append((h, desc))
                except (ValueError, TypeError):
                    pass
    return page_map, all_cands

def carregar_checksums(caminho):
    if not caminho or not Path(caminho).is_file():
        return {}
    c_map = {}
    for l in Path(caminho).read_text(encoding="utf-8").splitlines():
        parts = l.strip().split("\t")
        if len(parts) >= 4 and parts[0] != "url":
            c_map[parts[0].strip()] = parts[3].strip()
    return c_map

def reidratar(entrada_df, pasta_pdfs, checksums=None):
    """Reidrata a coluna description com base na ENTRADA."""
    checksum_map = carregar_checksums(checksums) if isinstance(checksums, (str, Path)) else (checksums or {})
    pasta = Path(pasta_pdfs)
    cache = {}
    checksum_erros = 0
    pdfs_checados = set()

    desc_out = []
    reid_out = []
    falhas = []

    for _, row in entrada_df.iterrows():
        doc_uid = row.get("doc_uid")
        url = row.get("url_actd")
        pag = row.get("pagina_actd")
        sha1_target = str(row.get("sha1_resumo") or "")
        orig_desc = row.get("description") if isinstance(row.get("description"), str) else ""

        if pd.isna(url) or not str(url).strip():
            reid_out.append("url_ausente")
            desc_out.append(None)
            falhas.append({"doc_uid": doc_uid, "url_actd": url, "pagina_actd": pag, "motivo": "url_ausente", "ratio": 0.0, "candidata": ""})
            continue

        url_str = str(url).strip()
        local_filename = nome_local(url_str)
        pdf_path = pasta / local_filename

        if not pdf_path.is_file():
            reid_out.append("pdf_ausente")
            desc_out.append(None)
            falhas.append({"doc_uid": doc_uid, "url_actd": url_str, "pagina_actd": pag, "motivo": "pdf_ausente", "ratio": 0.0, "candidata": ""})
            continue

        if pd.isna(pag) or pag is None or str(pag).strip() == "":
            reid_out.append("sem_pagina")
            desc_out.append(None)
            falhas.append({"doc_uid": doc_uid, "url_actd": url_str, "pagina_actd": pag, "motivo": "sem_pagina", "ratio": 0.0, "candidata": ""})
            continue

        try:
            pag_int = int(pag)
        except (ValueError, TypeError):
            pag_int = None

        if url_str not in cache:
            if url_str in checksum_map and url_str not in pdfs_checados:
                pdfs_checados.add(url_str)
                exp_sha = checksum_map[url_str]
                if exp_sha:
                    act_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
                    if act_sha != exp_sha:
                        print(f"AVISO: SHA-256 divergente para {pdf_path.name}", file=sys.stderr)
                        checksum_erros += 1
            cache[url_str] = extrair_candidatos_pdf(pdf_path, url_str)

        page_map, all_cands = cache[url_str]
        encontrado = None
        nivel = None

        if pag_int is not None and pag_int in page_map:
            for h, d in page_map[pag_int]:
                if h == sha1_target:
                    encontrado = d
                    nivel = "pagina"
                    break

        if not encontrado and pag_int is not None:
            for p_viz in (pag_int - 1, pag_int + 1):
                if p_viz in page_map:
                    for h, d in page_map[p_viz]:
                        if h == sha1_target:
                            encontrado = d
                            nivel = "pagina_vizinha"
                            break
                    if encontrado: break

        if not encontrado:
            for h, d, _ in all_cands:
                if h == sha1_target:
                    encontrado = d
                    nivel = "outra_pagina"
                    break

        if encontrado:
            reid_out.append(nivel)
            desc_out.append(encontrado)
        else:
            reid_out.append("texto_difere")
            desc_out.append(None)

            cand_list = page_map.get(pag_int, []) if (pag_int is not None and pag_int in page_map) else [(h, d) for h, d, _ in all_cands]
            best_cand = ""
            best_ratio = 0.0
            if cand_list:
                if orig_desc:
                    for _, d in cand_list:
                        r_val = difflib.SequenceMatcher(None, orig_desc, d).ratio()
                        if r_val > best_ratio:
                            best_ratio = r_val
                            best_cand = d
                if not best_cand:
                    best_cand = cand_list[0][1]
            falhas.append({"doc_uid": doc_uid, "url_actd": url_str, "pagina_actd": pag, "motivo": "texto_difere", "ratio": round(best_ratio, 4), "candidata": best_cand})

    df_saida = entrada_df.copy()
    df_saida["description"] = desc_out
    df_saida["reidratacao"] = reid_out

    falhas_df = pd.DataFrame(falhas, columns=["doc_uid", "url_actd", "pagina_actd", "motivo", "ratio", "candidata"])

    total = len(df_saida)
    sucessos = df_saida[df_saida["reidratacao"].isin(["pagina", "pagina_vizinha", "outra_pagina"])]
    qtd_sucesso = len(sucessos)
    taxa = round(qtd_sucesso / total * 100, 2) if total > 0 else 0.0

    por_nivel = {k: int(v) for k, v in sucessos["reidratacao"].value_counts().items()}
    falhas_por_motivo = {k: int(v) for k, v in falhas_df["motivo"].value_counts().items()} if not falhas_df.empty else {}
    falhas_por_pdf = {k: int(v) for k, v in falhas_df["url_actd"].value_counts().items()} if not falhas_df.empty else {}

    resumo_dict = {
        "total": total,
        "reidratados": qtd_sucesso,
        "taxa_pct": taxa,
        "por_nivel": por_nivel,
        "falhas_por_motivo": falhas_por_motivo,
        "falhas_por_pdf": falhas_por_pdf,
        "checksum_erros": checksum_erros
    }

    return df_saida, falhas_df, resumo_dict

def main():
    parser = argparse.ArgumentParser(description="Reidrata resumos dos PDFs do ACTD.")
    parser.add_argument("entrada", help="Arquivo ENTRADA.parquet")
    parser.add_argument("pasta_pdfs", help="Pasta contendo os arquivos PDF")
    parser.add_argument("--saida", default=None, help="Caminho do arquivo SAIDA.parquet")
    parser.add_argument("--checksums", default=None, help="Arquivo TSV com checksums SHA-256")
    parser.add_argument("--falhas", default=None, help="Caminho do arquivo FALHAS.csv")
    args = parser.parse_args()

    df_in = pd.read_parquet(args.entrada)
    df_out, df_falhas, resumo = reidratar(df_in, args.pasta_pdfs, checksums=args.checksums)

    # Nunca sobrescreve a entrada: sem --saida, grava ao lado dela.
    saida_path = args.saida or str(Path(args.entrada).with_name(Path(args.entrada).stem + "_reidratado.parquet"))
    df_out.to_parquet(saida_path)

    falhas_path = args.falhas or "falhas.csv"
    df_falhas.to_csv(falhas_path, index=False, encoding="utf-8")

    print(f"=== RESUMO DE REIDRATAÇÃO ===")
    print(f"Total: {resumo['total']}")
    print(f"Reidratados: {resumo['reidratados']} ({resumo['taxa_pct']}%)")
    for niv, qtd in resumo['por_nivel'].items():
        print(f"  - {niv}: {qtd}")
    print(f"Falhas por motivo: {resumo['falhas_por_motivo']}")
    print(f"Falhas por PDF: {resumo['falhas_por_pdf']}")
    if resumo['checksum_erros']:
        print(f"Checksum erros: {resumo['checksum_erros']}")

if __name__ == "__main__":
    main()
