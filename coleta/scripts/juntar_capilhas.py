"""Junta ao índice `corpus/resgate_docreader.tsv.gz` (e atualiza `app_dados/`) os quadros
recuperados pelas Fases 1 e 2 de leitura de capilhas (`~/resgate_capilhas/`).

Regras estritas para zero erro:
  1. Nunca sobrescreve uma referência já presente em `corpus/resgate_docreader.tsv.gz`.
  2. Aceita somente linhas com status `RESOLVIDO_TESSERACT` ou `RESOLVIDO_GEMINI`.
  3. Resolve cada número de documento pelo `mapa_canonico_ca.json`:
     - caso único (apenas 1 `reference_code` distinto); ou
     - desambiguado de forma única pela caixa da pasta (`cx == cx_parent` e sem link prévio).
     Qualquer outro caso ambíguo é descartado.
  4. Se a mesma referência aparecer em mais de uma página de capilha, usa o menor `pagfis`
     (preferindo Fase 2 sobre Fase 1 caso haja sobreposição).
  5. Confere que toda referência nova existe em `app_dados/documentos.parquet` e que nenhuma
     linha que já possuía `resgate_docreader` é alterada.

Uso: python coleta/scripts/juntar_capilhas.py [--capilhas-dir ~/resgate_capilhas] [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
INDICE = RAIZ / "corpus" / "resgate_docreader.tsv.gz"
DOCS_PARQUET = RAIZ / "app_dados" / "documentos.parquet"
MANIFEST_JSON = RAIZ / "app_dados" / "manifest.json"
SERIE_BIB = {"017-1_RJ": "017-001", "005_BA_CA": "005-001"}


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while pedaco := f.read(1024 * 1024):
            h.update(pedaco)
    return h.hexdigest()


def carregar_jsonl(caminho: Path) -> dict[str, dict]:
    itens: dict[str, dict] = {}
    if not caminho.exists():
        return itens
    with open(caminho, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                itens[obj["id"]] = obj
    return itens


def coletar_novos_links(capilhas_dir: Path, existentes: dict[str, tuple[str, int]],
                        refs_validas: set[str]) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
    mapa = json.loads((capilhas_dir / "mapa_canonico_ca.json").read_text(encoding="utf-8"))
    fila1 = carregar_jsonl(capilhas_dir / "fila_capilhas.jsonl")
    fila2 = carregar_jsonl(capilhas_dir / "fila_fase2.jsonl")
    res1 = carregar_jsonl(capilhas_dir / "ocr_results.jsonl")
    res2 = carregar_jsonl(capilhas_dir / "ocr_results_fase2.jsonl")

    candidatos_por_ref: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    stats = {
        "ignorados_existentes": 0,
        "ignorados_ambiguos": 0,
        "ignorados_fora_parquet": 0,
        "fase1_unicos": 0,
        "fase2_unicos": 0,
        "desambiguados_caixa": 0,
    }

    for rset, label, prio in ((res1.values(), "f1", 1), (res2.values(), "f2", 0)):
        for r in rset:
            if r.get("status") not in ("RESOLVIDO_TESSERACT", "RESOLVIDO_GEMINI"):
                continue
            bib = r.get("bib")
            serie = SERIE_BIB.get(bib)
            if not serie:
                continue
            pagfis = int(r["pagfis"])
            if pagfis < 1:
                continue
            parent = fila2.get(r.get("pasta_id")) if label == "f2" else fila1.get(r["id"])
            cx_parent = parent.get("caixa") if parent else None
            for d in r.get("resolved_docs") or []:
                cands = mapa.get(serie, {}).get(str(int(d)), [])
                uniq_refs = sorted({c["ref"] for c in cands})
                if len(uniq_refs) == 1:
                    ref = uniq_refs[0]
                    usou_cx = False
                else:
                    exact = sorted({c["ref"] for c in cands if c["cx"] == cx_parent and not c["tem_link"]})
                    if len(exact) != 1:
                        stats["ignorados_ambiguos"] += 1
                        continue
                    ref = exact[0]
                    usou_cx = True
                if ref in existentes:
                    stats["ignorados_existentes"] += 1
                    continue
                if ref not in refs_validas:
                    stats["ignorados_fora_parquet"] += 1
                    continue
                candidatos_por_ref[ref].append((prio, bib, pagfis))
                if usou_cx:
                    stats["desambiguados_caixa"] += 1

    novos: dict[str, tuple[str, int]] = {}
    for ref, hits in candidatos_por_ref.items():
        # Menor prio (0 = Fase 2 mais específica, 1 = Fase 1), depois menor pagfis.
        best_prio, bib, pagfis = min(hits, key=lambda x: (x[0], x[2]))
        novos[ref] = (bib, pagfis)
        if best_prio == 0:
            stats["fase2_unicos"] += 1
        else:
            stats["fase1_unicos"] += 1
    return novos, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capilhas-dir", default=str(Path.home() / "resgate_capilhas"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    capilhas_dir = Path(args.capilhas_dir)
    with gzip.open(INDICE, "rt", encoding="utf-8") as f:
        linhas = [linha.rstrip("\n").split("\t") for linha in f if linha.strip()]
    cabecalho, corpo = linhas[0], linhas[1:]
    existentes = {linha[0]: (linha[1], int(linha[2])) for linha in corpo}

    tabela = pq.read_table(DOCS_PARQUET)
    refs_col = tabela["reference_code"].to_pylist()
    old_dr_col = tabela["resgate_docreader"].to_pylist()
    refs_validas = {r for r in refs_col if isinstance(r, str)}

    novos, stats = coletar_novos_links(capilhas_dir, existentes, refs_validas)
    indice_completo = dict(existentes)
    indice_completo.update(novos)

    # Atualiza a coluna resgate_docreader garantindo que valores já preenchidos não mudam.
    novos_dr = []
    alterados_previos = 0
    adicionados_parquet = 0
    for ref, old_val in zip(refs_col, old_dr_col):
        if old_val is not None:
            novos_dr.append(old_val)
            if isinstance(ref, str) and ref in indice_completo:
                esperado = f"{indice_completo[ref][0]}#{indice_completo[ref][1]}"
                if old_val != esperado:
                    alterados_previos += 1
            continue
        if isinstance(ref, str) and ref in novos:
            bib, pf = novos[ref]
            novos_dr.append(f"{bib}#{pf}")
            adicionados_parquet += 1
        else:
            novos_dr.append(None)

    total_com_quadro = sum(1 for v in novos_dr if v is not None)
    relatorio = {
        "novos_no_indice": len(novos),
        "indice_total": len(existentes) + len(novos),
        "linhas_atualizadas_parquet": adicionados_parquet,
        "resgate_com_quadro_antes": sum(1 for v in old_dr_col if v is not None),
        "resgate_com_quadro_depois": total_com_quadro,
        "conflitos_previos": alterados_previos,
        **stats,
    }
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    corpo_atualizado = corpo + [[ref, bib, str(pf)] for ref, (bib, pf) in novos.items()]
    corpo_atualizado.sort()
    buf = io.StringIO()
    for linha in [cabecalho] + corpo_atualizado:
        buf.write("\t".join(linha) + "\n")
    with gzip.GzipFile(INDICE, "wb", mtime=0) as f:
        f.write(buf.getvalue().encode("utf-8"))

    idx_col = tabela.schema.get_field_index("resgate_docreader")
    nova_tabela = tabela.set_column(idx_col, "resgate_docreader", pa.array(novos_dr, type=pa.string()))
    pq.write_table(nova_tabela, DOCS_PARQUET)

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest["gerado_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.setdefault("acertos", {})["resgate_com_quadro"] = total_com_quadro
    manifest["artefatos"]["documentos.parquet"] = sha256_arquivo(DOCS_PARQUET)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
