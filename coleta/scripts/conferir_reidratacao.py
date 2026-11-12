"""Verifica a reidratação de resumos a partir do corpus atual.

Uso: python conferir_reidratacao.py RAIZ_DO_REPO PASTA_PDFS [--saida pasta]
"""
import sys, json, hashlib, tempfile, argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from reidratar_resumos import reidratar

def main():
    parser = argparse.ArgumentParser(description="Confere reidratação no corpus.")
    parser.add_argument("raiz_repo", help="Raiz do repositório")
    parser.add_argument("pasta_pdfs", help="Pasta de PDFs")
    parser.add_argument("--saida", default=None, help="Pasta de saída")
    args = parser.parse_args()

    raiz = Path(args.raiz_repo)
    pasta_saida = Path(args.saida) if args.saida else Path(tempfile.mkdtemp())
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arquivos_parquet = list(raiz.glob("corpus/colecoes/*/documentos.parquet"))
    linhas = []
    pulas_sem_desc = 0

    for f in arquivos_parquet:
        df = pd.read_parquet(f)
        if "url_actd" not in df.columns:
            continue
        df_sub = df[df["url_actd"].notna() & (df["url_actd"] != "")]
        for _, row in df_sub.iterrows():
            desc = row.get("description")
            if pd.isna(desc) or not str(desc).strip():
                pulas_sem_desc += 1
                continue
            desc_str = str(desc).strip()
            h = hashlib.sha1(desc_str.encode("utf-8")).hexdigest()
            rec = {
                "doc_uid": row.get("doc_uid"),
                "url_actd": row.get("url_actd"),
                "pagina_actd": row.get("pagina_actd"),
                "sha1_resumo": h,
                "description": desc_str
            }
            linhas.append(rec)

    entrada_df = pd.DataFrame(linhas)
    entrada_path = pasta_saida / "entrada.parquet"
    entrada_df.to_parquet(entrada_path)

    df_saida, df_falhas, resumo = reidratar(entrada_df, args.pasta_pdfs)
    resumo["pulas_sem_desc"] = pulas_sem_desc

    df_saida.to_parquet(pasta_saida / "saida.parquet")
    df_falhas.to_csv(pasta_saida / "falhas.csv", index=False, encoding="utf-8")
    with open(pasta_saida / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"=== CONFERÊNCIA CONCLUÍDA ===")
    print(f"Pulas sem descrição: {pulas_sem_desc}")
    print(f"Total entradas: {resumo['total']}")
    print(f"Reidratados: {resumo['reidratados']} ({resumo['taxa_pct']}%)")
    print(f"Por nível: {resumo['por_nivel']}")
    print(f"Falhas por motivo: {resumo['falhas_por_motivo']}")
    print(f"Saída salva em: {pasta_saida}")

if __name__ == "__main__":
    main()
