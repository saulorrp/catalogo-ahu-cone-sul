"""Gera arquivo TSV com checksums SHA-256 das fontes PDF do ACTD.

Uso: python checksums_fontes.py PASTA_PDFS SERIES_TSV SAIDA_TSV --data-acesso AAAA-MM-DD
"""
import sys, hashlib, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reidratar_resumos import nome_local

def main():
    parser = argparse.ArgumentParser(description="Gera TSV com checksums SHA-256 dos PDFs.")
    parser.add_argument("pasta_pdfs", help="Pasta contendo os arquivos PDF")
    parser.add_argument("series_tsv", help="Caminho de series_actd.tsv")
    parser.add_argument("saida_tsv", help="Caminho do arquivo TSV de saída")
    parser.add_argument("--data-acesso", required=True, help="Data de acesso no formato AAAA-MM-DD")
    args = parser.parse_args()

    pasta = Path(args.pasta_pdfs)
    linhas_out = ["url\tdata_acesso\ttamanho_bytes\tsha256"]

    with open(args.series_tsv, encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            parts = l.split("\t")
            if len(parts) < 3:
                continue
            caminho_rel = parts[2].strip()
            url = "https://actd.iict.pt/" + caminho_rel
            local_name = nome_local(url)
            pdf_path = pasta / local_name

            if pdf_path.is_file():
                tamanho = str(pdf_path.stat().st_size)
                sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            else:
                tamanho = ""
                sha256 = ""
                print(f"AVISO: PDF não encontrado: {local_name}", file=sys.stderr)

            linhas_out.append(f"{url}\t{args.data_acesso}\t{tamanho}\t{sha256}")

    with open(args.saida_tsv, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_out) + "\n")

if __name__ == "__main__":
    main()
