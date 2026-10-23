"""Extract a PDF's text layer with page markers: python pdf_para_texto.py IN.pdf OUT.txt"""
import sys, pymupdf
d = pymupdf.open(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i, p in enumerate(d, 1):
        f.write(f"=== PAGINA {i} ===\n{p.get_text()}\n")
