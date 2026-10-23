"""Extrai as imagens das paginas dos PDFs da Biblioteca Nacional (objdigital/) para paginas/.

Uma pagina com uma so imagem grande (o escaneamento) tem o JPEG copiado byte a byte,
sem recodificar. Uma pagina composta de varias imagens (capas, mapas) e renderizada
em PNG e listada em paginas/_bn_renderizadas.csv. Pode ser rodado de novo: pula o que existe.

Uso (na pasta deste kit):  python extrair_bn.py
"""
from pathlib import Path
import fitz

KIT = Path(__file__).parent
PDFS = {"mss1700651_v1": "pe_v1", "mss1700652_v2": "pe_v2", "mss1700653_v3": "pe_v3",
        "mss1700654_v4": "pe_v4", "mss1069572": "maranhao", "mss1054717": "colonia"}
GRANDE = 500_000  # pixels; smaller images are tiles, logos and rules


def main():
    renderizadas = []
    for pdf, nome in PDFS.items():
        doc = fitz.open(KIT / "objdigital" / f"{pdf}.pdf")
        pasta = KIT / "paginas" / nome
        pasta.mkdir(parents=True, exist_ok=True)
        for n, pagina in enumerate(doc, 1):
            if list(pasta.glob(f"{n:04d}.*")):
                continue
            grandes = [i for i in pagina.get_images(full=True) if i[2] * i[3] > GRANDE]
            filtro = grandes and doc.xref_get_key(grandes[0][0], "Filter")[1]
            if len(grandes) == 1 and filtro.replace(" ", "") in ("/DCTDecode", "[/DCTDecode]"):
                (pasta / f"{n:04d}.jpg").write_bytes(doc.xref_stream_raw(grandes[0][0]))
            else:
                pagina.get_pixmap(dpi=300).save(pasta / f"{n:04d}.png")
                renderizadas.append(f"{nome},{n},{len(grandes)} imagens grandes")
        print("ok", nome, len(doc))
    if renderizadas:
        with open(KIT / "paginas" / "_bn_renderizadas.csv", "a", encoding="utf-8") as f:
            f.write("\n".join(renderizadas) + "\n")
    print("renderizadas:", len(renderizadas))


if __name__ == "__main__":
    main()
