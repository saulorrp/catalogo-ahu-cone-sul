"""Estoque de Tesseract: le com o Tesseract (por) as paginas ainda sem transcricao, para que a
rota agy (imagem + texto do Tesseract -> correcao) nao precise esperar por ele.

Grava tesseract/<volume>/NNNN.txt (o mesmo lugar e formato de corrigir_tesseract.py, que reusa
o que achar). Nao e transcricao final. Prioridade: PDFs da BN, depois o docvirt alem do
orcamento do Batch, depois o resto. Pode ser parado e retomado.

Uso (na pasta deste kit):  python tesseract_estoque.py [--processos 6]
"""
import argparse, io, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import ocr_comum as c

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CRU = c.KIT / "tesseract"


def alvos():
    _, previstas, *_ = c.horizonte()
    do_batch = set(previstas)
    bn = [(v, p) for v in c.BN for p in c.paginas(v)]
    docvirt = [(v, p) for v in c.DOCVIRT for p in c.paginas(v)]
    ordem = bn + [x for x in docvirt if c.nome(*x) not in do_batch] + [x for x in docvirt if c.nome(*x) in do_batch]
    return [(v, p) for v, p in ordem
            if not c.feito(v, p) and not (CRU / v / f"{p:04d}.txt").exists()]


def le(alvo):
    volume, pag = alvo
    from PIL import Image
    buf = io.BytesIO()
    Image.open(c.imagem(volume, pag)).save(buf, "PNG")
    # one thread per tesseract: the pool gives the parallelism
    proc = subprocess.run([TESSERACT, "stdin", "stdout", "--tessdata-dir", str(c.KIT / "tessdata"),
                           "-l", "por", "--psm", "1"], input=buf.getvalue(), capture_output=True,
                          env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    if proc.returncode != 0 or not proc.stdout.strip():
        return f"ERRO {volume} {pag}: {proc.stderr[:200]!r}"
    destino = CRU / volume / f"{pag:04d}.txt"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    tmp.write_bytes(proc.stdout)
    os.replace(tmp, destino)  # never a half-written file for the agy route
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processos", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    lista = alvos()
    print(time.strftime("%H:%M"), f"{len(lista)} paginas, {args.processos} processos", flush=True)
    t0, feitas = time.time(), 0
    with ProcessPoolExecutor(args.processos) as pool:
        for erro in pool.map(le, lista, chunksize=4):
            feitas += 1
            if erro:
                print(erro, flush=True)
            if feitas % 100 == 0:
                ritmo = feitas / ((time.time() - t0) / 60)
                print(time.strftime("%H:%M"), f"{feitas}/{len(lista)}, {ritmo:.1f} pag/min, "
                      f"faltam ~{(len(lista) - feitas) / ritmo:.0f} min", flush=True)
    print(time.strftime("%H:%M"), "fim", feitas, flush=True)


if __name__ == "__main__":
    sys.exit(main())
