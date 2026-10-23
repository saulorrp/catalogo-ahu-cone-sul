"""Transcreve as paginas do docvirt pela Batch API do Gemini (metade do preco), ate o orcamento.

Roda em laco: a cada 10 min recolhe os lotes prontos (grava ocr_api/<volume>/NNNN.txt e o
custo real) e envia um lote novo com as paginas ja baixadas, na ordem de DOCVIRT, enquanto o
total gasto + o estimado dos lotes em voo couber em ORCAMENTO (ocr_comum.py). Uma pagina
recusada (RECITATION, vazia) e reenviada uma vez; na segunda vai para ocr_api/_bloqueadas.csv.
Estado em ocr_api/_lotes.json; pode ser parado e retomado.

Uso (na pasta deste kit, com GEMINI_API_KEY):  python ocr_batch.py [--uma-vez]
"""
import argparse, base64, json, sys, time, urllib.error
from datetime import datetime
import ocr_comum as c

LOTE_MAX = 300           # pages per batch (about 160 MB of JSONL)
LOTE_MIN = 100           # wait for this many new pages while the download runs
INTERVALO = 600          # seconds between rounds
PAUSA = c.OCR_API / "_pausa_custo.txt"


def agora():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def requisicao(volume, pag, texto=None):
    img = c.imagem(volume, pag)
    return {"contents": [{"parts": [
                {"inline_data": {"mime_type": c.MIME[img.suffix], "data": base64.b64encode(img.read_bytes()).decode()}},
                {"text": texto or c.prompt_api()}]}],
            "generation_config": {"temperature": 0, "thinking_config": {"thinking_level": "low"}}}


def envia(chaves, estado):
    """Upload one batch. A page refused before goes as image + Tesseract text (the correction
    prompt, which passes RECITATION); its result lands in ocr_corrigido. Returns the keys sent
    as corrections too."""
    import corrigir_tesseract as ct
    from concurrent.futures import ThreadPoolExecutor
    # Tesseract for the pages that go as corrections, 4 at a time (one by one took ~30 min for 70 pages)
    refazer = [k for k in chaves if estado["tentativas"].get(k, 0) >= 1]
    print(agora(), f"preparando lote: Tesseract de {len(refazer)} paginas recusadas", flush=True)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda k: ct.tesseract(k.split("/")[0], int(k.split("/")[1])), refazer))
    linhas, correcoes = [], []
    for k in chaves:
        volume, pag = k.split("/")[0], int(k.split("/")[1])
        if estado["tentativas"].get(k, 0) >= 1:
            cru = ct.tesseract(volume, pag)  # reuses the stock
            if ct.quase_branca(cru):
                destino = c.KIT / "ocr_corrigido" / volume / f"{pag:04d}.txt"
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_text("# SEM VERBETES\n", encoding="utf-8")
                continue
            linhas.append(json.dumps({"key": k, "request": requisicao(volume, pag, ct.pedido_texto(cru))}))
            correcoes.append(k)
        else:
            linhas.append(json.dumps({"key": k, "request": requisicao(volume, pag)}))
    if not linhas:
        return None, None, 0, []
    dados = ("\n".join(linhas) + "\n").encode()
    _, h = c.pede("https://generativelanguage.googleapis.com/upload/v1beta/files", bruto=True,
                  corpo={"file": {"display_name": f"resgate-{int(time.time())}"}},
                  cabecalhos={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                              "X-Goog-Upload-Header-Content-Length": str(len(dados)),
                              "X-Goog-Upload-Header-Content-Type": "application/jsonl"})
    arquivo = c.pede(h["X-Goog-Upload-URL"], dados=dados, metodo="POST",
                     cabecalhos={"X-Goog-Upload-Command": "upload, finalize", "X-Goog-Upload-Offset": "0",
                                 "Content-Type": "application/jsonl"})["file"]["name"]
    op = c.pede(f"{c.API}/models/{c.MODELO_API}:batchGenerateContent",
                corpo={"batch": {"display_name": f"resgate-{chaves[0]}", "input_config": {"file_name": arquivo}}})
    return op["name"], arquivo, len(dados), correcoes


def acha(obj, chave):
    """First value of `chave` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if chave in obj:
            return obj[chave]
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            r = acha(v, chave)
            if r is not None:
                return r
    return None


def recolhe(lote, estado):
    info = c.pede(f"{c.API}/{lote['nome']}")
    situacao = acha(info, "state") or ""
    lote["estado"] = situacao
    if situacao in ("BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED", "JOB_STATE_FAILED"):
        lote["liberado"] = True  # its pages go back to the queue
        print(agora(), "lote", lote["nome"], situacao, "- paginas voltam para a fila", flush=True)
        return
    if "SUCCEEDED" not in situacao:
        return
    arquivo = acha(info, "responsesFile")
    conteudo, _ = c.pede(f"https://generativelanguage.googleapis.com/download/v1beta/{arquivo}:download?alt=media",
                         bruto=True)
    custo, ok, recusadas = 0.0, 0, []
    for linha in conteudo.decode("utf-8").splitlines():
        if not linha.strip():
            continue
        r = json.loads(linha)
        chave = r.get("key")
        resp = r.get("response") or {}
        uso = resp.get("usageMetadata", {})
        custo += c.custo(uso, "batch")
        cand = (resp.get("candidates") or [{}])[0]
        texto = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        volume, pag = chave.split("/")[0], int(chave.split("/")[1])
        suspeita = texto.strip() and c.suspeita(volume, pag, texto)
        if suspeita:
            print(agora(), "SUSPEITA", chave, suspeita, "- nao gravada", flush=True)
            texto = ""
        if texto.strip():
            pasta = c.KIT / "ocr_corrigido" if chave in lote.get("correcoes", []) else c.OCR_API
            saida = pasta / volume / f"{pag:04d}.txt"
            saida.parent.mkdir(parents=True, exist_ok=True)
            saida.write_text(c.limpa(texto), encoding="utf-8")
            ok += 1
            continue
        motivo = cand.get("finishReason") or (r.get("error") or {}).get("message", "vazia")
        n = estado["tentativas"].get(chave, 0) + 1
        estado["tentativas"][chave] = n
        if n >= 2:
            c.bloqueia(volume, pag, "batch", motivo)
        recusadas.append(f"{chave}:{motivo}")
    lote["custo_real"] = round(custo, 4)
    lote["liberado"] = True
    # no cap any more (auto-reload), only a sanity guard: ~3x the normal cost per page stops new batches
    if lote["n"] >= 20 and custo / lote["n"] > 0.012:
        PAUSA.write_text(f"{agora()} lote {lote['nome']}: US$ {custo / lote['n']:.4f}/pagina\n", encoding="utf-8")
        print(agora(), "PAUSA: custo por pagina anormal; apague ocr_api/_pausa_custo.txt para retomar", flush=True)
    print(agora(), "lote", lote["nome"], f"ok {ok}/{lote['n']}, US$ {custo:.3f}",
          ("recusadas: " + ", ".join(recusadas)) if recusadas else "", flush=True)


def rodada(estado):
    for lote in estado["lotes"]:
        if not lote.get("liberado"):
            try:
                recolhe(lote, estado)
            except (urllib.error.URLError, OSError) as e:
                print(agora(), "erro ao consultar", lote["nome"], e, flush=True)
            c.grava_lotes(estado)
    falhos = [l for l in estado["lotes"][-3:] if l.get("estado", "").endswith(("FAILED", "EXPIRED", "CANCELLED"))]
    if len(falhos) == 3 and not PAUSA.exists():
        PAUSA.write_text(f"{agora()} tres lotes seguidos falharam\n", encoding="utf-8")
        print(agora(), "PAUSA: tres lotes seguidos falharam; apague ocr_api/_pausa_custo.txt para retomar", flush=True)
    if PAUSA.exists():
        print(agora(), "em PAUSA (ocr_api/_pausa_custo.txt):", PAUSA.read_text(encoding="utf-8").strip(), flush=True)
        return True
    em_voo, previstas, gasto, pendente, por_pagina = c.horizonte(estado)
    prontas = [k for k in previstas
               if c.imagem(k.split("/")[0], int(k.split("/")[1]))
               and not (c.OCR_AGY / f"{k}.lock").exists()][:LOTE_MAX]
    baixando = any(len(c.paginas(v)) < t for v, t in c.DOCVIRT.items())
    print(agora(), f"gasto US$ {gasto:.2f}, em voo {len(em_voo)} pag. (~US$ {pendente:.2f}), "
          f"cabem mais {len(previstas)}, prontas {len(prontas)}", flush=True)
    if prontas and (len(prontas) >= LOTE_MIN or not baixando or len(prontas) == len(previstas)):
        try:
            nome, arquivo, tamanho, correcoes = envia(prontas, estado)
        except (urllib.error.URLError, OSError, KeyError) as e:
            print(agora(), "erro ao enviar lote", e, flush=True)
            return True
        if nome:
            enviadas = [k for k in prontas if not c.feito(k.split("/")[0], int(k.split("/")[1]))]
            estado["lotes"].append({"nome": nome, "arquivo": arquivo, "n": len(enviadas), "chaves": enviadas,
                                    "correcoes": correcoes, "enviado": agora(), "bytes": tamanho})
            c.grava_lotes(estado)
            print(agora(), "enviado", nome, len(enviadas), "paginas,", f"{len(correcoes)} como correcao,",
                  prontas[0], "a", prontas[-1], flush=True)
    return bool(em_voo or previstas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uma-vez", action="store_true")
    args = ap.parse_args()
    c.OCR_API.mkdir(exist_ok=True)
    while True:
        estado = c.le_lotes()
        continua = rodada(estado)
        if args.uma_vez or not continua:
            print(agora(), "fim: orcamento esgotado ou docvirt completo" if not continua else "", flush=True)
            return
        time.sleep(INTERVALO)


if __name__ == "__main__":
    sys.exit(main())
