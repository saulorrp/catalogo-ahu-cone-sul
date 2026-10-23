"""Lista, caixa a caixa, os registros do DigitArq de todas as caixas que o corpus cita.

O DigitArq descreve muitos documentos em intervalo (`PT/AHU/CU/017-001/0016/03265.03267`), e o
intervalo dele nem sempre começa onde começa o do catálogo. Este índice deixa o `build/cotas.py`
dar a cada documento o registro do DigitArq que o contém.

Usa a API pública que o próprio site do DigitArq chama (`/api/docs/search`, dez resultados por
página), com quatro pedidos ao mesmo tempo. Pode ser parado e retomado: as caixas já listadas ficam
no arquivo de progresso.

Uso (na raiz do repositório):
  python coleta/scripts/varrer_digitarq.py PROGRESSO.jsonl
  python coleta/scripts/varrer_digitarq.py PROGRESSO.jsonl --gerar corpus/digitarq_referencias.tsv.gz
"""
import argparse, gzip, json, sys, threading, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from ahu.conversor import para_referencia, reconhecer  # noqa: E402

API = "https://digitarq.arquivos.pt/api/docs/search?descriptionLevel=DC&max=10&fromIndex={}&query={}"


def caixas_do_corpus(app_dados):
    d = pd.read_parquet(Path(app_dados) / "documentos.parquet", columns=["reference_code", "new_code"])
    caixas = set()
    for rc, nc in zip(d["reference_code"], d["new_code"]):
        for texto in (nc, rc):
            cota = reconhecer(texto) if isinstance(texto, str) else None
            ref = para_referencia(cota) if cota else None
            if ref:
                caixas.add(ref.rsplit("/", 1)[0])
    return sorted(caixas)


def pede(consulta, inicio):
    url = API.format(inicio, urllib.parse.quote(consulta, safe=""))
    for tentativa in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except OSError:
            time.sleep(5 * (tentativa + 1))
    raise RuntimeError(url)


def lista_caixa(caixa, passadas=8):
    """A paginação da API não tem ordem estável: uma passada pelas páginas repete uns registros e
    pula outros. Por isso a caixa é percorrida de novo até juntar o total que a API anuncia."""
    refs, total = set(), None
    for _ in range(passadas):
        for inicio in range(0, total or 10, 10):
            dados = pede(caixa, inicio)
            total = dados.get("total", 0) if total is None else total
            refs |= {x["referenceCode"]["value"] for x in dados.get("results", [])}
            time.sleep(0.1)
        if len(refs) >= total:
            break
    return total, sorted(r for r in refs if r.startswith(caixa + "/"))


def varrer(progresso, app_dados):
    caixas = caixas_do_corpus(app_dados)
    feitas = set()
    if progresso.exists():
        linhas = [json.loads(linha) for linha in progresso.open(encoding="utf-8")]
        feitas = {j["caixa"] for j in linhas if len(j["refs"]) >= j["total"]}
    faltam = [c for c in caixas if c not in feitas]
    print(f"{len(caixas)} caixas, {len(faltam)} por listar", flush=True)
    trava = threading.Lock()

    def tarefa(caixa):
        try:
            total, refs = lista_caixa(caixa)
        except RuntimeError as e:
            print("ERRO", caixa, e, flush=True)
            return
        with trava, progresso.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"caixa": caixa, "total": total, "refs": refs}) + "\n")

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(tarefa, faltam))


def gerar(progresso, saida):
    refs = set()
    for linha in progresso.open(encoding="utf-8"):
        refs.update(json.loads(linha)["refs"])
    with gzip.open(saida, "wt", encoding="utf-8") as f:
        f.write(f"# Registros do DigitArq (PT/AHU/CU), listados caixa a caixa em {date.today()}\n")
        f.write("# por coleta/scripts/varrer_digitarq.py\n")
        for ref in sorted(refs):
            f.write(ref + "\n")
    print(f"{len(refs)} registros em {saida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("progresso")
    ap.add_argument("--app-dados", default=str(RAIZ / "app_dados"))
    ap.add_argument("--gerar", help="em vez de varrer, grava o índice .tsv.gz a partir do progresso")
    a = ap.parse_args()
    if a.gerar:
        gerar(Path(a.progresso), Path(a.gerar))
    else:
        varrer(Path(a.progresso), a.app_dados)


if __name__ == "__main__":
    main()
