"""Baixa as imagens das paginas dos catalogos do Projeto Resgate hospedados no docvirt.

Uso (na pasta deste kit):  python baixar_paginas.py
Cria paginas/<id>_<nome>/0001.webp ... Pode ser interrompido e retomado.
So usa a biblioteca padrao do Python.
"""
import time, urllib.request
from pathlib import Path

# id da pasta no docvirt (biblioteca crv_guiafontes) -> (nome, numero de paginas)
VOLUMES = {
    86: ("bahia_v1", 708), 87: ("bahia_v2", 478),
    75: ("sao_paulo_v1", 311), 76: ("sao_paulo_v2", 804),
    91: ("minas_v1", 358), 92: ("minas_v2", 374), 93: ("minas_v3", 360),
    79: ("para_v1", 451), 80: ("para_v2", 490), 81: ("para_v3", 262),
    # 88 (ultramar_brasil_geral) dropped 2026-10-22: its catalogue exists as a text PDF at actd.iict.pt
    84: ("codices_cu", 179), 89: ("cartografia", 429),
}
URL = "https://api.docvirt.com/v1/documents/crv_guiafontes/{fid}/{pag}"
BASE = Path(__file__).parent / "paginas"

def baixa(fid, pag, destino):
    req = urllib.request.Request(URL.format(fid=fid, pag=pag), headers={"Authorization": "none", "User-Agent": "Mozilla/5.0"})
    for tentativa in range(5):
        try:
            dados = urllib.request.urlopen(req, timeout=90).read()
            if dados[:4] == b"RIFF":
                destino.write_bytes(dados); return True
        except Exception as e:
            print("  erro", fid, pag, e)
        time.sleep(2 ** tentativa)
    return False

for fid, (nome, total) in VOLUMES.items():
    pasta = BASE / f"{fid}_{nome}"
    pasta.mkdir(parents=True, exist_ok=True)
    for pag in range(1, total + 1):
        f = pasta / f"{pag:04d}.webp"
        if f.exists() and f.stat().st_size > 1000: continue
        if not baixa(fid, pag, f): print("FALHOU", nome, pag)
        time.sleep(0.2)
    print("ok", nome, total)
