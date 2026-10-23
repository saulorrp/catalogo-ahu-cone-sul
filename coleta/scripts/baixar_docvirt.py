"""Download page images of Projeto Resgate catalogs from docvirt (library crv_guiafontes)."""
import json, sys, time
from pathlib import Path
import requests
OUT = Path(sys.argv[1])
PASTAS = {86: "bahia_v1", 87: "bahia_v2", 75: "sao_paulo_v1", 76: "sao_paulo_v2",
          91: "minas_v1", 92: "minas_v2", 93: "minas_v3", 79: "para_v1", 80: "para_v2", 81: "para_v3"}
d = json.loads(json.load(open(sys.argv[2]))["data"])
paginas = {}
def walk(L):
    for n in L:
        paginas[n["Id"]] = n["Pages"]
        if n.get("SubFolders"): walk(n["SubFolders"])
walk(d)
S = requests.Session(); S.headers["Authorization"] = "none"
for fid, nome in PASTAS.items():
    pasta = OUT / f"{fid}_{nome}"; pasta.mkdir(parents=True, exist_ok=True)
    for p in range(1, paginas[fid] + 1):
        f = pasta / f"{p:04d}.webp"
        if f.exists() and f.stat().st_size > 1000: continue
        for t in range(5):
            try:
                r = S.get(f"https://api.docvirt.com/v1/documents/crv_guiafontes/{fid}/{p}", timeout=90)
                if r.status_code == 200 and r.content[:4] == b"RIFF":
                    f.write_bytes(r.content); break
            except Exception as e:
                print("erro", fid, p, e, flush=True)
            time.sleep(2 ** t)
        time.sleep(0.3)
    print("ok", fid, nome, paginas[fid], flush=True)
