"""propor_tipologia_para.py — proposta de tipologia para os verbetes dos Códices com tipologia "PARA".

Nos Códices de registro de correspondência expedida (ex.: Cód. 584), o resumo começa por "Para o
governador de Pernambuco, ...", e a extração da tipologia (primeira palavra em maiúsculas) ficou
com "PARA". Este script pede ao Gemini, pelo `gemini-ask` (sem custo em execução, R1), uma
tipologia para cada verbete a partir do resumo, e grava a proposta **à parte**:

    corpus/correcoes/tipologia_para_gemini.parquet
        doc_uid, tipologia_atual, tipologia_proposta, modelo, data

Nada é sobrescrito: a camada tipologia_v1 e o app continuam como estão até o autor revisar a
proposta e decidir como (e se) ela entra.

Uso: python build/propor_tipologia_para.py [--lote 40] [--modelo gemini-3.8-flash-high] [--limite N]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "corpus" / "correcoes" / "tipologia_para_gemini.parquet"

INSTRUCAO = """Você classifica verbetes de um catálogo de documentos do Conselho Ultramarino (séc. XVII-XIX).
Cada verbete abaixo é o resumo de um documento registrado num códice de correspondência expedida;
o resumo começa por "Para <destinatário>", então o tipo do documento não está escrito no início.
Para cada verbete, dê a tipologia documental mais provável em MAIÚSCULAS, uma palavra ou
expressão curta da diplomática luso-brasileira (ex.: CARTA, ORDEM, PROVISÃO, AVISO, OFÍCIO,
CONSULTA, CARTA RÉGIA, DESPACHO). Se não der para saber, responda INDETERMINADA.
NÃO use ferramentas. Responda SÓ com um objeto JSON {"<id>": "<TIPOLOGIA>", ...}, com todos os ids."""


def perguntar(lote: pd.DataFrame, modelo: str) -> dict[str, str]:
    linhas = "\n".join(f"{uid}\t{texto}" for uid, texto in zip(lote["doc_uid"], lote["description"]))
    r = subprocess.run(["gemini-ask", INSTRUCAO, "--stdin", "--json", "--modelo", modelo,
                        "--timeout", "20m"], input=linhas, capture_output=True, text=True)
    saida = json.loads(r.stdout or "{}")
    if not saida.get("ok"):
        sys.exit(f"gemini-ask falhou: {json.dumps(saida, ensure_ascii=False)[:500]}")
    m = re.search(r"\{.*\}", saida["texto"], re.S)
    return json.loads(m.group(0)) if m else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lote", type=int, default=40)
    ap.add_argument("--modelo", default="gemini-3.8-flash-high")
    ap.add_argument("--limite", type=int, default=None)
    a = ap.parse_args()

    docs = pd.read_parquet(RAIZ / "app_dados" / "documentos.parquet",
                           columns=["doc_uid", "tipologia", "description"])
    alvo = docs[docs["tipologia"] == "PARA"].reset_index(drop=True)
    if a.limite:
        alvo = alvo.head(a.limite)
    feitas = pd.read_parquet(SAIDA) if SAIDA.exists() else pd.DataFrame(columns=["doc_uid"])
    falta = alvo[~alvo["doc_uid"].isin(set(feitas["doc_uid"]))]   # retomada
    # Ordem embaralhada (semente fixa): um resultado parcial já cobre os dois códices (584 e 610).
    falta = falta.sample(frac=1, random_state=0)
    novas = []
    for i in range(0, len(falta), a.lote):
        lote = falta.iloc[i:i + a.lote]
        respostas = perguntar(lote, a.modelo)
        for uid, atual in zip(lote["doc_uid"], lote["tipologia"]):
            proposta = str(respostas.get(uid) or "").strip().upper() or None
            novas.append({"doc_uid": uid, "tipologia_atual": atual, "tipologia_proposta": proposta,
                          "modelo": a.modelo, "data": date.today().isoformat()})
        # Grava a cada lote: uma chamada que trave não perde o que já veio (retomada).
        resultado = pd.concat([feitas, pd.DataFrame(novas)], ignore_index=True)
        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_parquet(SAIDA, index=False)
        print(f"{min(i + a.lote, len(falta))}/{len(falta)}", file=sys.stderr, flush=True)
    resultado = pd.read_parquet(SAIDA)
    print(resultado["tipologia_proposta"].value_counts(dropna=False).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
