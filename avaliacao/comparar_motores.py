"""comparar_motores.py — léxico, híbrido e híbrido + reranker nas consultas de ouro.

Enquanto o julgamento humano (`julgar.py`) não existe, os candidatos são julgados por um modelo
de linguagem (Gemini, pela API), às cegas: ele vê a consulta e o resumo, não
o motor nem a posição. As notas ficam em `avaliacao/julgamentos_llm.json`, **separadas** das do
autor, e servem para comparar motores entre si — não substituem o julgamento do autor no artigo.

    python avaliacao/comparar_motores.py rodar      # gera as listas de cada motor
    python avaliacao/comparar_motores.py julgar     # julga o que falta (Gemini, retomável)
    python avaliacao/comparar_motores.py medir      # nDCG@10, Recall@20, MRR por motor

O julgamento usa a escala do `julgar.py`: 0 irrelevante, 1 parcialmente relevante, 2 relevante.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from avaliacao.avaliar_busca import agregar_metricas, calcular_metricas_consulta  # noqa: E402

PASTA = RAIZ / "avaliacao"
CONSULTAS = PASTA / "consultas_ouro.json"
LISTAS = PASTA / "listas_motores.json"
JULGAMENTOS = PASTA / "julgamentos_llm.json"
PROFUNDIDADE = 20      # quantos de cada motor entram no conjunto a julgar
LOTE_JULGAMENTO = 20
# Pela API, com a chave de estudante. Cada modelo tem cota diária própria: esgotada a de um (429),
# passa-se ao seguinte. O modelo que deu cada nota fica registrado no julgamento.
MODELOS_JUIZ = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
                "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

# nome -> (modo, reranker: (modelo, arquivo, limite) ou None)
MOTORES = {
    "lexico": ("exata", None),
    "hibrido": ("automatico", None),
    "hibrido+mmarco30": ("automatico", ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                                        "onnx/model_quint8_avx2.onnx", 30)),
    "hibrido+mmarco20": ("automatico", ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                                        "onnx/model_quint8_avx2.onnx", 20)),
    "hibrido+mmarco50": ("automatico", ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                                        "onnx/model_quint8_avx2.onnx", 50)),
    "hibrido+bgem3_30": ("automatico", ("onnx-community/bge-reranker-v2-m3-ONNX",
                                        "onnx/model_quantized.onnx", 30)),
}


def _consultas(com_cota: bool = False) -> list[dict]:
    todas = json.loads(CONSULTAS.read_text(encoding="utf-8"))
    return [q for q in todas if com_cota or q["tipo"] != "cota"]


def rodar(motores: list[str]) -> None:
    from ahu import busca, consulta_onnx, reranker
    from ahu.dados import carregar_sem_cache

    dados = carregar_sem_cache()
    encoder = consulta_onnx.criar_encoder_de_dados(dados)
    listas = json.loads(LISTAS.read_text(encoding="utf-8")) if LISTAS.exists() else {}
    for nome in motores:
        modo, cfg = MOTORES[nome]
        rr = None
        if cfg:
            rr = reranker.criar_reranker(cfg[0], cfg[1])
            busca.LIMITE_RERANKER = cfg[2]
        tempos = []
        for q in _consultas(com_cota=True):
            t = time.perf_counter()
            resp = busca.buscar(dados, q["consulta"], modo=modo,
                                encoder=None if modo == "exata" else encoder, reranker=rr)
            tempos.append(time.perf_counter() - t)
            listas.setdefault(nome, {})[q["id"]] = [r.doc_uid for r in resp.resultados[:50]]
        tempos.sort()
        print(f"{nome}: mediana {tempos[len(tempos) // 2]:.2f} s, pior {tempos[-1]:.2f} s",
              flush=True)
        listas.setdefault("_tempos", {})[nome] = {"mediana": tempos[len(tempos) // 2],
                                                  "pior": tempos[-1]}
        LISTAS.write_text(json.dumps(listas, ensure_ascii=False, indent=1), encoding="utf-8")


def _prompt(consulta: str, itens: list[tuple[str, str]]) -> str:
    linhas = "\n".join(f"[{i}] {texto}" for i, (_, texto) in enumerate(itens, start=1))
    return (
        "Você avalia um motor de busca sobre o catálogo do Arquivo Histórico Ultramarino "
        "(documentos coloniais luso-brasileiros, séc. XVII–XIX). Para a consulta de um "
        "pesquisador, julgue cada verbete do catálogo:\n"
        "2 = relevante (o documento trata do que se pergunta; para nome de pessoa, o documento "
        "é dessa pessoa ou sobre ela; para lugar, o documento trata desse lugar),\n"
        "1 = parcialmente relevante (toca o assunto de passagem ou de modo indireto),\n"
        "0 = irrelevante.\n"
        "Atenção a homônimos e grafias antigas (ex.: Brito/Britto, Paes/Pais).\n\n"
        f"CONSULTA: {consulta}\n\nVERBETES:\n{linhas}\n\n"
        'Responda SÓ com um objeto JSON {"1": nota, "2": nota, ...}, uma nota por verbete.'
    )


def _chave_gemini() -> str:
    """A chave da API do Gemini: `GEMINI_API_KEY_ESTUDANTE`, do ambiente ou do chaves.env do servidor."""
    import os
    if os.environ.get("GEMINI_API_KEY_ESTUDANTE"):
        return os.environ["GEMINI_API_KEY_ESTUDANTE"]
    arquivo = Path.home() / ".config" / "agiulfo-tools" / "chaves.env"
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        if linha.startswith("GEMINI_API_KEY_ESTUDANTE="):
            return linha.split("=", 1)[1].strip().strip("\r").strip('"')
    raise RuntimeError("GEMINI_API_KEY_ESTUDANTE não encontrada")


_esgotados: set[str] = set()


def _perguntar(prompt: str) -> tuple[dict[str, int], str]:
    for modelo in MODELOS_JUIZ:
        if modelo in _esgotados:
            continue
        notas = _perguntar_a(modelo, prompt)
        if notas is not None:
            return notas, modelo
        _esgotados.add(modelo)
        print(f"  {modelo} esgotado; passando ao seguinte", flush=True)
    raise RuntimeError("todos os modelos do Gemini esgotaram a cota")


def _perguntar_a(modelo: str, prompt: str) -> dict[str, int] | None:
    import urllib.error
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
    corpo = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"responseMimeType": "application/json",
                                             "temperature": 0}}).encode()
    for tentativa in range(4):
        try:
            pedido = urllib.request.Request(url, corpo, {"Content-Type": "application/json",
                                                         "x-goog-api-key": _chave_gemini()})
            with urllib.request.urlopen(pedido, timeout=300) as r:
                resposta = json.load(r)
            texto = "".join(p.get("text", "") for p in
                            resposta["candidates"][0]["content"]["parts"])
            inicio, fim = texto.index("{"), texto.rindex("}") + 1
            return {k: int(v) for k, v in json.loads(texto[inicio:fim]).items()}
        except urllib.error.HTTPError as e:
            print(f"  {modelo}, tentativa {tentativa + 1}: {e}", flush=True)
            if e.code == 429 and tentativa >= 1:
                return None      # cota do dia, não do minuto
            time.sleep(30 * (tentativa + 1))
        except Exception as e:      # rede ou resposta fora do formato: tenta de novo
            print(f"  {modelo}, tentativa {tentativa + 1}: {str(e)[:200]}", flush=True)
            time.sleep(10)
    return None


def julgar() -> None:
    from ahu.dados import carregar_sem_cache
    from ahu.reranker import texto_do_documento

    dados = carregar_sem_cache()
    docs = dados.documentos.set_index("doc_uid")
    listas = json.loads(LISTAS.read_text(encoding="utf-8"))
    julg = json.loads(JULGAMENTOS.read_text(encoding="utf-8")) if JULGAMENTOS.exists() else {}
    for q in _consultas():
        pool = []
        for nome, por_q in listas.items():
            if nome.startswith("_"):
                continue
            pool.extend(por_q.get(q["id"], [])[:PROFUNDIDADE])
        feitos = julg.setdefault(q["id"], {"consulta": q["consulta"], "notas": {}})["notas"]
        faltam = sorted(set(pool) - set(feitos))
        random.Random(q["id"]).shuffle(faltam)    # às cegas: nem motor, nem posição
        for inicio in range(0, len(faltam), LOTE_JULGAMENTO):
            lote = faltam[inicio: inicio + LOTE_JULGAMENTO]
            itens = [(uid, texto_do_documento(docs.loc[uid])) for uid in lote]
            notas, modelo = _perguntar(_prompt(q["consulta"], itens))
            juizes = julg[q["id"]].setdefault("modelos", {})
            for i, uid in enumerate(lote, start=1):
                if str(i) in notas:
                    feitos[uid] = max(0, min(2, notas[str(i)]))
                    juizes[modelo] = juizes.get(modelo, 0) + 1
            JULGAMENTOS.write_text(json.dumps(julg, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
            print(f"{q['id']}: {len(feitos)} julgados", flush=True)


def medir() -> None:
    listas = json.loads(LISTAS.read_text(encoding="utf-8"))
    julg = json.loads(JULGAMENTOS.read_text(encoding="utf-8"))
    print("| motor | nDCG@10 | Recall@20 | MRR | tema | nome | lugar | mediana (s) | pior (s) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for nome, por_q in listas.items():
        if nome.startswith("_"):
            continue
        itens = []
        for q in _consultas():
            notas = julg.get(q["id"], {}).get("notas", {})
            if notas:
                itens.append({"tipo": q["tipo"],
                              **calcular_metricas_consulta(por_q[q["id"]], notas)})
        m = agregar_metricas(itens)
        g, t = m["global"], listas.get("_tempos", {}).get(nome, {})
        tipos = " | ".join(f"{m['por_tipo'].get(k, {}).get('ndcg_10', 0):.3f}"
                           for k in ("tema", "nome", "lugar"))
        print(f"| {nome} | {g['ndcg_10']:.3f} | {g['recall_20']:.3f} | {g['mrr']:.3f} | {tipos} | "
              f"{t.get('mediana', 0):.2f} | {t.get('pior', 0):.2f} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("acao", choices=("rodar", "julgar", "medir"))
    p.add_argument("--motores", nargs="*", default=list(MOTORES))
    args = p.parse_args()
    {"rodar": lambda: rodar(args.motores), "julgar": julgar, "medir": medir}[args.acao]()


if __name__ == "__main__":
    main()
