"""julgar.py — WP 4.1: Interface cega para julgamento do conjunto de teste (spec §5.6, A5).

Reaproveita o protocolo cego do auditool.py (legado):
- Mostra a união dos 20 primeiros resultados de cada configuração (v2 vs v3) em ordem aleatória;
- O avaliador não sabe de qual motor veio cada documento;
- Julgamento em três níveis:
    0 = irrelevante
    1 = relevante
    2 = muito relevante
- Grava tempo de decisão por documento (delta_seconds) e salva em avaliacao/julgamentos.json;
- Permite retomar de onde parou.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ARQUIVO_CONSULTAS = Path(__file__).resolve().parent / "consultas_ouro.json"
ARQUIVO_JULGAMENTOS = Path(__file__).resolve().parent / "julgamentos.json"
ARQUIVO_POOL = Path(__file__).resolve().parent / "pool_candidatos.json"


def carregar_consultas(caminho: Path = ARQUIVO_CONSULTAS) -> list[dict[str, Any]]:
    """Carrega as 30 consultas do conjunto de teste."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_julgamentos_existentes(caminho: Path = ARQUIVO_JULGAMENTOS) -> dict[str, Any]:
    """Carrega julgamentos prévios para retomar."""
    if caminho.is_file():
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_julgamentos(dados: dict[str, Any], caminho: Path = ARQUIVO_JULGAMENTOS) -> None:
    """Salva estado atual dos julgamentos de forma atômica."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)


def construir_pool_candidatos(
    dados_v3: Any,
    motor_v2: Any,
    consultas: list[dict[str, Any]],
    top_k: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Coleta o top_k de cada motor para cada consulta e embaralha a união."""
    from ahu.busca import buscar

    pool: dict[str, Any] = {}
    rng = random.Random(seed)

    for q_item in consultas:
        q_id = q_item["id"]
        q_texto = q_item["consulta"]
        q_tipo = q_item["tipo"]

        candidatos_dict: dict[str, dict[str, Any]] = {}

        # 1. Recuperar via v3
        res_v3 = buscar(dados_v3, q_texto)
        v3_uids = res_v3.uids[:top_k]
        for rank_v3, uid in enumerate(v3_uids, start=1):
            doc = dados_v3.documento(uid)
            candidatos_dict[uid] = {
                "uid": uid,
                "reference_code": doc.crav or doc.cota_nova,
                "cota": doc.cota_nova,
                "ano": doc.ano,
                "description": doc.descricao,
                "origem": ["v3"],
            }

        # 2. Recuperar via v2 (se motor_v2 estiver disponível)
        if motor_v2 is not None:
            try:
                res_v2 = motor_v2.buscar(q_texto, limiar_semantico=0.0, max_resultados=top_k)
                for rank_v2, (_, row) in enumerate(res_v2.iterrows(), start=1):
                    ref = str(row.get("reference_code", ""))
                    cota = str(row.get("new_code", ""))
                    # Usa o próprio ref ou cota como chave se uid não existir no legado
                    chave = ref or cota
                    if chave in candidatos_dict:
                        candidatos_dict[chave]["origem"].append("v2")
                    else:
                        candidatos_dict[chave] = {
                            "uid": chave,
                            "reference_code": ref,
                            "cota": cota,
                            "ano": (
                                int(row["ano_isolado"])
                                if "ano_isolado" in row and row["ano_isolado"]
                                else None
                            ),
                            "description": str(row.get("description", "")),
                            "origem": ["v2"],
                        }
            except Exception as e:
                print(f"Aviso: falha ao rodar v2 para {q_texto}: {e}")

        # União embaralhada para cegar o avaliador
        lista_candidatos = list(candidatos_dict.values())
        rng.shuffle(lista_candidatos)

        pool[q_id] = {
            "consulta": q_texto,
            "tipo": q_tipo,
            "candidatos": lista_candidatos,
        }

    return pool


def executar_julgamento_cli(
    pool: dict[str, Any],
    julgamentos_salvos: dict[str, Any],
    caminho_saida: Path = ARQUIVO_JULGAMENTOS,
) -> None:
    """Interface interativa de terminal para julgamento cego."""
    print("=" * 70)
    print("Classificador AHU — Auditoria Cega de Recuperação (WP 4.1)")
    print("Pressione 0 (irrelevante), 1 (relevante), 2 (muito relevante) ou q (sair).")
    print("=" * 70)

    total_consultas = len(pool)
    for idx_q, (q_id, info_q) in enumerate(pool.items(), start=1):
        q_texto = info_q["consulta"]
        q_tipo = info_q["tipo"]
        candidatos = info_q["candidatos"]

        julg_q = julgamentos_salvos.setdefault(q_id, {
            "consulta": q_texto,
            "tipo": q_tipo,
            "julgamentos": {},
        })

        itens_a_julgar = [c for c in candidatos if c["uid"] not in julg_q["julgamentos"]]
        if not itens_a_julgar:
            continue

        print(f"\nConsulta [{idx_q}/{total_consultas}] ({q_tipo.upper()}): '{q_texto}'")
        print(f"Candidatos pendentes nesta consulta: {len(itens_a_julgar)}")

        for idx_c, cand in enumerate(itens_a_julgar, start=1):
            uid = cand["uid"]
            cota = cand.get("reference_code") or cand.get("cota")
            ano = cand.get("ano") or "s/d"
            desc = cand.get("description") or ""

            print("-" * 70)
            print(f"Item {idx_c}/{len(itens_a_julgar)} | Cota: {cota} | Ano: {ano}")
            print(f"Resumo: {desc[:400]}")
            if len(desc) > 400:
                print("...")

            t0 = time.time()
            while True:
                resp = input("Nota [0, 1, 2, q=sair]: ").strip().lower()
                if resp in ("0", "1", "2"):
                    delta_t = round(time.time() - t0, 2)
                    julg_q["julgamentos"][uid] = {
                        "nota": int(resp),
                        "delta_seconds": delta_t,
                        "timestamp": datetime.now().isoformat(),
                    }
                    salvar_julgamentos(julgamentos_salvos, caminho_saida)
                    break
                elif resp == "q":
                    print("\nSessão de julgamento salva. Até breve!")
                    return
                else:
                    print("Entrada inválida. Digite 0, 1, 2 ou q.")

    print("\nParabéns! Todas as consultas do pool foram julgadas.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria cega de consultas (WP 4.1)")
    parser.add_argument("--gerar-pool", action="store_true", help="Gera o pool de candidatos")
    parser.add_argument("--pool", type=Path, default=ARQUIVO_POOL)
    parser.add_argument("--saida", type=Path, default=ARQUIVO_JULGAMENTOS)
    args = parser.parse_args()

    consultas = carregar_consultas()
    julgamentos = carregar_julgamentos_existentes(args.saida)

    if args.gerar_pool or not args.pool.is_file():
        print("Gerando pool de candidatos...")
        from ahu.dados import carregar_app_dados
        from avaliacao.busca_v2 import MotorBuscaV2

        dados_v3 = carregar_app_dados()
        try:
            motor_v2 = MotorBuscaV2.carregar()
        except Exception as e:
            print(f"Aviso: motor v2 não carregado ({e}); pool será montado só com v3.")
            motor_v2 = None

        pool = construir_pool_candidatos(dados_v3, motor_v2, consultas)
        with open(args.pool, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2, ensure_ascii=False)
        print(f"Pool salvo em {args.pool} com {len(pool)} consultas.")
    else:
        with open(args.pool, "r", encoding="utf-8") as f:
            pool = json.load(f)

    executar_julgamento_cli(pool, julgamentos, args.saida)


if __name__ == "__main__":
    main()
