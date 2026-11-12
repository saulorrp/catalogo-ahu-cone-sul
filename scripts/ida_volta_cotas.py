"""ida_volta_cotas.py — ida e volta das cotas em todo o corpus (auditoria científica, tarefa 7).

Para cada documento com cota do Resgate (`new_code`): Resgate -> referência -> Resgate pelo
`ahu/conversor.py`, e a busca pela cota (`ahu/busca.py`, só a rota de cotas, sem modelo) tem de
devolver o próprio `doc_uid`. É pesado (duas buscas por documento, 167 mil documentos) e por isso
fica fora do pytest padrão:

    python scripts/ida_volta_cotas.py [--app-dados app_dados] [--saida pasta] [--limite N]

Grava `falhas.csv` (uma linha por documento com alguma classe) e `resumo.json` (taxas, contagem
por classe, exemplos, por coleção). A taxa estrita conta como falha qualquer classe, inclusive
`intervalo_perde_fim` (a referência só guarda o primeiro documento de `D. 20-21`) e
`nao_e_o_primeiro` (vários documentos com a mesma cota); a de recuperação só exige que o documento
apareça entre os resultados da busca pela referência.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ahu import busca, conversor  # noqa: E402
from ahu.dados import carregar_sem_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conferência de ida e volta e busca por cota em documentos.parquet."
    )
    parser.add_argument(
        "--app-dados",
        type=Path,
        default=Path("app_dados"),
        help="Pasta dos artefatos publicados (padrão: app_dados).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
        help="Pasta para falhas.csv e resumo.json (padrão: pasta temporária).",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Limite de documentos a analisar (para depuração rápida).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pasta_dados = args.app_dados.resolve()

    if args.saida is not None:
        pasta_saida = args.saida.resolve()
        if pasta_dados in pasta_saida.parents or pasta_saida == pasta_dados:
            sys.exit("A pasta de saída não pode estar dentro de app_dados.")
    else:
        pasta_saida = Path(tempfile.mkdtemp(prefix="ida_volta_cotas_"))

    pasta_saida.mkdir(parents=True, exist_ok=True)

    dados = carregar_sem_cache(pasta_dados)
    df_docs = dados.documentos
    total_corpus = len(df_docs)

    # Considera apenas documentos com new_code preenchido
    filtro = df_docs["new_code"].notna() & (df_docs["new_code"].astype(str).str.strip() != "")
    docs_analise = df_docs[filtro]
    if args.limite is not None:
        docs_analise = docs_analise.iloc[: args.limite]

    total_com_cota = len(docs_analise)

    falhas: list[dict] = []
    contagem_classes: Counter[str] = Counter()
    exemplos_classes: defaultdict[str, list[dict]] = defaultdict(list)
    agrupado_colecao: defaultdict[str, dict] = defaultdict(
        lambda: {"total": 0, "estrito": 0, "recuperado": 0, "classes": Counter()}
    )

    sucessos_estritos = 0
    sucessos_recuperacao = 0

    for idx, (_, row) in enumerate(docs_analise.iterrows()):
        if (idx + 1) % 10000 == 0:
            sys.stderr.write(f"Analisados {idx + 1}/{total_com_cota} documentos...\n")
            sys.stderr.flush()

        doc_uid = str(row["doc_uid"])
        colecao = str(row["colecao"]) if pd.notna(row.get("colecao")) else ""
        folder = str(row["folder"]) if pd.notna(row.get("folder")) else ""
        new_code = str(row["new_code"]).strip()
        reference_code = (
            str(row["reference_code"]).strip() if pd.notna(row.get("reference_code")) else ""
        )

        classes_doc: list[str] = []
        referencia_gerada = ""
        resgate_de_volta = ""
        posicao: int | None = None
        n_resultados = 0
        recuperado_por_ref = False

        # 1. Reconhecimento da cota Resgate original
        c1 = conversor.reconhecer(new_code)
        if c1 is None:
            classes_doc.append("resgate_nao_reconhecido")
        else:
            # Intervalo com documento_fim perde o fim no código de referência
            if getattr(c1, "documento_fim", None) is not None:
                classes_doc.append("intervalo_perde_fim")

            # 2. Conversão para referência e volta para Resgate
            ref = conversor.para_referencia(c1)
            if ref is None:
                classes_doc.append("sem_serie")
            else:
                referencia_gerada = ref
                c2 = conversor.reconhecer(ref)
                if c2 is None:
                    classes_doc.append("ida_volta_difere")
                else:
                    volta = conversor.para_resgate(c2)
                    resgate_de_volta = volta or ""
                    # 3. Comparação estrutural dos quatro elementos
                    if (
                        c1.serie != c2.serie
                        or c1.subserie != c2.subserie
                        or c1.caixa != c2.caixa
                        or c1.documento != c2.documento
                    ):
                        classes_doc.append("ida_volta_difere")

        # 4. Confronto com o reference_code existente na base
        if reference_code and referencia_gerada:
            if reference_code != referencia_gerada:
                classes_doc.append("referencia_diverge_da_base")

        # 5. Busca léxica pelo código de referência gerado
        if referencia_gerada:
            resp_ref = busca.buscar(dados, referencia_gerada, encoder=None)
            uids_ref = [r.doc_uid for r in resp_ref.resultados]
            n_resultados = len(uids_ref)
            if doc_uid in uids_ref:
                recuperado_por_ref = True
                pos = uids_ref.index(doc_uid) + 1
                posicao = pos
                if pos > 1:
                    classes_doc.append("nao_e_o_primeiro")
            else:
                classes_doc.append("nao_encontrado_pela_referencia")
        else:
            classes_doc.append("nao_encontrado_pela_referencia")

        # Busca pelo new_code original
        # Se a busca não lê a cota, ela cairia na busca textual do corpus inteiro (~0,7 s por
        # documento, horas no total) e o resultado não diria nada sobre a rota de cotas.
        if busca.chave_cota(new_code) is None:
            classes_doc.append("busca_nao_reconhece_cota")
        else:
            resp_resgate = busca.buscar(dados, new_code, encoder=None)
            uids_resgate = [r.doc_uid for r in resp_resgate.resultados]
            if doc_uid not in uids_resgate:
                classes_doc.append("nao_encontrado_pelo_resgate")

        # Contabilidade por coleção e agregada
        eh_estrito = len(classes_doc) == 0
        if eh_estrito:
            sucessos_estritos += 1
        if recuperado_por_ref:
            sucessos_recuperacao += 1

        info_colecao = agrupado_colecao[colecao]
        info_colecao["total"] += 1
        if eh_estrito:
            info_colecao["estrito"] += 1
        if recuperado_por_ref:
            info_colecao["recuperado"] += 1

        for cls in classes_doc:
            contagem_classes[cls] += 1
            info_colecao["classes"][cls] += 1
            if len(exemplos_classes[cls]) < 3:
                exemplos_classes[cls].append(
                    {
                        "doc_uid": doc_uid,
                        "colecao": colecao,
                        "new_code": new_code,
                        "reference_code": reference_code,
                        "referencia_gerada": referencia_gerada,
                    }
                )

        if classes_doc:
            falhas.append(
                {
                    "doc_uid": doc_uid,
                    "colecao": colecao,
                    "folder": folder,
                    "new_code": new_code,
                    "reference_code": reference_code,
                    "referencia_gerada": referencia_gerada,
                    "resgate_de_volta": resgate_de_volta,
                    "classes": ";".join(classes_doc),
                    "posicao": str(posicao) if posicao is not None else "",
                    "n_resultados": str(n_resultados),
                }
            )

    # Gravação do falhas.csv
    caminho_csv = pasta_saida / "falhas.csv"
    with open(caminho_csv, "w", encoding="utf-8", newline="") as f:
        campos = [
            "doc_uid",
            "colecao",
            "folder",
            "new_code",
            "reference_code",
            "referencia_gerada",
            "resgate_de_volta",
            "classes",
            "posicao",
            "n_resultados",
        ]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(falhas)

    taxa_estrita = sucessos_estritos / total_com_cota if total_com_cota else 0.0
    taxa_recuperacao = sucessos_recuperacao / total_com_cota if total_com_cota else 0.0

    top_5 = []
    for cls, count in contagem_classes.most_common(5):
        top_5.append({"classe": cls, "contagem": count, "exemplos": exemplos_classes[cls]})

    por_colecao_resumo = {}
    for col, dados_col in sorted(agrupado_colecao.items()):
        tot = dados_col["total"]
        por_colecao_resumo[col] = {
            "total": tot,
            "taxa_estrita": dados_col["estrito"] / tot if tot else 0.0,
            "taxa_recuperacao": dados_col["recuperado"] / tot if tot else 0.0,
            "classes": dict(dados_col["classes"]),
        }

    resumo_dict = {
        "total": total_corpus,
        "com_cota": total_com_cota,
        "taxa_estrita": taxa_estrita,
        "taxa_recuperacao": taxa_recuperacao,
        "contagem_por_classe": dict(contagem_classes),
        "classes_mais_comuns": top_5,
        "por_colecao": por_colecao_resumo,
    }

    caminho_json = pasta_saida / "resumo.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resumo_dict, f, indent=2, ensure_ascii=False)

    # Relatório impresso no stdout
    print("\n=== Conferência de Ida e Volta de Cotas ===")
    print(f"Total de documentos no corpus: {total_corpus}")
    print(f"Documentos analisados com cota: {total_com_cota}")
    print(f"Taxa estrita (sem nenhuma discrepância): {taxa_estrita * 100:.2f}%")
    print(f"Taxa de recuperação pela referência:    {taxa_recuperacao * 100:.2f}%")
    print("\nOcorrências por classe:")
    if contagem_classes:
        for cls, count in contagem_classes.most_common():
            print(f"  {cls:<32}: {count}")
    else:
        print("  Nenhuma divergência registrada.")
    print(f"\nArquivos gerados em: {pasta_saida}")
    print(f"  - CSV de falhas: {caminho_csv}")
    print(f"  - Resumo JSON:   {caminho_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
