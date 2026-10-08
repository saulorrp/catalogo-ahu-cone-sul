"""importar_biografias_v2.py — WP 1.3: camadas `entidades_v2` e `biografias_v2`.

Entradas (pasta do pipeline de biografias v2, spec §3.2):
    ahu_sul_catalog_biografias.json   remetentes já consolidados por documento (saída do p5)
    entidades_por_remetente.json      entidades por `sender_name` bruto, com títulos e cargos (p1)
    doc_pessoas.json                  pessoa_id por documento e posição (p2)
    pessoas.json                      pessoas consolidadas (p2)
    biografias_validadas.json         nascimento validado por pessoa (p3-p4, com --wikidata)
    revisao_biografias.csv            coluna `decisao_manual` preenchida à mão pelo autor

Saídas:
    corpus/camadas/entidades_v2/
        entidades.parquet        doc_uid x posicao x nome x tipo x pessoa_uid (+ títulos, cargos)
        pessoas.parquet          uma linha por pessoa_uid: identidade e atividade (p1-p2)
        registro_pessoas.json    assinatura -> pessoa_uid permanente (spec §3.3)
        camada.json
    corpus/camadas/biografias_v2/
        biografias.parquet       uma linha por pessoa_uid: nascimento, nível, decisão, fonte, Wikidata
        camada.json

Regras:
  * o `P0001...` do pipeline muda a cada execução do p2; o `pessoa_uid` é permanente e vem da
    assinatura (chave do nome principal + chaves dos títulos + intervalo de anos). Uid nunca é
    reaproveitado, nem de pessoa que saiu (spec §3.3);
  * o contrato da §3.2 é verificado antes de qualquer escrita, e a importação para com uma
    mensagem clara;
  * **D2:** só ALTA e MÉDIA (ou `decisao_manual = aceitar`) são exibíveis. REJEITADO e BAIXA ficam
    na camada, marcados `exibivel = False`, para que a correção possa voltar pelo `decisao_manual`.

Uso:
    python build/importar_biografias_v2.py
    python build/importar_biografias_v2.py --pipeline ../biografias_v2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ahu.texto import chave_nome, chave_titulo  # noqa: E402
from build.importar_base import (  # noqa: E402
    CAMPOS_PROIBIDOS,
    ErroImportacao,
    commit_atual,
    sha1_hex,
    sha256_arquivo,
)
from build.importar_camadas import registrar_no_manifesto, uids_da_entrada  # noqa: E402

VERSAO_SCRIPT = "1.0.0"
CAMADA_ENTIDADES = "entidades_v2"
CAMADA_BIOGRAFIAS = "biografias_v2"
PIPELINE_PADRAO = RAIZ.parent / "biografias_v2"

# Subir esta versão invalida os pessoa_uid já publicados: exige migrar o registro.
ASSINATURA_VERSAO = 1
REGRA_ASSINATURA = (
    "assinatura = chave_nome(nome_principal) | chaves dos títulos em ordem | ano_min-ano_max; "
    "pessoa_uid = 'pes:' + sha1(assinatura)[:8]"
)

ARQUIVOS = {
    "catalogo": "ahu_sul_catalog_biografias.json",
    "entidades": "entidades_por_remetente.json",
    "doc_pessoas": "doc_pessoas.json",
    "pessoas": "pessoas.json",
    "biografias": "biografias_validadas.json",
    "revisao": "revisao_biografias.csv",
}

# Níveis de confiança do p4, do melhor para o pior. Vêm do dado; o código só os ordena.
NIVEIS_EXIBIVEIS = ("ALTA", "MEDIA")
DECISOES = {"", "aceitar", "rejeitar"}

# Rótulos da spec §6.3, decididos pelo autor em 2026-10-08. Nunca "Desconhecida", que sugere
# falta de pesquisa.
ROTULOS = {
    "ALTA": "Nascimento identificado (confiança alta)",
    "MEDIA": "Nascimento identificado (confiança média)",
    "BAIXA": "Nascimento não confirmado",
    "REJEITADO": "Não identificado",
    "IDENTIFICADO_SEM_LOCAL": "Identificado, local não encontrado",
    "NAO_ENCONTRADO": "Não identificado",
}

CAMPOS_BIOGRAFIA = [
    "nome_na_fonte", "ano_nascimento", "ano_morte", "cidade_nascimento", "regiao_nascimento",
    "pais_nascimento_epoca", "pais_atual", "url_fonte", "trecho_fonte", "justificativa_identidade",
    "confianca_modelo", "motivo_rejeicao", "conflito_local_wikidata",
]
CAMPOS_ANO = {"ano_nascimento", "ano_morte", "wikidata_nascimento", "wikidata_morte"}


# ---------------------------------------------------------------------------
# Leitura e contrato de entrada (spec §3.2)
# ---------------------------------------------------------------------------

def _ler_json(caminho: Path):
    if not caminho.exists():
        raise ErroImportacao(f"Arquivo do pipeline não encontrado: {caminho}")
    try:
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ErroImportacao(
            f"{caminho.name} está truncado ou não é JSON válido ({e}). Rode o pipeline de novo."
        ) from e


def ler_revisao(caminho: Path) -> dict[str, str]:
    """`revisao_biografias.csv` -> {pessoa_id: decisao_manual}. CSV com BOM e separador ';'."""
    if not caminho.exists():
        raise ErroImportacao(f"Arquivo do pipeline não encontrado: {caminho}")
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    if linhas and "decisao_manual" not in linhas[0]:
        raise ErroImportacao(
            f"{caminho.name} não tem a coluna 'decisao_manual'. Gere o CSV de revisão de novo (p4)."
        )
    decisoes = {}
    for linha in linhas:
        pid = (linha.get("id") or "").strip()
        decisao = (linha.get("decisao_manual") or "").strip().lower()
        if decisao not in DECISOES:
            raise ErroImportacao(
                f"{caminho.name}, pessoa {pid}: decisao_manual = {decisao!r}. "
                f"Use {', '.join(sorted(d for d in DECISOES if d))} ou deixe vazio."
            )
        decisoes[pid] = decisao
    return decisoes


def verificar_contrato(dados: dict, esperado_documentos: int) -> None:
    """As quatro verificações da spec §3.2. Para antes de qualquer escrita."""
    catalogo, doc_pessoas = dados["catalogo"], dados["doc_pessoas"]
    pessoas, biografias = dados["pessoas"], dados["biografias"]

    if not isinstance(catalogo, list) or len(catalogo) != esperado_documentos:
        raise ErroImportacao(
            f"(1) O catálogo deveria ter {esperado_documentos} documentos, tem "
            f"{len(catalogo) if isinstance(catalogo, list) else 'formato inválido'}."
        )

    if not isinstance(doc_pessoas, list) or len(doc_pessoas) != len(catalogo):
        raise ErroImportacao(
            f"(2) doc_pessoas.json tem {len(doc_pessoas) if isinstance(doc_pessoas, list) else '?'} "
            f"documentos e o catálogo {len(catalogo)}. Rode o p2 sobre o mesmo catálogo."
        )
    for i, (doc, ids) in enumerate(zip(catalogo, doc_pessoas)):
        if not isinstance(doc, dict) or not isinstance(doc.get("remetentes"), list):
            raise ErroImportacao(f"(1) Documento {i} sem a lista 'remetentes'. Rode o p5 de novo.")
        do_catalogo = [r.get("pessoa_id") for r in doc["remetentes"]]
        if list(ids or []) != do_catalogo:
            raise ErroImportacao(
                f"(2) Documento {i}: doc_pessoas.json diz {list(ids or [])} e o catálogo "
                f"{do_catalogo}. A ordem ou o número de entidades não bate; rode o p5 de novo."
            )
        sobrou = [c for c in CAMPOS_PROIBIDOS if c in doc]
        if sobrou:
            raise ErroImportacao(
                f"(4) Documento {i}: sobrou campo do pipeline antigo ({', '.join(sobrou)}). "
                "Rode o p5 de novo."
            )

    for nome, tabela in (("pessoas.json", pessoas), ("biografias_validadas.json", biografias)):
        if not isinstance(tabela, list):
            raise ErroImportacao(f"(3) {nome} deveria ser uma lista de pessoas.")
    em_pessoas = {p.get("id") for p in pessoas}
    em_biografias = {b.get("id") for b in biografias}
    citados = {r.get("pessoa_id") for doc in catalogo for r in doc["remetentes"] if r.get("pessoa_id")}
    for nome, existentes in (("pessoas.json", em_pessoas), ("biografias_validadas.json", em_biografias)):
        faltam = sorted(citados - existentes)
        if faltam:
            raise ErroImportacao(
                f"(3) {len(faltam)} pessoa_id citados no catálogo não existem em {nome} "
                f"(ex.: {', '.join(faltam[:5])}). Rode o p2 e o p4 sobre o mesmo catálogo."
            )
    if len(em_pessoas) != len(pessoas) or len(em_biografias) != len(biografias):
        raise ErroImportacao("(3) pessoas.json ou biografias_validadas.json tem id repetido.")


def verificar_colecao_unica(corpus: Path, colecao: str, doc_uids: list[str]) -> None:
    """As pessoas são consolidadas por coleção; não dá para sobrepor duas camadas sem perder uma.

    Consolidar pessoas entre coleções (a mesma pessoa em `ahu-sul` e em `ahu-nordeste`) é o WP 5.4.
    Até lá, reimportar outra coleção por cima apagaria a primeira: paramos antes disso.
    """
    arq = corpus / "camadas" / CAMADA_ENTIDADES / "entidades.parquet"
    if not arq.exists():
        return
    de_outras = set(pd.read_parquet(arq, columns=["doc_uid"])["doc_uid"]) - set(doc_uids)
    if de_outras:
        raise ErroImportacao(
            f"A camada {CAMADA_ENTIDADES} já cobre {len(de_outras)} documentos de outra coleção. "
            f"Importar '{colecao}' por cima apagaria as pessoas daquela coleção, e consolidar "
            "pessoas entre coleções é o WP 5.4. Apague a camada à mão se for isso que você quer."
        )


# ---------------------------------------------------------------------------
# pessoa_uid (spec §3.3)
# ---------------------------------------------------------------------------

def assinatura_pessoa(pessoa: dict) -> str:
    """Chave do nome principal + chaves dos títulos + intervalo de anos."""
    nome = chave_nome(pessoa.get("nome_principal") or "")
    titulos = sorted({chave_titulo(t) for t in (pessoa.get("titulos") or []) if t})
    anos = f"{pessoa.get('ano_min') or ''}-{pessoa.get('ano_max') or ''}"
    return "|".join([nome, ";".join(titulos), anos])


def atribuir_pessoa_uids(pessoas: list[dict], registro_anterior: dict | None) -> dict[str, str]:
    """{pessoa_id: pessoa_uid}, reaproveitando os uids já registrados pela assinatura."""
    anterior = (registro_anterior or {}).get("pessoas", [])
    if (registro_anterior or {}).get("assinatura_versao", ASSINATURA_VERSAO) != ASSINATURA_VERSAO:
        raise ErroImportacao(
            f"registro_pessoas.json foi escrito com a assinatura versão "
            f"{registro_anterior['assinatura_versao']} e o script usa a {ASSINATURA_VERSAO}. "
            "Migre o registro antes de reimportar: os pessoa_uid são públicos."
        )
    por_assinatura: dict[str, list[str]] = {}
    for r in anterior:
        por_assinatura.setdefault(r["assinatura"], []).append(r["pessoa_uid"])
    reservados = {r["pessoa_uid"] for r in anterior}

    usados: set[str] = set()
    uid_por_id: dict[str, str] = {}
    for pessoa in pessoas:
        assinatura = assinatura_pessoa(pessoa)
        uid = next((u for u in por_assinatura.get(assinatura, []) if u not in usados), None)
        if uid is None:
            n = 1
            while True:
                base = assinatura if n == 1 else f"{assinatura}|#{n}"
                uid = f"pes:{sha1_hex(base)[:8]}"
                if uid not in usados and uid not in reservados:
                    break
                n += 1
        usados.add(uid)
        uid_por_id[pessoa["id"]] = uid

    if len(set(uid_por_id.values())) != len(uid_por_id):
        raise ErroImportacao("Colisão de pessoa_uid não resolvida (não deveria ocorrer).")
    return uid_por_id


def montar_registro_pessoas(
    pessoas: list[dict], uid_por_id: dict[str, str], registro_anterior: dict | None
) -> dict:
    atuais = [
        {"pessoa_uid": uid_por_id[p["id"]], "assinatura": assinatura_pessoa(p),
         "pessoa_id_atual": p["id"], "nome": p.get("nome_principal") or "; ".join(p.get("titulos") or [])}
        for p in pessoas
    ]
    ativos = {r["pessoa_uid"] for r in atuais}
    # Pessoas que saíram ficam no registro para que o uid nunca seja reaproveitado.
    aposentadas = [
        {**r, "pessoa_id_atual": None}
        for r in (registro_anterior or {}).get("pessoas", [])
        if r["pessoa_uid"] not in ativos
    ]
    return {
        "regra": REGRA_ASSINATURA,
        "assinatura_versao": ASSINATURA_VERSAO,
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pessoas": atuais + aposentadas,
    }


# ---------------------------------------------------------------------------
# Camada entidades_v2
# ---------------------------------------------------------------------------

def montar_entidades(
    catalogo: list[dict], doc_uids: list[str], por_remetente: dict, uid_por_id: dict[str, str]
) -> pd.DataFrame:
    linhas = []
    for doc, doc_uid in zip(catalogo, doc_uids):
        extras = por_remetente.get(doc.get("sender_name"), [])
        for posicao, remetente in enumerate(doc["remetentes"]):
            extra = extras[posicao] if posicao < len(extras) else {}
            if extra and extra.get("nome") != remetente.get("nome"):
                extra = next((e for e in extras if e.get("nome") == remetente.get("nome")), {})
            pessoa_id = remetente.get("pessoa_id")
            linhas.append({
                "doc_uid": doc_uid,
                "posicao": posicao,
                "nome": remetente.get("nome"),
                "tipo": remetente.get("tipo"),
                "pessoa_uid": uid_por_id.get(pessoa_id) if pessoa_id else None,
                "titulos": list(extra.get("titulos") or []),
                "cargos": list(extra.get("cargos") or []),
                "autoria_verificada": bool(doc.get("autoria_verificada")),
            })
    df = pd.DataFrame(linhas, columns=["doc_uid", "posicao", "nome", "tipo", "pessoa_uid",
                                       "titulos", "cargos", "autoria_verificada"])
    df["posicao"] = df["posicao"].astype("int16")
    return df


def montar_pessoas(pessoas: list[dict], uid_por_id: dict[str, str]) -> pd.DataFrame:
    linhas = [{
        "pessoa_uid": uid_por_id[p["id"]],
        "tipo": p.get("tipo"),
        "nome_principal": p.get("nome_principal"),
        "nome_exibicao": p.get("nome_principal") or "; ".join(p.get("titulos") or []) or "(sem nome)",
        "variantes_nome": list(p.get("variantes_nome") or []),
        "titulos": list(p.get("titulos") or []),
        "cargos": list(p.get("cargos") or []),
        "ano_min": p.get("ano_min"),
        "ano_max": p.get("ano_max"),
        "n_documentos": p.get("n_documentos"),
        "locais_escrita": list(p.get("locais_escrita") or []),
        "destinatarios": list(p.get("destinatarios") or []),
        "categorias": list(p.get("categorias") or []),
        "nome_curto": bool(p.get("nome_curto")),
        "pessoa_id_pipeline": p["id"],
    } for p in pessoas]
    df = pd.DataFrame(linhas)
    for coluna in ("ano_min", "ano_max", "n_documentos"):
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Camada biografias_v2
# ---------------------------------------------------------------------------

def _texto(valor):
    if valor is None:
        return None
    return str(valor).strip() or None


def montar_biografias(
    biografias: list[dict], uid_por_id: dict[str, str], decisoes: dict[str, str]
) -> pd.DataFrame:
    sem_decisao = [b["id"] for b in biografias if b["id"] not in decisoes]
    if sem_decisao:
        raise ErroImportacao(
            f"{len(sem_decisao)} pessoas de biografias_validadas.json não estão no CSV de revisão "
            f"(ex.: {', '.join(sem_decisao[:5])}). Gere o CSV de novo (p4)."
        )

    linhas = []
    for b in biografias:
        nivel = (b.get("confianca_calculada") or "").strip().upper()
        if nivel not in ROTULOS:
            raise ErroImportacao(
                f"{b['id']}: confianca_calculada = {b.get('confianca_calculada')!r}, que o script não "
                f"conhece. Níveis esperados: {', '.join(ROTULOS)}."
            )
        decisao = decisoes[b["id"]]
        wikidata = b.get("wikidata") or {}
        linha = {
            "pessoa_uid": uid_por_id[b["id"]],
            "nivel": nivel,
            "decisao_manual": decisao or None,
            # D2: a decisão manual do autor manda sobre o nível calculado, nos dois sentidos.
            "exibivel": decisao == "aceitar" or (decisao != "rejeitar" and nivel in NIVEIS_EXIBIVEIS),
            "rotulo": ROTULOS[nivel],
            "origem_local": b.get("origem_local") or ("fonte" if b.get("cidade_nascimento") else None),
            "wikidata_qid": _texto(wikidata.get("qid")),
            "wikidata_nascimento": wikidata.get("nascimento"),
            "wikidata_morte": wikidata.get("morte"),
            "wikidata_local": _texto(wikidata.get("local")),
            "wikidata_pais": _texto(wikidata.get("pais")),
            "checagens": json.dumps(b.get("checagens") or {}, ensure_ascii=False),
            "pessoa_id_pipeline": b["id"],
        }
        for campo in CAMPOS_BIOGRAFIA:
            valor = b.get(campo)
            linha[campo] = valor if campo in CAMPOS_ANO else _texto(valor)
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    for coluna in CAMPOS_ANO:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").astype("Int64")
    df["exibivel"] = df["exibivel"].astype(bool)
    colunas = ["pessoa_uid", "nivel", "decisao_manual", "exibivel", "rotulo", "nome_na_fonte",
               "ano_nascimento", "ano_morte", "cidade_nascimento", "regiao_nascimento",
               "pais_nascimento_epoca", "pais_atual", "origem_local", "url_fonte", "trecho_fonte",
               "justificativa_identidade", "confianca_modelo", "motivo_rejeicao",
               "conflito_local_wikidata", "wikidata_qid", "wikidata_nascimento", "wikidata_morte",
               "wikidata_local", "wikidata_pais", "checagens", "pessoa_id_pipeline"]
    return df[colunas]


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _contagens(serie) -> dict:
    return {str(k): int(v) for k, v in serie.dropna().value_counts().sort_index().items()}


def gravar_camada(pasta: Path, tabelas: dict[str, pd.DataFrame], camada: dict) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    for nome, df in tabelas.items():
        df.to_parquet(pasta / nome, index=False)
    camada = {
        **camada,
        "esquema": {nome: {c: str(t) for c, t in df.dtypes.items()} for nome, df in tabelas.items()},
        "data": _agora(),
    }
    with open(pasta / "camada.json", "w", encoding="utf-8") as f:
        json.dump(camada, f, ensure_ascii=False, indent=2)
    return camada


def importar(
    pipeline: Path, corpus: Path, colecao: str = "ahu-sul", esperado_documentos: int | None = None
) -> dict:
    pipeline, corpus = Path(pipeline), Path(corpus)
    dados = {chave: _ler_json(pipeline / nome) for chave, nome in ARQUIVOS.items()
             if nome.endswith(".json")}
    decisoes = ler_revisao(pipeline / ARQUIVOS["revisao"])

    doc_uids = uids_da_entrada(dados["catalogo"], corpus / "colecoes" / colecao)
    verificar_contrato(dados, esperado_documentos or len(doc_uids))
    verificar_colecao_unica(corpus, colecao, doc_uids)

    pasta_entidades = corpus / "camadas" / CAMADA_ENTIDADES
    arq_registro = pasta_entidades / "registro_pessoas.json"
    registro_anterior = _ler_json(arq_registro) if arq_registro.exists() else None
    uid_por_id = atribuir_pessoa_uids(dados["pessoas"], registro_anterior)

    entidades = montar_entidades(dados["catalogo"], doc_uids, dados["entidades"], uid_por_id)
    pessoas = montar_pessoas(dados["pessoas"], uid_por_id)
    biografias = montar_biografias(dados["biografias"], uid_por_id, decisoes)

    processo = {"script": "build/importar_biografias_v2.py", "versao_script": VERSAO_SCRIPT,
                "commit": commit_atual()}
    entradas = {nome: sha256_arquivo(pipeline / nome) for nome in ARQUIVOS.values()}

    registro = montar_registro_pessoas(dados["pessoas"], uid_por_id, registro_anterior)
    pasta_entidades.mkdir(parents=True, exist_ok=True)
    with open(arq_registro, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=1)

    camada = gravar_camada(
        pasta_entidades, {"entidades.parquet": entidades, "pessoas.parquet": pessoas},
        {
            "nome": CAMADA_ENTIDADES, "tipo": "entidades", "versao": "2", "colecoes": [colecao],
            "origem": "pipeline de biografias v2, etapas p1 (extrair entidades) e p2 (consolidar)",
            "regra_pessoa_uid": REGRA_ASSINATURA, "assinatura_versao": ASSINATURA_VERSAO,
            "documentos": int(entidades["doc_uid"].nunique()),
            "entidades": int(len(entidades)),
            "pessoas": int(len(pessoas)),
            "contagem_por_tipo": _contagens(entidades["tipo"]),
            "entidades_com_pessoa_uid": int(entidades["pessoa_uid"].notna().sum()),
            "processo": processo, "entradas": entradas,
        },
    )
    registrar_no_manifesto(corpus, pasta_entidades, camada,
                           ("entidades.parquet", "pessoas.parquet", "registro_pessoas.json",
                            "camada.json"))

    pasta_biografias = corpus / "camadas" / CAMADA_BIOGRAFIAS
    camada_bio = gravar_camada(
        pasta_biografias, {"biografias.parquet": biografias},
        {
            "nome": CAMADA_BIOGRAFIAS, "tipo": "biografias", "versao": "2", "colecoes": [colecao],
            "origem": "pipeline de biografias v2, etapas p3 (buscar), p4 (validar, --wikidata) e p5",
            "documentos": int(entidades["doc_uid"].nunique()),
            "pessoas": int(len(biografias)),
            "exibiveis": int(biografias["exibivel"].sum()),
            "regra_exibicao": "D2: nível ALTA ou MÉDIA, ou decisao_manual = aceitar; "
                              "decisao_manual = rejeitar nunca é exibida",
            "rotulos": ROTULOS,
            "contagem_por_nivel": _contagens(biografias["nivel"]),
            "contagem_por_decisao": _contagens(biografias["decisao_manual"]),
            "com_wikidata": int(biografias["wikidata_qid"].notna().sum()),
            "processo": processo, "entradas": entradas,
        },
    )
    registrar_no_manifesto(corpus, pasta_biografias, camada_bio,
                           ("biografias.parquet", "camada.json"))

    return {
        "documentos": len(doc_uids),
        "entidades": len(entidades),
        "pessoas": len(pessoas),
        "pessoa_uids_novos": sum(
            1 for r in registro["pessoas"]
            if r["pessoa_id_atual"] and r["pessoa_uid"] not in
            {x["pessoa_uid"] for x in (registro_anterior or {}).get("pessoas", [])}
        ),
        "exibiveis": int(biografias["exibivel"].sum()),
        "por_nivel": camada_bio["contagem_por_nivel"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default=str(PIPELINE_PADRAO),
                    help="pasta com a saída do pipeline de biografias v2")
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    ap.add_argument("--colecao", default="ahu-sul")
    args = ap.parse_args()
    try:
        r = importar(Path(args.pipeline), Path(args.corpus), args.colecao)
    except ErroImportacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"{r['documentos']} documentos, {r['entidades']} entidades, {r['pessoas']} pessoas "
          f"({r['pessoa_uids_novos']} pessoa_uid novos).")
    print("Nascimento por nível: " + ", ".join(f"{k}={v}" for k, v in r["por_nivel"].items()))
    print(f"Exibíveis (ALTA, MÉDIA ou aceitar): {r['exibiveis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
