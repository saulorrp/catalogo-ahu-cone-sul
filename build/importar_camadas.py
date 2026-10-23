"""importar_camadas.py — WP 1.2: camadas `tipologia_v1` e `sociolinguistica_deepseek_v3`.

Entrada : ahu_sul_catalog_biografias.json (saída do p5) + a base do WP 1.1, que dá os `doc_uid`
Saídas  : corpus/camadas/tipologia_v1/
              tipologia.parquet    doc_uid, tipologia_bruta, tipologia, tipologia_chave,
                                   typology_status, tipologia_corrigida
              camada.json          regra de normalização, tabela de correções aplicada, contagens
          corpus/camadas/sociolinguistica_deepseek_v3/
              sociolinguistica.parquet  doc_uid, sender_category, vector, typology_impact,
                                        vernacular_score, scribe_mediation_likely, justificativa
              camada.json          modelo, prompt do preprint §2, auditoria §2.1, contagens

Regras (spec §3.1, §6.2; AGENTS.md 7 e 9):
  * uma linha por `doc_uid`, na ordem da base;
  * `vernacular_score` é inteiro de 1 a 9 — **nunca** dividido por 10. Valor fora disso para a importação;
  * as tipologias vêm dos dados. O script só normaliza a forma (maiúsculas, espaços, pontuação) e
    aplica a tabela de correções de `corpus/correcoes_tipologia.json`, que o autor edita à mão.
    Nenhum vocabulário de tipologias está embutido no código;
  * `scribe_mediation_likely` fica nesta camada, mas o `construir_app.py` (WP 1.5) só o leva para o
    `pdf_extra.parquet` (D1).

Uso:
    python build/importar_camadas.py
    python build/importar_camadas.py --prompt caminho/prompt_deepseek.txt   # grava o prompt do preprint §2
    python build/importar_camadas.py --so tipologia
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from build.importar_base import (  # noqa: E402
    ENTRADA_PADRAO,
    ErroImportacao,
    commit_atual,
    hash_conteudo,
    normalizar,
    sha256_arquivo,
)

VERSAO_SCRIPT = "1.0.0"
CAMADA_TIPOLOGIA = "tipologia_v1"
CAMADA_SOCIO = "sociolinguistica_deepseek_v3"
MODELO_SOCIO = "deepseek-v3"

REGRA_TIPOLOGIA = (
    "maiúsculas, NFC, espaços colapsados, pontuação e aspas das pontas removidas; "
    "depois a tabela de correções de corpus/correcoes_tipologia.json, casada pela forma normalizada "
    "com acentos (para que a tabela possa corrigir acento trocado). "
    "Valor vazio -> tipologia nula e typology_status falso."
)

# Campos do DeepSeek (preprint §2) -> nome da coluna na camada.
CAMPOS_SOCIO = {
    "sender_category": "sender_category",
    "vector": "vector",
    "typology_impact": "typology_impact",
    "vernacular_score": "vernacular_score",
    "scribe_mediation_likely": "scribe_mediation_likely",
    "sociolinguistic_reasoning_by_deepseek_v3": "justificativa",
}

# Números da auditoria do preprint §2.1, mostrados na interface (spec §6.2).
# Escala do SRSP desta camada. É um dado da camada, não uma regra do código: uma reclassificação
# futura é uma camada nova, que declara a sua própria escala no seu camada.json (decisão 2026-10-08).
ESCALA_PADRAO = [1, 9]

AUDITORIA_SRSP = {
    "escala": "inteira 1-9 (SRSP)",
    "mae": 1.13,
    "r": 0.48,
    "vies": 0.59,
    "n": 705,
    "avaliadores": 1,
    "nota": "Diferenças de ±1 ponto não são distinguíveis.",
}

PROMPT_PENDENTE = (
    "PENDENTE: colar aqui o prompt do preprint §2, rodando "
    "`python build/importar_camadas.py --prompt <arquivo.txt>`."
)

# Tabela inicial de correções: só erros de extração inequívocos. O autor acrescenta os que faltarem.
CORRECOES_INICIAIS = {
    "_LEIA-ME": (
        "Correções de tipologia da camada tipologia_v1. Chave = tipologia como vem do catálogo, "
        "em maiúsculas e com os acentos tal como estão (é por isso que acento trocado pode ser "
        "corrigido aqui); valor = forma correta (string vazia = tipologia desconhecida). "
        "Acrescente aqui qualquer erro de extração e rode build/importar_camadas.py de novo. "
        "Nenhuma tipologia é definida no código: esta tabela e os dados mandam."
    ),
    "COSIJLTA": "CONSULTA",
    "OFICÍO": "OFÍCIO",
    "RELÁTORIO": "RELATÓRIO",
    "UNKNOWN": "",
}

RE_PONTAS = re.compile(r"^[\s\"'“”«»\[\](){}.,;:–—-]+|[\s\"'“”«»\[\](){}.,;:–—-]+$")


# ---------------------------------------------------------------------------
# Tipologia
# ---------------------------------------------------------------------------

def normalizar_tipologia(bruto) -> str:
    """Só a forma: maiúsculas, espaços colapsados, pontuação das pontas fora. Não traduz nada."""
    if bruto is None:
        return ""
    t = unicodedata.normalize("NFC", str(bruto))
    t = RE_PONTAS.sub("", t)
    return re.sub(r"\s+", " ", t).strip().upper()


def chave_tipologia(tipologia: str) -> str:
    """Chave de filtro e de URL: sem acentos, maiúscula (usa a normalização da base)."""
    return normalizar(tipologia).upper()


def carregar_correcoes(arquivo: Path) -> dict[str, str]:
    """Lê (ou cria com os padrões) a tabela de correções. Devolve só os pares úteis."""
    if not arquivo.exists():
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(CORRECOES_INICIAIS, f, ensure_ascii=False, indent=2)
    with open(arquivo, encoding="utf-8") as f:
        tabela = json.load(f)
    if not isinstance(tabela, dict):
        raise ErroImportacao(f"{arquivo.name} deve ser um objeto JSON (chave = tipologia, valor = correção).")
    correcoes = {}
    for chave, valor in tabela.items():
        if chave.startswith("_"):
            continue
        if not isinstance(valor, str):
            raise ErroImportacao(f"{arquivo.name}: a correção de {chave!r} deve ser uma string.")
        correcoes[normalizar_tipologia(chave)] = normalizar_tipologia(valor)
    return correcoes


def vocabulario_canonico(docs: list[dict]) -> set[str]:
    """As tipologias que o catálogo marca como canônicas em algum documento.

    Sai dos dados, nunca do código (AGENTS.md 7). Serve para que uma correção de forma
    ('OFICÍO' -> 'OFÍCIO') herde o status da forma certa, em vez de ficar como a revisar.
    """
    return {
        normalizar_tipologia(d.get("extracted_typology"))
        for d in docs
        if d.get("typology_status") is True and normalizar_tipologia(d.get("extracted_typology"))
    }


def montar_tipologia(docs: list[dict], uids: list[str], correcoes: dict[str, str]) -> pd.DataFrame:
    canonicas = vocabulario_canonico(docs)
    linhas = []
    for doc, uid in zip(docs, uids):
        bruto = doc.get("extracted_typology")
        forma = normalizar_tipologia(bruto)
        corrigida = forma in correcoes
        tipologia = correcoes[forma] if corrigida else forma
        status = doc.get("typology_status")
        if not isinstance(status, bool):
            raise ErroImportacao(
                f"{uid}: typology_status deveria ser verdadeiro ou falso, veio {status!r}."
            )
        linhas.append({
            "doc_uid": uid,
            "tipologia_bruta": None if bruto is None else str(bruto),
            "tipologia": tipologia or None,
            "tipologia_chave": chave_tipologia(tipologia) or None,
            "typology_status": bool(tipologia) and (status or tipologia in canonicas),
            "tipologia_corrigida": corrigida,
        })
    df = pd.DataFrame(linhas, columns=[
        "doc_uid", "tipologia_bruta", "tipologia", "tipologia_chave",
        "typology_status", "tipologia_corrigida",
    ])
    df["typology_status"] = df["typology_status"].astype(bool)
    df["tipologia_corrigida"] = df["tipologia_corrigida"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Sociolinguística
# ---------------------------------------------------------------------------

def ler_escala(pasta: Path) -> list[int]:
    """A escala do SRSP sai do `camada.json` da camada, não do código (decisão 2026-10-08).

    Uma camada que ainda não existe nasce com `ESCALA_PADRAO`; uma reclassificação com outra
    escala é uma camada nova, e basta declarar `"escala_srsp": [min, max]` no seu `camada.json`.
    """
    arq = pasta / "camada.json"
    if not arq.exists():
        return list(ESCALA_PADRAO)
    with open(arq, encoding="utf-8") as f:
        escala = json.load(f).get("escala_srsp")
    if escala is None or isinstance(escala, str):
        # Camadas escritas antes da decisão de 08/10 guardavam aqui uma frase; valem o padrão.
        return list(ESCALA_PADRAO)
    if (not isinstance(escala, list) or len(escala) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in escala)
            or escala[0] >= escala[1]):
        raise ErroImportacao(
            f"{arq}: 'escala_srsp' deve ser [minimo, maximo] com dois inteiros crescentes, "
            f"veio {escala!r}."
        )
    return list(escala)


def validar_srsp(valor, uid: str, escala: list[int] | None = None) -> int:
    """SRSP é inteiro, na escala que a camada declara. Fração ou valor fora dela para a importação."""
    minimo, maximo = escala or ESCALA_PADRAO
    if isinstance(valor, bool) or valor is None:
        raise ErroImportacao(f"{uid}: vernacular_score ausente ou inválido ({valor!r}).")
    if isinstance(valor, float) and not valor.is_integer():
        raise ErroImportacao(
            f"{uid}: vernacular_score veio fracionário ({valor!r}). A escala SRSP desta camada é "
            f"inteira de {minimo} a {maximo}; o valor nunca é dividido por 10."
        )
    try:
        n = int(valor)
    except (TypeError, ValueError) as e:
        raise ErroImportacao(f"{uid}: vernacular_score não é um número ({valor!r}).") from e
    if not minimo <= n <= maximo:
        raise ErroImportacao(
            f"{uid}: vernacular_score fora da escala {minimo}-{maximo} desta camada ({valor!r}). "
            f"Se vier de 0 a 1, o p5 dividiu o valor por 10 — rode o p5 de novo sem dividir. "
            f"Se a reclassificação usa outra escala, ela é uma camada nova, com o seu "
            f"'escala_srsp' no camada.json."
        )
    return n


def montar_sociolinguistica(docs: list[dict], uids: list[str],
                            escala: list[int] | None = None) -> pd.DataFrame:
    linhas = []
    for doc, uid in zip(docs, uids):
        faltam = [c for c in CAMPOS_SOCIO if c not in doc]
        if faltam:
            raise ErroImportacao(
                f"{uid}: faltam campos do DeepSeek ({', '.join(faltam)}). Rode o p5 de novo."
            )
        mediacao = doc["scribe_mediation_likely"]
        if not isinstance(mediacao, bool):
            raise ErroImportacao(
                f"{uid}: scribe_mediation_likely deveria ser verdadeiro ou falso, veio {mediacao!r}."
            )
        linha = {"doc_uid": uid,
                 "vernacular_score": validar_srsp(doc["vernacular_score"], uid, escala),
                 "scribe_mediation_likely": mediacao}
        for origem, coluna in CAMPOS_SOCIO.items():
            if coluna in linha:
                continue
            valor = doc.get(origem)
            linha[coluna] = None if valor is None else str(valor).strip() or None
        linhas.append(linha)
    colunas = ["doc_uid", *CAMPOS_SOCIO.values()]
    df = pd.DataFrame(linhas, columns=colunas)
    df["vernacular_score"] = df["vernacular_score"].astype("Int64")
    df["scribe_mediation_likely"] = df["scribe_mediation_likely"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Ligação com a base: cada documento da entrada recebe o doc_uid do WP 1.1
# ---------------------------------------------------------------------------

def uids_da_entrada(docs: list[dict], pasta_colecao: Path) -> list[str]:
    """Casa cada documento da entrada com o seu doc_uid pelo hash de conteúdo do registro."""
    arq_registro = pasta_colecao / "registro_ids.json"
    arq_base = pasta_colecao / "documentos.parquet"
    for arq in (arq_registro, arq_base):
        if not arq.exists():
            raise ErroImportacao(f"{arq} não existe. Rode build/importar_base.py antes (WP 1.1).")
    with open(arq_registro, encoding="utf-8") as f:
        registro = json.load(f)
    da_base = set(pd.read_parquet(arq_base, columns=["doc_uid"])["doc_uid"].astype(str))

    por_hash: dict[str, list[str]] = {}
    for r in registro.get("documentos", []):
        if r["doc_uid"] in da_base:
            por_hash.setdefault(r["hash_conteudo"], []).append(r["doc_uid"])

    uids: list[str] = []
    for i, doc in enumerate(docs):
        candidatos = por_hash.get(hash_conteudo(doc))
        if not candidatos:
            raise ErroImportacao(
                f"Documento {i} da entrada não está na base ({doc.get('reference_code')!r}). "
                "A entrada mudou desde o WP 1.1: rode build/importar_base.py de novo."
            )
        uids.append(candidatos.pop(0))

    if len(uids) != len(da_base):
        raise ErroImportacao(
            f"A entrada tem {len(uids)} documentos e a base {len(da_base)}. "
            "Rode build/importar_base.py de novo."
        )
    return uids


# ---------------------------------------------------------------------------
# Escrita das camadas e do manifesto
# ---------------------------------------------------------------------------

def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _contagens(df: pd.DataFrame, coluna: str) -> dict:
    serie = df[coluna].dropna()
    return {str(k): int(v) for k, v in serie.value_counts().sort_index().items()}


def mesclar_colecoes(pasta: Path, arquivo_dados: str, df: pd.DataFrame) -> pd.DataFrame:
    """Guarda as linhas das outras coleções que já estavam na camada (D6, A11).

    Cada execução cobre uma coleção. Sem isto, importar `teste-mini` apagaria `ahu-sul` da camada.
    A ordem das linhas não importa: o `construir_app.py` junta tudo pela ordem da base.
    """
    arq = pasta / arquivo_dados
    if not arq.exists():
        return df
    anterior = pd.read_parquet(arq)
    de_outras = anterior[~anterior["doc_uid"].isin(set(df["doc_uid"]))]
    if de_outras.empty:
        return df
    return pd.concat([df, de_outras], ignore_index=True)


def colecoes_cobertas(pasta: Path, colecao: str) -> list[str]:
    """As coleções que a camada passa a cobrir: as de antes mais a desta execução."""
    anteriores = []
    arq = pasta / "camada.json"
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            anteriores = list(json.load(f).get("colecoes") or [])
    return sorted({*anteriores, colecao})


def gravar_camada(pasta: Path, arquivo_dados: str, df: pd.DataFrame, camada: dict) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pasta / arquivo_dados, index=False)
    camada = {
        **camada,
        "documentos": int(len(df)),
        "esquema": {c: str(t) for c, t in df.dtypes.items()},
        "data": _agora(),
    }
    with open(pasta / "camada.json", "w", encoding="utf-8") as f:
        json.dump(camada, f, ensure_ascii=False, indent=2)
    return camada


def registrar_no_manifesto(corpus: Path, pasta: Path, camada: dict, arquivos: tuple[str, ...]) -> None:
    arq = corpus / "manifest.json"
    manifesto = {"versao_corpus": "0.1.0", "colecoes": {}, "camadas": {}}
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            manifesto = json.load(f)
    manifesto.setdefault("camadas", {})[pasta.name] = {
        "tipo": camada["tipo"],
        "versao": camada["versao"],
        "documentos": camada["documentos"],
        "arquivos": {f"camadas/{pasta.name}/{n}": sha256_arquivo(pasta / n) for n in arquivos},
        "ativa": True,
    }
    manifesto["gerado_em"] = camada["data"]
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)


def importar_camadas(
    entrada: Path, corpus: Path, colecao: str = "ahu-sul", prompt: str | None = None,
    so: str | None = None,
) -> dict:
    entrada, corpus = Path(entrada), Path(corpus)
    if not entrada.exists():
        raise ErroImportacao(f"Arquivo de entrada não encontrado: {entrada}")
    if entrada.suffix == ".parquet":
        # Uma base já com doc_uid (ex.: corpus/colecoes/ahu-brasil/documentos.parquet). Só a
        # tipologia sai dela: a sociolinguística precisa dos campos do DeepSeek.
        if so != "tipologia":
            raise ErroImportacao("Com uma entrada .parquet, use --so tipologia.")
        tabela = pd.read_parquet(entrada)
        uids = tabela["doc_uid"].astype(str).tolist()
        docs = tabela.drop(columns=["doc_uid"]).astype(object).where(tabela.notna(), None
                                                                   ).to_dict("records")
        # Sem `typology_status` no catálogo: é canônica a tipologia que já é canônica na camada.
        existente = corpus / "camadas" / CAMADA_TIPOLOGIA / "tipologia.parquet"
        canonicas = set()
        if existente.exists():
            t = pd.read_parquet(existente, columns=["tipologia", "typology_status"])
            canonicas = set(t.loc[t["typology_status"], "tipologia"].dropna())
        for d in docs:
            if d.get("typology_status") is None:
                d["typology_status"] = normalizar_tipologia(d.get("extracted_typology")) in canonicas
    else:
        try:
            with open(entrada, encoding="utf-8") as f:
                docs = json.load(f)
        except json.JSONDecodeError as e:
            raise ErroImportacao(f"JSON inválido ou truncado em {entrada}: {e}") from e
        if not isinstance(docs, list):
            raise ErroImportacao("O catálogo de entrada deve ser uma lista de documentos.")
        uids = uids_da_entrada(docs, corpus / "colecoes" / colecao)
    processo = {"script": "build/importar_camadas.py", "versao_script": VERSAO_SCRIPT,
                "commit": commit_atual()}
    proveniencia_entrada = {"arquivo": entrada.name, "sha256": sha256_arquivo(entrada)}
    resultado: dict = {"documentos": len(uids), "camadas": {}}

    if so in (None, "tipologia"):
        arq_correcoes = corpus / "correcoes_tipologia.json"
        correcoes = carregar_correcoes(arq_correcoes)
        pasta = corpus / "camadas" / CAMADA_TIPOLOGIA
        df = mesclar_colecoes(pasta, "tipologia.parquet", montar_tipologia(docs, uids, correcoes))
        camada = gravar_camada(pasta, "tipologia.parquet", df, {
            "nome": CAMADA_TIPOLOGIA,
            "tipo": "tipologia",
            "versao": "1",
            "colecoes": colecoes_cobertas(pasta, colecao),
            "origem_do_campo": "extracted_typology e typology_status do catálogo (preprint §1, Bellotto)",
            "regra_normalizacao": REGRA_TIPOLOGIA,
            "regra_status": "typology_status do catálogo, ou a tipologia corrigida já ser canônica "
                            "em outro documento. O vocabulário canônico sai dos dados.",
            "correcoes": {"arquivo": arq_correcoes.name, "tabela": correcoes,
                          "aplicadas": int(df["tipologia_corrigida"].sum())},
            "tipologias_distintas": int(df["tipologia"].nunique(dropna=True)),
            "canonicas": int(df["typology_status"].sum()),
            "a_revisar": int((~df["typology_status"]).sum()),
            "contagem_por_tipologia": _contagens(df, "tipologia"),
            "processo": processo,
            "entrada": proveniencia_entrada,
        })
        registrar_no_manifesto(corpus, pasta, camada, ("tipologia.parquet", "camada.json"))
        resultado["camadas"][CAMADA_TIPOLOGIA] = {
            "distintas": camada["tipologias_distintas"], "canonicas": camada["canonicas"],
            "a_revisar": camada["a_revisar"], "corrigidas": camada["correcoes"]["aplicadas"],
        }

    if so in (None, "sociolinguistica"):
        pasta = corpus / "camadas" / CAMADA_SOCIO
        cobertas = colecoes_cobertas(pasta, colecao)
        escala = ler_escala(pasta)
        df = mesclar_colecoes(pasta, "sociolinguistica.parquet",
                              montar_sociolinguistica(docs, uids, escala))
        anterior = pasta / "camada.json"
        if prompt is None and anterior.exists():  # não perde um prompt já gravado
            with open(anterior, encoding="utf-8") as f:
                guardado = json.load(f).get("prompt")
            prompt = guardado if guardado and guardado != PROMPT_PENDENTE else None
        camada = gravar_camada(pasta, "sociolinguistica.parquet", df, {
            "nome": CAMADA_SOCIO,
            "tipo": "sociolinguistica",
            "versao": "3",
            "colecoes": cobertas,
            "modelo": MODELO_SOCIO,
            "prompt": prompt or PROMPT_PENDENTE,
            "campos_de_origem": CAMPOS_SOCIO,
            "escala_srsp": escala,
            "escala_srsp_nota": f"inteira de {escala[0]} a {escala[1]}, como no dado; nunca "
                                "dividida por 10. É esta escala que a importação exige: outra "
                                "classificação é outra camada, com a sua própria escala.",
            "auditoria": AUDITORIA_SRSP,
            "uso_restrito": {
                "scribe_mediation_likely": "só o dossiê em PDF (D1); nunca interface, filtros, CSV ou JSON",
            },
            "contagem_por_srsp": _contagens(df, "vernacular_score"),
            "contagem_por_categoria": _contagens(df, "sender_category"),
            "contagem_por_vetor": _contagens(df, "vector"),
            "mediacao_provavel": int(df["scribe_mediation_likely"].sum()),
            "processo": processo,
            "entrada": proveniencia_entrada,
        })
        registrar_no_manifesto(corpus, pasta, camada, ("sociolinguistica.parquet", "camada.json"))
        resultado["camadas"][CAMADA_SOCIO] = {
            "srsp_medio": round(float(df["vernacular_score"].mean()), 3),
            "mediacao_provavel": camada["mediacao_provavel"],
            "prompt_pendente": camada["prompt"] == PROMPT_PENDENTE,
        }

    return resultado


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(ENTRADA_PADRAO))
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    ap.add_argument("--colecao", default="ahu-sul")
    ap.add_argument("--prompt", default=None,
                    help="arquivo de texto com o prompt do preprint §2 (guardado no camada.json)")
    ap.add_argument("--so", choices=["tipologia", "sociolinguistica"], default=None,
                    help="construir só uma das camadas")
    args = ap.parse_args()

    prompt = None
    if args.prompt:
        caminho = Path(args.prompt)
        if not caminho.exists():
            print(f"ERRO: arquivo de prompt não encontrado: {caminho}", file=sys.stderr)
            return 1
        prompt = caminho.read_text(encoding="utf-8").strip()

    try:
        r = importar_camadas(Path(args.entrada), Path(args.corpus), args.colecao, prompt, args.so)
    except ErroImportacao as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1

    print(f"{r['documentos']} documentos.")
    for nome, info in r["camadas"].items():
        print(f"  {nome}: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    if r["camadas"].get(CAMADA_SOCIO, {}).get("prompt_pendente"):
        print("ATENÇÃO: o prompt do preprint §2 ainda não foi gravado. "
              "Rode de novo com --prompt <arquivo.txt>.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
