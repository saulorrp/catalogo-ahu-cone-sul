"""construir_indice.py — WP 1.4: índice semântico reprodutível (substitui o ahu_semantic_index.pkl).

Entrada : corpus/colecoes/<colecao>/documentos.parquet   (WP 1.1)
Saída   : corpus/camadas/<nome_da_camada>/
              embeddings.f16.npy   linha i = documento i de indice.parquet, normalizado, float16
              indice.parquet       doc_uid + texto_sha1 por linha (os uids ficam JUNTOS dos vetores)
              camada.json          modelo, prefixos, receita do texto, versões, hashes, sanidade

Receita do texto (preprint §3):  "passage: <tipologia>. <remetente>. <pasta>. <descrição>"
A consulta usa o prefixo "query: " (feito pelo app, ahu/busca.py). Vetores L2-normalizados:
cosseno = produto escalar.

Roda em CPU, em lotes, com retomada (R3): se for interrompido, rode o mesmo comando de novo.
Os vetores parciais ficam em corpus/camadas/<camada>/parciais/ e são apagados ao concluir.

Uso:
    python build/construir_indice.py                      # e5-large, coleção ahu-sul
    python build/construir_indice.py --limite 200         # teste rápido (camada *_amostra)
    python build/construir_indice.py --modelo intfloat/multilingual-e5-base   # WP 4.2
    python build/construir_indice.py --forcar             # refaz mesmo que esteja atualizado
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from build.importar_base import commit_atual, sha256_arquivo  # noqa: E402

VERSAO_SCRIPT = "1.0.0"
MODELO_PADRAO = "intfloat/multilingual-e5-large"
PREFIXO_DOCUMENTO = "passage: "
PREFIXO_CONSULTA = "query: "
MAX_TOKENS = 512
TAMANHO_CHUNK = 128
RECEITA_TEXTO = (
    'passage: <extracted_typology>. <sender_name>. <folder>. <description> '
    "(partes vazias são omitidas; separador '. ')"
)

Encoder = Callable[[list[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Texto do documento
# ---------------------------------------------------------------------------

def _limpo(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)) or valor is pd.NA:
        return ""
    return " ".join(str(valor).split())


def montar_texto(tipologia, remetente, pasta, descricao) -> str:
    """Texto do documento para o modelo, com o prefixo 'passage: '."""
    partes = [_limpo(tipologia), _limpo(remetente), _limpo(pasta), _limpo(descricao)]
    return PREFIXO_DOCUMENTO + ". ".join(p for p in partes if p)


def textos_da_base(df: pd.DataFrame) -> list[str]:
    return [
        montar_texto(t, s, f, d)
        for t, s, f, d in zip(df["extracted_typology"], df["sender_name"], df["folder"], df["description"])
    ]


def sha1(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def nome_camada_para(modelo: str) -> str:
    """e5-large (padrão) -> embeddings_e5_v1; e5-base -> embeddings_e5base_v1;
    e5-small -> embeddings_e5small_v1."""
    fim = modelo.rsplit("/", 1)[-1].lower()
    if fim.endswith("e5-large"):
        return "embeddings_e5_v1"
    for tamanho in ("base", "small"):
        if fim.endswith(f"e5-{tamanho}"):
            return f"embeddings_e5{tamanho}_v1"
    return "embeddings_" + "".join(c if c.isalnum() else "_" for c in fim) + "_v1"


# ---------------------------------------------------------------------------
# Codificação em lotes, com retomada
# ---------------------------------------------------------------------------

def _normalizar(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    normas = np.linalg.norm(emb, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return emb / normas


def _chave_chunk(uids: list[str], hashes: list[str]) -> str:
    return sha1("\n".join(f"{u}|{h}" for u, h in zip(uids, hashes)))


def construir_embeddings(
    textos: list[str],
    uids: list[str],
    hashes: list[str],
    encoder: Encoder,
    pasta_parciais: Path,
    tamanho_chunk: int = TAMANHO_CHUNK,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """Codifica em chunks gravados em disco. Chunks válidos de execuções anteriores são reaproveitados."""
    n = len(textos)
    pasta_parciais.mkdir(parents=True, exist_ok=True)
    partes: list[np.ndarray] = []
    inicio = time.time()
    codificados = 0
    total_chunks = (n + tamanho_chunk - 1) // tamanho_chunk

    for k in range(total_chunks):
        a, b = k * tamanho_chunk, min((k + 1) * tamanho_chunk, n)
        chave = _chave_chunk(uids[a:b], hashes[a:b])
        arq = pasta_parciais / f"chunk_{k:05d}.npz"
        reaproveitado = None
        if arq.exists():
            try:
                with np.load(arq, allow_pickle=False) as z:
                    if str(z["chave"]) == chave:
                        reaproveitado = z["emb"]
            except (OSError, ValueError, KeyError):
                reaproveitado = None  # arquivo truncado por uma interrupção: refaz o chunk
        if reaproveitado is not None:
            partes.append(reaproveitado)
            continue

        emb = _normalizar(encoder(textos[a:b])).astype(np.float16)
        if emb.shape[0] != b - a:
            raise RuntimeError(f"O codificador devolveu {emb.shape[0]} vetores para {b - a} textos.")
        tmp = arq.with_name(arq.name + ".tmp.npz")
        np.savez(tmp, chave=np.array(chave), emb=emb)
        tmp.replace(arq)  # gravação atômica: nunca fica um chunk pela metade
        partes.append(emb)
        codificados += b - a
        decorrido = time.time() - inicio
        faltam = n - b
        eta = decorrido / codificados * faltam if codificados else 0
        log(f"  {b}/{n} documentos  ({decorrido / 60:.1f} min decorridos, ~{eta / 60:.1f} min restantes)")

    return np.vstack(partes) if partes else np.zeros((0, 0), dtype=np.float16)


# ---------------------------------------------------------------------------
# Verificação de sanidade
# ---------------------------------------------------------------------------

def verificar_sanidade(
    df: pd.DataFrame, emb: np.ndarray, encoder: Encoder, quantidade: int = 5, tolerancia: float = 1e-3
) -> list[dict]:
    """Para `quantidade` documentos espaçados, a busca pela própria descrição deve devolver o documento em 1º.

    Empates (descrições idênticas) contam como acerto: o documento só precisa estar no topo da pontuação.
    """
    n = len(df)
    if n == 0:
        return []
    idx = sorted({int(i) for i in np.linspace(0, n - 1, min(quantidade, n))})
    consultas = [PREFIXO_CONSULTA + _limpo(df["description"].iloc[i]) for i in idx]
    q = _normalizar(encoder(consultas))
    notas = q @ emb.astype(np.float32).T
    resultado = []
    for linha, i in enumerate(idx):
        topo = int(np.argmax(notas[linha]))
        propria = float(notas[linha, i])
        resultado.append({
            "doc_uid": str(df["doc_uid"].iloc[i]), "linha": i, "primeiro": topo == i,
            "aceito": propria >= float(notas[linha, topo]) - tolerancia, "cosseno_proprio": round(propria, 4),
        })
    return resultado


# ---------------------------------------------------------------------------
# Escrita da camada e do manifesto
# ---------------------------------------------------------------------------

def versoes_bibliotecas() -> dict:
    from importlib import metadata

    versoes = {"numpy": np.__version__, "pandas": pd.__version__}
    for pacote in ("sentence-transformers", "torch"):
        try:
            versoes[pacote] = metadata.version(pacote)
        except metadata.PackageNotFoundError:  # ausente (ex.: nos testes com codificador falso)
            versoes[pacote] = None
    return versoes


def gravar_camada(
    pasta: Path, df: pd.DataFrame, emb: np.ndarray, hashes: list[str], modelo: str,
    colecoes: list[str], sanidade: list[dict], nome: str, amostra: bool,
) -> dict:
    pasta.mkdir(parents=True, exist_ok=True)
    arq_emb = pasta / "embeddings.f16.npy"
    np.save(arq_emb, emb.astype(np.float16))
    pd.DataFrame({"doc_uid": df["doc_uid"].astype(str).tolist(), "texto_sha1": hashes}).to_parquet(
        pasta / "indice.parquet", index=False
    )
    camada = {
        "nome": nome,
        "tipo": "embeddings",
        "versao": "1",
        "amostra": amostra,
        "colecoes": list(colecoes),
        "modelo": modelo,
        "prefixo_documento": PREFIXO_DOCUMENTO,
        "prefixo_consulta": PREFIXO_CONSULTA,
        "normalizado_l2": True,
        "dtype": "float16",
        "dimensao": int(emb.shape[1]) if emb.size else 0,
        "documentos": int(emb.shape[0]),
        "max_tokens": MAX_TOKENS,
        "receita_texto": RECEITA_TEXTO,
        "texto_global_sha1": sha1("".join(hashes)),
        "esquema": {"embeddings.f16.npy": "float16[n, dimensao], linha i = indice.parquet linha i",
                    "indice.parquet": "doc_uid, texto_sha1"},
        "processo": {
            "script": "build/construir_indice.py",
            "versao_script": VERSAO_SCRIPT,
            "commit": commit_atual(),
            "dispositivo": "cpu (padrão; usa GPU se disponível)",
        },
        "bibliotecas": versoes_bibliotecas(),
        "data": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sanidade": sanidade,
        "sanidade_aprovada": all(s["aceito"] for s in sanidade),
    }
    with open(pasta / "camada.json", "w", encoding="utf-8") as f:
        json.dump(camada, f, ensure_ascii=False, indent=2)
    return camada


def registrar_no_manifesto(corpus: Path, pasta: Path, camada: dict) -> None:
    arq = corpus / "manifest.json"
    manifesto = {"versao_corpus": "0.1.0", "colecoes": {}, "camadas": {}}
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            manifesto = json.load(f)
    arquivos = {
        f"camadas/{pasta.name}/{nome}": sha256_arquivo(pasta / nome)
        for nome in ("embeddings.f16.npy", "indice.parquet", "camada.json")
    }
    manifesto.setdefault("camadas", {})[pasta.name] = {
        "tipo": "embeddings", "versao": camada["versao"], "modelo": camada["modelo"],
        "documentos": camada["documentos"], "arquivos": arquivos,
        "ativa": not camada["amostra"] and camada["sanidade_aprovada"],
    }
    manifesto["gerado_em"] = camada["data"]
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, ensure_ascii=False, indent=2)


def indice_atualizado(pasta: Path, hashes: list[str], uids: list[str]) -> bool:
    arq_emb, arq_idx = pasta / "embeddings.f16.npy", pasta / "indice.parquet"
    if not (arq_emb.exists() and arq_idx.exists()):
        return False
    atual = pd.read_parquet(arq_idx)
    return atual["doc_uid"].tolist() == uids and atual["texto_sha1"].tolist() == hashes


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def criar_encoder(modelo: str, lote: int, dispositivo: str | None) -> Encoder:
    from sentence_transformers import SentenceTransformer  # import tardio: pesado

    m = SentenceTransformer(modelo, device=dispositivo)
    m.max_seq_length = MAX_TOKENS

    def encoder(textos: list[str]) -> np.ndarray:
        return m.encode(textos, batch_size=lote, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)

    return encoder


def colecoes_do_manifesto(corpus: Path) -> list[str]:
    """As coleções registradas, na ordem do manifesto. Saem dos dados, nunca do código (D6, A11)."""
    arq = corpus / "manifest.json"
    if not arq.exists():
        raise FileNotFoundError(f"{arq} não existe. Rode build/importar_base.py antes.")
    with open(arq, encoding="utf-8") as f:
        colecoes = list((json.load(f).get("colecoes") or {}))
    if not colecoes:
        raise FileNotFoundError(f"{arq} não tem nenhuma coleção. Rode build/importar_base.py antes.")
    return colecoes


def construir_indice(
    corpus: Path, colecao: str | list[str] | None, modelo: str, encoder: Encoder,
    limite: int | None = None, forcar: bool = False, manter_parciais: bool = False,
    tamanho_chunk: int = TAMANHO_CHUNK, log: Callable[[str], None] = print,
) -> dict:
    """Indexa uma coleção, várias, ou (com `colecao=None`) todas as do manifesto, nessa ordem."""
    if colecao is None:
        colecoes = colecoes_do_manifesto(corpus)
    elif isinstance(colecao, str):
        colecoes = [colecao]
    else:
        colecoes = list(colecao)

    partes = []
    for nome_colecao in colecoes:
        arq_base = corpus / "colecoes" / nome_colecao / "documentos.parquet"
        if not arq_base.exists():
            raise FileNotFoundError(
                f"Base não encontrada: {arq_base}. Rode build/importar_base.py antes."
            )
        partes.append(pd.read_parquet(
            arq_base,
            columns=["doc_uid", "extracted_typology", "sender_name", "folder", "description"],
        ))
    df = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]
    amostra = limite is not None
    if amostra:
        df = df.iloc[:limite].reset_index(drop=True)

    nome = nome_camada_para(modelo) + ("_amostra" if amostra else "")
    pasta = corpus / "camadas" / nome
    uids = df["doc_uid"].astype(str).tolist()
    textos = textos_da_base(df)
    hashes = [sha1(t) for t in textos]

    if not forcar and indice_atualizado(pasta, hashes, uids):
        log(f"O índice {nome} já está atualizado ({len(uids)} documentos). Use --forcar para refazer.")
        return {"nome": nome, "atualizado": False, "documentos": len(uids)}

    log(f"Codificando {len(textos)} documentos com {modelo} (camada {nome})...")
    emb = construir_embeddings(textos, uids, hashes, encoder, pasta / "parciais", tamanho_chunk, log)
    sanidade = verificar_sanidade(df, emb, encoder)
    camada = gravar_camada(pasta, df, emb, hashes, modelo, colecoes, sanidade, nome, amostra)
    registrar_no_manifesto(corpus, pasta, camada)
    if not manter_parciais:
        shutil.rmtree(pasta / "parciais", ignore_errors=True)

    for s in sanidade:
        res = 'OK' if s['aceito'] else 'FALHOU'
        log(f"  sanidade {s['doc_uid']}: {res} (cosseno {s['cosseno_proprio']})")
    return {
        "nome": nome,
        "atualizado": True,
        "documentos": len(uids),
        "sanidade_aprovada": camada["sanidade_aprovada"],
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(RAIZ / "corpus"))
    ap.add_argument("--colecao", action="append", default=None,
                    help="coleção a indexar; pode repetir. Padrão: todas as do manifesto")
    ap.add_argument("--modelo", default=MODELO_PADRAO)
    ap.add_argument("--lote", type=int, default=16, help="documentos por lote do modelo (memória)")
    ap.add_argument("--dispositivo", default=None, help="cpu, cuda... (padrão: automático)")
    ap.add_argument(
        "--limite", type=int, default=None, help="só os N primeiros (teste; grava camada *_amostra)"
    )
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--manter-parciais", action="store_true")
    args = ap.parse_args()

    try:
        encoder = criar_encoder(args.modelo, args.lote, args.dispositivo)
        r = construir_indice(Path(args.corpus), args.colecao, args.modelo, encoder, args.limite,
                             args.forcar, args.manter_parciais)
    except FileNotFoundError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    if r.get("atualizado") and not r.get("sanidade_aprovada", True):
        msg = "ATENÇÃO: a verificação de sanidade falhou; a camada não foi marcada como ativa."
        print(msg, file=sys.stderr)
        return 2
    print("Concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
