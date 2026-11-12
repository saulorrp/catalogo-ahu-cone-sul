#!/usr/bin/env bash
# scripts/reconstruir.sh — Refaz app_dados/ a partir de corpus/ e confere com manifesto.
# Uso: scripts/reconstruir.sh [PYTHON]  (padrão: python3)

set -euo pipefail

PY="${1:-python3}"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

TMP="$(mktemp -d)"
CRIOU_NPY=0

# Determina a camada ativa de embeddings a partir do manifesto publicado
CAMADA="$("$PY" -c '
import json, sys
try:
    with open("app_dados/manifest.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    c = m.get("corpus", {}).get("camadas", {}).get("embeddings")
    if not c:
        sys.exit("Camada de embeddings ausente em app_dados/manifest.json")
    print(c)
except Exception as err:
    sys.exit(f"Erro ao ler app_dados/manifest.json: {err}")
')"

NPY="$RAIZ/corpus/camadas/$CAMADA/embeddings.f16.npy"

# Se o arquivo já existe no corpus, abortamos para não destruir dados existentes
if [ -e "$NPY" ]; then
    echo "Erro: $NPY já existe. Abortando para evitar sobrescrita." >&2
    rm -rf "$TMP"
    exit 1
fi

limpar() {
    if [ "$CRIOU_NPY" -eq 1 ] && [ -f "$NPY" ]; then
        rm -f "$NPY"
    fi
    rm -rf "$TMP"
}
trap limpar EXIT

# 2. Concatena partes de app_dados/embeddings e reordena para a ordem do corpus
echo "Reconstruindo $NPY a partir das partes de app_dados/embeddings..."
# Marcado antes de gravar: um .npy pela metade também tem de ser apagado na saída.
CRIOU_NPY=1
"$PY" - "$CAMADA" "$NPY" << 'EOF'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

camada = sys.argv[1]
caminho_npy = Path(sys.argv[2])
app_dados = Path("app_dados")

partes_caminhos = sorted((app_dados / "embeddings").glob("parte_*.npy"))
if not partes_caminhos:
    sys.exit("Nenhuma parte encontrada em app_dados/embeddings/parte_*.npy")

partes = [np.load(p, mmap_mode="r") for p in partes_caminhos]
total_linhas = sum(p.shape[0] for p in partes)

docs = pd.read_parquet(app_dados / "documentos.parquet", columns=["doc_uid"])
if total_linhas != len(docs):
    sys.exit(f"Desalinhamento: {total_linhas} vetores vs {len(docs)} documentos.")

indice_path = Path("corpus/camadas") / camada / "indice.parquet"
if not indice_path.exists():
    sys.exit(f"Arquivo de índice ausente: {indice_path}")

indice = pd.read_parquet(indice_path, columns=["doc_uid"])
uid_para_linha = {uid: i for i, uid in enumerate(docs["doc_uid"])}

indices_origem = []
for uid in indice["doc_uid"]:
    pos = uid_para_linha.get(uid)
    if pos is None:
        sys.exit(f"UID do índice ausente em documentos.parquet: {uid}")
    indices_origem.append(pos)

vetores_app = np.concatenate(partes, axis=0)
vetores_reordenados = vetores_app[np.array(indices_origem, dtype=np.int64)]

caminho_npy.parent.mkdir(parents=True, exist_ok=True)
np.save(caminho_npy, vetores_reordenados.astype(np.float16))
EOF

# 3. Reconstrói app_dados na pasta temporária
echo "Executando build/construir_app.py para $TMP/app_dados..."
"$PY" build/construir_app.py --corpus corpus --saida "$TMP/app_dados"

# 4. Reconstrói modelo estático caso o meta.json exista
META_ESTATICO="app_dados/estatico/meta.json"
if [ -f "$META_ESTATICO" ]; then
    if ! "$PY" -c "import model2vec" 2>/dev/null; then
        echo "estático não reconstruído (pip install model2vec==0.10.0)"
    else
        MODELO="$("$PY" -c '
import json; m=json.load(open("app_dados/estatico/meta.json")); print(m.get("modelo", ""))
')"
        REVISAO="$("$PY" -c '
import json; m=json.load(open("app_dados/estatico/meta.json")); print(m.get("revisao", ""))
')"
        if [ -n "$MODELO" ] && [ -n "$REVISAO" ]; then
            echo "Reconstruindo estático (modelo=$MODELO, revisao=$REVISAO)..."
            "$PY" build/construir_estatico.py \
                --app-dados "$TMP/app_dados" \
                --modelo "$MODELO" \
                --revisao "$REVISAO"
        fi
    fi
fi

# 5. Compara publicado com reconstruído e gera tabela
"$PY" - "app_dados" "$TMP/app_dados" "${RELATORIO:-}" << 'EOF'
import hashlib
import json
import sys
from pathlib import Path

pasta_pub = Path(sys.argv[1])
pasta_rec = Path(sys.argv[2])
caminho_relatorio = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while pedaco := f.read(1024 * 1024):
            h.update(pedaco)
    return h.hexdigest()

def conteudo_sqlite(caminho: Path) -> str:
    import sqlite3
    h = hashlib.sha256()
    con = sqlite3.connect(f"{caminho.resolve().as_uri()}?mode=ro", uri=True)
    for linha in con.iterdump():
        h.update(linha.encode("utf-8"))
    con.close()
    return h.hexdigest()

with open(pasta_pub / "manifest.json", "r", encoding="utf-8") as f:
    pub_manifest = json.load(f)

manifest_rec_path = pasta_rec / "manifest.json"
rec_manifest = json.load(open(manifest_rec_path, "r", encoding="utf-8")) if manifest_rec_path.exists() else {}

linhas_tabela = []
total_itens = 0
batem_itens = 0

# (a) Cada artefato presente em manifest["artefatos"]
artefatos = pub_manifest.get("artefatos", {})
for rel_path, pub_hash in sorted(artefatos.items()):
    total_itens += 1
    rec_path = pasta_rec / rel_path
    if not rec_path.exists():
        status = "FALTA"
        rec_hash_str = "-"
    else:
        rec_hash = sha256_arquivo(rec_path)
        status = "BATE" if rec_hash == pub_hash else "DIFERE"
        rec_hash_str = rec_hash[:8] + "..."
        # O SQLite grava bytes que dependem da versão da biblioteca e da ordem das páginas: se o
        # arquivo difere, confere o conteúdo lógico (as tabelas, linha a linha) antes de concluir.
        if status == "DIFERE" and rel_path.endswith(".sqlite"):
            if conteudo_sqlite(rec_path) == conteudo_sqlite(pasta_pub / rel_path):
                status = "DIFERE (bytes; conteúdo igual)"
    if status == "BATE":
        batem_itens += 1
    linhas_tabela.append((rel_path, pub_hash[:8] + "...", rec_hash_str, status))

# (b) Arquivos de estatico/* que não entram no manifesto
dir_estatico_pub = pasta_pub / "estatico"
if dir_estatico_pub.is_dir():
    for f_pub in sorted(dir_estatico_pub.rglob("*")):
        if f_pub.is_file():
            rel_path = f_pub.relative_to(pasta_pub).as_posix()
            total_itens += 1
            pub_hash = sha256_arquivo(f_pub)
            f_rec = pasta_rec / rel_path
            if not f_rec.exists():
                status = "FALTA"
                rec_hash_str = "-"
            else:
                rec_hash = sha256_arquivo(f_rec)
                status = "BATE" if rec_hash == pub_hash else "DIFERE"
                rec_hash_str = rec_hash[:8] + "..."
            if status == "BATE":
                batem_itens += 1
            linhas_tabela.append((rel_path, pub_hash[:8] + "...", rec_hash_str, status))

# (c) Metadados específicos do manifesto
campos_meta = [
    "doc_uids_sha256",
    "pessoa_uids_sha256",
    "contagens",
    "acertos",
    "modelo_embeddings",
    "lexico",
    "srsp",
    "modelos",
]
for campo in campos_meta:
    if campo in pub_manifest:
        total_itens += 1
        val_pub = pub_manifest.get(campo)
        if campo not in rec_manifest:
            status = "FALTA"
            s_pub = (val_pub if isinstance(val_pub, str) else json.dumps(val_pub, sort_keys=True))[:12]
            s_rec = "-"
        else:
            val_rec = rec_manifest.get(campo)
            status = "BATE" if val_rec == val_pub else "DIFERE"
            s_pub = (val_pub if isinstance(val_pub, str) else json.dumps(val_pub, sort_keys=True))[:12]
            s_rec = (val_rec if isinstance(val_rec, str) else json.dumps(val_rec, sort_keys=True))[:12]
        if status == "BATE":
            batem_itens += 1
        linhas_tabela.append((f"[meta] {campo}", s_pub, s_rec, status))

largura_arq = max(len(l[0]) for l in linhas_tabela) if linhas_tabela else 30
largura_arq = max(largura_arq, len("arquivo"))
cabecalho = f"{'arquivo':<{largura_arq}}  {'publicado':<12}  {'reconstruído':<12}  status"
separador = "-" * len(cabecalho)

saida = [cabecalho, separador]
for arq, pub, rec, st in linhas_tabela:
    saida.append(f"{arq:<{largura_arq}}  {pub:<12}  {rec:<12}  {st}")
saida.append(separador)
resumo = f"{batem_itens} de {total_itens} batem"
saida.append(resumo)

relatorio_final = "\n".join(saida) + "\n"
sys.stdout.write(relatorio_final)

if caminho_relatorio:
    p_rel = Path(caminho_relatorio)
    p_rel.parent.mkdir(parents=True, exist_ok=True)
    with open(p_rel, "w", encoding="utf-8") as f:
        f.write(relatorio_final)

sys.exit(0 if (batem_itens == total_itens and total_itens > 0) else 1)
EOF
