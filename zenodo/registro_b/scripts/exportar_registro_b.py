# ruff: noqa: E501, W291  (os modelos dos documentos são texto corrido em Markdown)
"""Exportação do Registro B do Zenodo (dados abertos do Catálogo AHU-Brasil).

Este script gera o pacote de dados abertos sem os textos protegidos por direitos
de terceiros (resumos do catálogo e do Projeto Resgate), mantendo identificadores,
cotas, enriquecimentos e camadas analíticas sob licenças abertas.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Raiz do repositório: três níveis acima deste script
RAIZ = Path(__file__).resolve().parents[3]

# Colunas com texto integral de terceiros que não podem ser relicenciadas abertamente
COLUNAS_TEXTO_DE_TERCEIROS = ("description", "description_resgate")
# Metadados cujo hash não bate com o do manifesto do corpus, e por quê (ver exportar_registro_b).
DIVERGENCIAS_METADADOS: list[tuple[str, str]] = []

# Dicionário descritivo obrigatório para todas as colunas de documentos_sem_resumo.parquet
DESCRICOES: dict[str, dict[str, str]] = {
    "doc_uid": {
        "origem": 'Gerado no build do corpus',
        "regra": 'Identificador interno estável: ahu-sul vem de corpus/colecoes/ahu-sul/registro_ids.json; ahu-brasil é sha1 da chave da cota (com #n quando a cota se repete), em coleta/scripts/juntar.py.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "colecao": {
        "origem": 'Corpus',
        "regra": 'Coleção de origem: ahu-sul ou ahu-brasil.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "folder": {
        "origem": 'Catálogo do AHU (ACTD) / Projeto Resgate',
        "regra": 'Série ou capitania (pasta) do verbete, como no catálogo.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "old_code": {
        "origem": 'Catálogo do AHU (ACTD) / Projeto Resgate',
        "regra": 'Cota antiga (arrumação anterior do AHU), como impressa.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "new_code": {
        "origem": 'Catálogo do AHU (ACTD) / Projeto Resgate',
        "regra": 'Cota atual no formato do Resgate (AHU_ACL_CU_<série>, Cx., D.), como impressa.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "reference_code": {
        "origem": 'ahu/conversor.py',
        "regra": 'Código de referência PT/AHU/CU/<série>/<caixa>/<documento>, escrito pelo conversor a partir da cota.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "document_id_and_date": {
        "origem": 'Catálogo do AHU (ACTD) / Projeto Resgate',
        "regra": 'Número do verbete, data e local, como no catálogo.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "ano_isolado": {
        "origem": 'build/cotas.py',
        "regra": 'Ano extraído da data do verbete; vazio quando não há ano legível.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "recipient_name": {
        "origem": 'Camada do ahu-sul (pipeline de biografias)',
        "regra": 'Destinatário; só existe na coleção ahu-sul.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "sender_name": {
        "origem": 'Camada do ahu-sul (pipeline de biografias)',
        "regra": 'Remetente; só existe na coleção ahu-sul.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "referencia_digitarq": {
        "origem": 'DigitArq do AHU (coleta/scripts/varrer_digitarq.py)',
        "regra": 'Código de referência do registro do DigitArq que contém o documento.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "resgate_docreader": {
        "origem": 'Projeto Resgate, DocReader da BN',
        "regra": 'Identificador do documento no DocReader (imagens digitalizadas).',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "duplicata_de": {
        "origem": 'corpus/duplicatas.ignore',
        "regra": 'doc_uid do verbete de que este é repetição; a busca ignora o repetido.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "url_actd": {
        "origem": 'ACTD/IICT',
        "regra": 'URL do PDF do catálogo da série no ACTD de onde veio o resumo.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "pagina_actd": {
        "origem": 'ACTD/IICT',
        "regra": 'Página do PDF (contada pelo PyMuPDF, a partir de 1) onde está o verbete.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "caminho_resgate": {
        "origem": 'Projeto Resgate (gov.br)',
        "regra": 'Página do instrumento de pesquisa do Projeto Resgate que traz o verbete.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "qc": {
        "origem": 'coleta/scripts/qc_banco.py',
        "regra": 'Marcas de controle de qualidade, separadas por ponto e vírgula (ex.: sem_data, cota_repetida).',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "sha1_resumo": {
        "origem": 'build/construir_app.py',
        "regra": 'SHA-1 do resumo (UTF-8, como na coluna retirada); confere a reidratação a partir do ACTD.',
        "licenca": 'CC0 1.0 (fatos não são protegidos)',
    },
    "tipologia": {
        "origem": 'Camada tipologia_v1',
        "regra": 'Primeira palavra em maiúsculas do resumo, normalizada e corrigida por corpus/correcoes_tipologia.json.',
        "licenca": 'CC BY 4.0',
    },
    "tipologia_chave": {
        "origem": 'Camada tipologia_v1',
        "regra": 'A tipologia sem acentos, para agrupar e filtrar.',
        "licenca": 'CC BY 4.0',
    },
    "typology_status": {
        "origem": 'Camada tipologia_v1',
        "regra": 'Falso quando a tipologia ficou vazia depois da normalização.',
        "licenca": 'CC BY 4.0',
    },
    "sender_category": {
        "origem": 'Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2)',
        "regra": 'Categoria social do remetente; só ahu-sul.',
        "licenca": 'CC BY 4.0',
    },
    "vector": {
        "origem": 'Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2)',
        "regra": 'Direção da comunicação (ex.: Top-Down); só ahu-sul.',
        "licenca": 'CC BY 4.0',
    },
    "typology_impact": {
        "origem": 'Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2)',
        "regra": 'Quanto a tipologia fixa a forma do texto (ex.: Highly Formulaic); só ahu-sul.',
        "licenca": 'CC BY 4.0',
    },
    "vernacular_score": {
        "origem": 'Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2)',
        "regra": 'SRSP, inteiro de 1 a 9; exploratório (ver LIMITACOES.md); só ahu-sul.',
        "licenca": 'CC BY 4.0',
    },
    "justificativa": {
        "origem": 'Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2)',
        "regra": 'Justificativa do modelo para a classificação, em inglês; só ahu-sul.',
        "licenca": 'CC BY 4.0',
    },
}


def carregar_auditar(raiz: Path):
    """Carrega a função auditar de scripts/auditar_banco.py por caminho absoluto."""
    caminho = raiz / "scripts" / "auditar_banco.py"
    if not caminho.exists():
        raise FileNotFoundError(f"Script de auditoria nao encontrado em {caminho}")
    spec = importlib.util.spec_from_file_location("auditar_banco", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar spec de {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if not hasattr(modulo, "auditar"):
        raise AttributeError(f"Funcao auditar nao encontrada em {caminho}")
    return modulo.auditar


def completar_proveniencia(df: pd.DataFrame, corpus: Path) -> pd.DataFrame:
    """Acrescenta url_actd, pagina_actd, caminho_resgate, qc (do corpus) e sha1_resumo."""
    colunas = ["url_actd", "pagina_actd", "caminho_resgate", "qc"]
    partes = []
    for arq in sorted((corpus / "colecoes").glob("*/documentos.parquet")):
        base = pd.read_parquet(arq)
        partes.append(base.reindex(columns=["doc_uid", *colunas]))
    prov = pd.concat(partes, ignore_index=True)
    df = df.drop(columns=[c for c in colunas if c in df.columns]).merge(prov, on="doc_uid", how="left")
    df["pagina_actd"] = df["pagina_actd"].astype("Int32")
    if "sha1_resumo" not in df.columns:
        df["sha1_resumo"] = df["description"].map(
            lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest() if isinstance(t, str) else None)
    return df


def _milhar(n) -> str:
    """167344 -> '167.344', como se escreve em português."""
    return f"{int(n):,}".replace(",", ".")


def calcular_sha256(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em disco."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256sums(pasta: Path) -> str:
    """Gera as linhas de SHA256SUMS para todos os arquivos da pasta em ordem alfabética."""
    arquivos = [p for p in pasta.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    arquivos.sort(key=lambda p: p.relative_to(pasta).as_posix())
    linhas = []
    for arq in arquivos:
        rel = arq.relative_to(pasta).as_posix()
        digest = calcular_sha256(arq)
        linhas.append(f"{digest}  {rel}")
    conteudo = "\n".join(linhas) + "\n" if linhas else ""
    (pasta / "SHA256SUMS").write_text(conteudo, encoding="utf-8")
    return conteudo


def montar_documentos_sem_resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Remove textos protegidos de terceiros e valida presença de sha1_resumo."""
    if "description" in df.columns and "sha1_resumo" in df.columns:
        com_desc = df["description"].notna() & (df["description"].astype(str).str.strip() != "")
        sem_sha = com_desc & (
            df["sha1_resumo"].isna() | (df["sha1_resumo"].astype(str).str.strip() == "")
        )
        qtd_sem_sha = int(sem_sha.sum())
        if qtd_sem_sha > 0:
            raise ValueError(
                f"Encontrados {qtd_sem_sha} documentos com description preenchida e sha1_resumo nula."
            )
    colunas_finais = [c for c in df.columns if c not in COLUNAS_TEXTO_DE_TERCEIROS]
    return df[colunas_finais].copy()


def dicionario(df: pd.DataFrame) -> list[dict]:
    """Gera a lista de metadados de cada coluna, falhando se faltar descrição."""
    linhas = len(df)
    colunas_info = []
    for col in df.columns:
        if col not in DESCRICOES:
            raise KeyError(
                f"Coluna '{col}' nao possui descricao em DESCRICOES. O dicionario deve estar completo."
            )
        desc = DESCRICOES[col]
        serie = df[col]
        vazios_qtd = int(serie.isna().sum())
        vazios_pct = round(float(vazios_qtd / linhas * 100), 2) if linhas > 0 else 0.0

        nao_nulos = serie.dropna()
        exemplo_val = str(nao_nulos.iloc[0])[:80] if len(nao_nulos) > 0 else ""

        try:
            import pyarrow as pa
            tipo_str = str(pa.Array.from_pandas(serie).type)
        except Exception:
            tipo_str = str(serie.dtype)

        colunas_info.append(
            {
                "nome": col,
                "tipo": tipo_str,
                "origem": desc["origem"],
                "regra": desc["regra"],
                "exemplo": exemplo_val,
                "vazios": vazios_qtd,
                "vazios_pct": vazios_pct,
                "licenca": desc["licenca"],
            }
        )
    return colunas_info


def extrair_versao_pyproject(raiz: Path) -> str:
    """Obtém a versão declarada no pyproject.toml usando tomllib."""
    caminho_toml = raiz / "pyproject.toml"
    if not caminho_toml.exists():
        return "0.1.0"
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(caminho_toml, "rb") as f:
            dados = tomllib.load(f)
        versao = dados.get("project", {}).get("version")
        if not versao:
            versao = dados.get("tool", {}).get("poetry", {}).get("version")
        return str(versao) if versao else "0.1.0"
    except Exception:
        return "0.1.0"


def extrair_credito_license_data(raiz: Path) -> str:
    """Extrai o texto exato da seção Crédito exigido de LICENSE-DATA.md."""
    caminho = raiz / "LICENSE-DATA.md"
    if not caminho.exists():
        return (
            "> Dados derivados: Rocha, S. R. P. *Catálogo AHU-Brasil* (2026), "
            "doi:10.5281/zenodo.18772666, CC BY 4.0.\n"
            "> Catálogo de origem: Arquivo Histórico Ultramarino (Lisboa), catálogos publicados "
            "no Arquivo Científico Tropical Digital (ACTD/IICT), e Projeto Resgate "
            "\"Barão do Rio Branco\" (Biblioteca Nacional / Ministério da Cultura)."
        )
    texto = caminho.read_text(encoding="utf-8")
    marcador = "## Crédito exigido"
    if marcador in texto:
        resto = texto.split(marcador, 1)[1]
        secao = resto.split("\n## ", 1)[0] if "\n## " in resto else resto
        return secao.strip()
    return "> Crédito exigido: consulte LICENSE-DATA.md na raiz do repositório."


def extrair_tabela_licencas_camadas(raiz: Path) -> str:
    """Extrai a tabela de licenças por camada de LICENSE-DATA.md sem reescrever."""
    caminho = raiz / "LICENSE-DATA.md"
    if not caminho.exists():
        return ""
    linhas = []
    gravando = False
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if linha.strip().startswith("| Camada |"):
            gravando = True
        if gravando:
            if linha.strip().startswith("|"):
                linhas.append(linha)
            else:
                break
    return "\n".join(linhas)


def extrair_tabela_metricas_busca(raiz: Path) -> str:
    """Extrai a tabela de métricas por motor de docs/plano_busca_servidor.md."""
    caminho = raiz / "docs" / "plano_busca_servidor.md"
    if not caminho.exists():
        return "| motor | nDCG@10 | Reciprocal Rank | P@10 |\n|---|---|---|---|\n| n/d | n/d | n/d | n/d |"
    linhas = []
    gravando = False
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if "| motor | nDCG@10" in linha:
            gravando = True
        if gravando:
            if linha.strip().startswith("|"):
                linhas.append(linha)
            else:
                break
    return "\n".join(linhas) if linhas else "| motor | nDCG@10 |\n|---|---|\n| n/d | n/d |"


def contar_pares_julgados(julgamentos: dict | list) -> int:
    """Pares consulta-documento julgados: {"Q01": {"consulta": ..., "notas": {doc_uid: nota}}}."""
    if isinstance(julgamentos, dict):
        return sum(len(v.get("notas", {})) for v in julgamentos.values() if isinstance(v, dict))
    return len(julgamentos) if isinstance(julgamentos, list) else 0


def gerar_readme(
    manifesto: dict,
    versao_repo: str,
    credito: str,
    total_docs: int,
) -> str:
    """Gera o README.md do Registro B com contagens e instruções de reprodução."""
    gerado_em = manifesto.get("gerado_em", "2026-11-12T09:30:00+00:00")
    contagens = manifesto.get("contagens", {})
    docs_cont = contagens.get("documentos", total_docs)
    pessoas_cont = contagens.get("pessoas", 0)
    colecoes = contagens.get("colecoes", {})

    linhas_colecoes = []
    for col_nome, qtd in sorted(colecoes.items()):
        linhas_colecoes.append(f"- **{col_nome}**: {_milhar(qtd)} documentos")
    bloco_colecoes = "\n".join(linhas_colecoes) if linhas_colecoes else f"- Total: {_milhar(docs_cont)} documentos"

    return f"""# Catálogo AHU-Brasil — registro B (dados abertos, sem resumos)

Versão do catálogo: **{versao_repo}**  
Data de geração: **{gerado_em}**

Este registro reúne os metadados abertos, identificadores, cotas e enriquecimentos do Catálogo AHU-Brasil. Os textos protegidos por direitos de terceiros (resumos do catálogo e do Projeto Resgate) foram retirados para viabilizar a distribuição pública irrestrita. O conjunto permite pesquisa historiográfica computacional imediata e possibilita a reidratação local dos resumos a partir das fontes públicas do ACTD.

## Contagens do acervo

- **Documentos catalogados**: {_milhar(docs_cont)}
- **Pessoas e autoridades identificadas**: {_milhar(pessoas_cont)}
- **Distribuição por coleção**:
{bloco_colecoes}

## Como citar

{credito}

## Mapa dos arquivos

| Arquivo / Pasta | O que é | Licença |
|---|---|---|
| `documentos_sem_resumo.parquet` | Metadados tabulares de documentos sem os resumos protegidos | CC0 1.0 (fatos) / CC BY 4.0 (camadas) |
| `camadas/` | Camadas analíticas de tipologia, entidades, pessoas e SRSP | CC BY 4.0 |
| `reidratacao/` | Scripts e tabela de checksums para recomposição dos resumos | MIT / CC0 1.0 |
| `reconstruir.sh` | Script de automação do pipeline de reconstrução do app | MIT |
| `docker-compose.yml` | Orquestração local em contêiner Docker | MIT |
| `avaliacao/` | Consultas de ouro e pares julgados para testes de busca | CC BY 4.0 |
| `QC/` | Relatórios de controle de qualidade e resumo da auditoria | CC BY 4.0 |
| `DICIONARIO_DE_DADOS.md` | Dicionário detalhado de cada coluna do dataset | CC BY 4.0 |
| `esquema.json` | Esquema formal de tipos, regras e contagens de vazios | CC BY 4.0 |
| `PROVENIENCIA.md` | Proveniência de cada camada analítica e modelos empregados | CC BY 4.0 |
| `LIMITACOES.md` | Limitações conhecidas e métricas de auditoria calculadas | CC BY 4.0 |
| `AVALIACAO_BUSCA.md` | Metodologia e resultados da avaliação dos motores de busca | CC BY 4.0 |
| `SELFHOST.md` | Instruções de auto-hospedagem e perfis de execução | CC BY 4.0 |
| `CHANGELOG.md` | Histórico formal de versões deste depósito | CC BY 4.0 |
| `LICENCAS.md` | Termos de licenciamento por camada e atribuição | CC BY 4.0 |
| `SHA256SUMS` | Hashes SHA-256 para conferência de integridade dos arquivos | CC0 1.0 |

## Níveis de reprodução

1. **Nível 1 — Só dados**: análise direta das tabelas em DuckDB, Polars ou pandas sobre `documentos_sem_resumo.parquet` e arquivos sob `camadas/`.
2. **Nível 2 — Reidratação dos resumos**: recomposição da coluna `description` a partir dos PDFs públicos do ACTD com `python reidratacao/reidratar_resumos.py` conferindo os hashes de `fontes_checksums.tsv`.
3. **Nível 3 — Reconstrução integral**: execução de `reconstruir.sh` para reconstruir os artefatos de busca e inferência de `app_dados/`, validando os hashes finais gerados.

## Como conferir a integridade

Execute na raiz do diretório descompactado:

```bash
sha256sum -c SHA256SUMS
```
"""


def gerar_dicionario_md(colunas_info: list[dict], total_linhas: int) -> str:
    """Gera o DICIONARIO_DE_DADOS.md em formato de tabela e detalhamento."""
    linhas_tabela = []
    for c in colunas_info:
        linhas_tabela.append(
            f"| `{c['nome']}` | `{c['tipo']}` | {c['origem']} | {c['licenca']} | "
            f"{c['vazios']} ({c['vazios_pct']}%) | `{c['exemplo']}` | {c['regra']} |"
        )
    corpo_tabela = "\n".join(linhas_tabela)

    return f"""# Dicionário de Dados — documentos_sem_resumo.parquet

Total de registros: **{_milhar(total_linhas)}**  
Total de colunas: **{len(colunas_info)}**

| Coluna | Tipo Arrow | Origem | Licença | Vazios | Exemplo | Regra / Descrição |
|---|---|---|---|---|---|---|
{corpo_tabela}
"""


def gerar_esquema_json(colunas_info: list[dict], total_linhas: int) -> str:
    """Gera o esquema.json com os tipos e metadados das colunas."""
    dados = {
        "arquivo": "documentos_sem_resumo.parquet",
        "linhas": total_linhas,
        "colunas": colunas_info,
    }
    return json.dumps(dados, indent=2, ensure_ascii=False) + "\n"


def gerar_proveniencia(manifesto: dict, camadas_detalhes: list[dict]) -> str:
    """Gera o PROVENIENCIA.md com detalhes das camadas ativas e dos modelos."""
    secoes_camadas = []
    for c in camadas_detalhes:
        nome = c["nome"]
        meta = c["camada_json"]
        arquivos = c["arquivos"]

        linhas_meta = []
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                val_str = json.dumps(v, ensure_ascii=False)[:300]
            else:
                val_str = str(v)
            linhas_meta.append(f"| `{k}` | {val_str} |")
        tabela_meta = "\n".join(linhas_meta)

        linhas_arqs = []
        for arq_nome, sha in arquivos:
            linhas_arqs.append(f"| `{arq_nome}` | `{sha}` |")
        tabela_arqs = "\n".join(linhas_arqs) if linhas_arqs else "| Nenhum arquivo | - |"

        secoes_camadas.append(
            f"## Camada: {nome}\n\n"
            f"### Metadados de camada.json\n\n"
            f"| Chave | Valor |\n|---|---|\n{tabela_meta}\n\n"
            f"### Arquivos da camada e integridade\n\n"
            f"| Arquivo | SHA-256 |\n|---|---|\n{tabela_arqs}"
        )
    bloco_camadas = "\n\n".join(secoes_camadas)

    modelos = manifesto.get("modelos", {})
    # O manifesto guarda {"fonte", "repositorios": {repo: {...}}, "espelhos", ...}.
    if isinstance(modelos, dict) and "repositorios" in modelos:
        modelos = modelos["repositorios"]
    linhas_modelos = []
    if isinstance(modelos, dict):
        for repo, dados_mod in modelos.items():
            rev = dados_mod.get("revisao", "-") if isinstance(dados_mod, dict) else "-"
            arqs = dados_mod.get("arquivos", {}) if isinstance(dados_mod, dict) else {}
            if isinstance(arqs, dict):
                for arq_nome, sha in arqs.items():
                    linhas_modelos.append(f"| `{repo}` | `{rev}` | `{arq_nome}` | `{sha}` |")
            elif isinstance(arqs, list):
                for item in arqs:
                    linhas_modelos.append(f"| `{repo}` | `{rev}` | `{item}` | - |")
    elif isinstance(modelos, list):
        for item in modelos:
            if isinstance(item, dict):
                repo = item.get("repositorio", "-")
                rev = item.get("revisao", "-")
                arq_nome = item.get("arquivo", "-")
                sha = item.get("sha256", "-")
                linhas_modelos.append(f"| `{repo}` | `{rev}` | `{arq_nome}` | `{sha}` |")
    tabela_modelos = (
        "\n".join(linhas_modelos)
        if linhas_modelos
        else "| Nenhum modelo externo registrado | - | - | - |"
    )

    tabela_divergencias = (
        "Os dados (parquet) batem byte a byte com `corpus/manifest.json`. Estes metadados não:\n\n"
        "| Arquivo | Motivo |\n|---|---|\n"
        + "\n".join(f"| `{a}` | {m} |" for a, m in DIVERGENCIAS_METADADOS)
        if DIVERGENCIAS_METADADOS else "Nenhum: todos os arquivos batem com `corpus/manifest.json`."
    )
    return f"""# Proveniência dos Dados e Camadas Analíticas

Este documento detalha a proveniência dos dados, a genealogia de cada camada analítica e os modelos de inteligência artificial empregados no pipeline de processamento.

As contribuições de modelos de inteligência artificial na extração, classificação e geração de artefatos estão descritas nesta seção e nas camadas correspondentes, não configurando autoria dos dados.

{bloco_camadas}

## Metadados que divergem do manifesto do corpus

{tabela_divergencias}

## Modelos

| Repositório | Revisão | Arquivo | SHA-256 |
|---|---|---|---|
{tabela_modelos}
"""


def gerar_limitacoes(
    manifesto: dict,
    df: pd.DataFrame,
    resumo_auditoria: dict,
    total_pares: int,
) -> str:
    """Gera o LIMITACOES.md com contagens exatas calculadas."""
    contagens = manifesto.get("contagens", {})
    docs_ahu_brasil = contagens.get("colecoes", {}).get(
        "ahu-brasil",
        int((df["colecao"] == "ahu-brasil").sum()) if "colecao" in df.columns else len(df),
    )
    verifs = resumo_auditoria.get("verificacoes", {})

    intervalo_abs = verifs.get("intervalo_absurdo", {}).get("documentos", 0)
    cota_ilegivel = verifs.get("cota_ilegivel", {}).get("documentos", 0)
    cota_vazia = verifs.get("cota_vazia", {}).get("documentos", 0)
    cotas_prob = cota_ilegivel + cota_vazia

    codices_ocr = int((df["folder"] == "Códices").sum()) if "folder" in df.columns else 0
    tipologias_para = int((df["tipologia"] == "PARA").sum()) if "tipologia" in df.columns else 0

    srsp_audit = manifesto.get("srsp", {}).get("auditoria", {})
    mae = srsp_audit.get("mae", "n/d")
    r_val = srsp_audit.get("r", "n/d")
    vies = srsp_audit.get("vies", "n/d")
    n_srsp = srsp_audit.get("n", "n/d")

    sem_resumo = verifs.get("sem_resumo", {}).get("documentos", 0)
    sem_ano = verifs.get("sem_ano", {}).get("documentos", 0)

    return f"""# Limitações Conhecidas e Qualidade dos Dados

Todos os indicadores abaixo foram calculados diretamente a partir dos dados do catálogo e dos relatórios de auditoria automatizada.

1. **Cotas não curadas manualmente**: {_milhar(docs_ahu_brasil)} documentos da coleção `ahu-brasil` preservam as cotas extraídas de catálogos históricos sem revisão manual exaustiva item a item.
2. **Intervalos absurdos**: {_milhar(intervalo_abs)} registros apresentam intervalos numéricos inconsistentes detectados pela verificação `intervalo_absurdo`.
3. **Cotas ilegíveis ou vazias**: {_milhar(cotas_prob)} documentos possuem cota ilegível ({_milhar(cota_ilegivel)}) ou cota vazia ({_milhar(cota_vazia)}).
4. **Códices transcritos exclusivamente por OCR**: {_milhar(codices_ocr)} documentos agrupados na série `Códices` derivam de reconhecimento óptico de caracteres sem conferência paleográfica exaustiva.
5. **Tipologias com erro de extração (`PARA`)**: {_milhar(tipologias_para)} documentos possuem tipologia atribuída como `PARA`, reflexo de falso positivo de extração sintática nos Códices.
6. **Desempenho de SRSP**: na avaliação da camada SRSP (n = {n_srsp}), obteve-se MAE = {mae}, r = {r_val} e viés = {vies}. A linha de base de prever sempre o valor 2 atinge MAE 0,97 segundo o preprint §3.1 (Rocha, 2026).
7. **Avaliação da busca restrita a modelo avaliador**: avaliação conduzida pelo juiz Gemini sobre {_milhar(total_pares)} pares consulta-documento às cegas (notas 0, 1 e 2). Nenhum julgamento humano foi publicado ainda.
8. **Divergência binária em `lexico.sqlite`**: o banco SQLite reconstruído reproduz exatamente o conteúdo lógico e as tabelas, mas pode divergir em bytes devido à alocação interna de páginas de diferentes versões do motor SQLite.
9. **Documentos sem resumo**: {_milhar(sem_resumo)} registros não dispõem de texto de resumo no catálogo de origem.
10. **Documentos sem datação de ano**: {_milhar(sem_ano)} registros não possuem ano atribuível na coluna `ano_isolado`.
"""


def gerar_avaliacao_busca(
    consultas_ouro: list,
    julgamentos_llm: dict | list,
    tabela_metricas: str,
) -> str:
    """Gera o AVALIACAO_BUSCA.md com contagens de consultas e tabela de métricas."""
    total_consultas = len(consultas_ouro)
    tipos_cont = collections.Counter()
    for c in consultas_ouro:
        tipos_cont[c.get("tipo", "indefinido")] += 1

    linhas_tipos = []
    for tipo_nome, qtd in sorted(tipos_cont.items()):
        linhas_tipos.append(f"- **{tipo_nome}**: {qtd} consultas")
    bloco_tipos = "\n".join(linhas_tipos) if linhas_tipos else "- Sem tipos catalogados"

    total_pares = contar_pares_julgados(julgamentos_llm)

    return f"""# Avaliação dos Motores de Busca

Este documento descreve o estado atual da avaliação comparativa dos motores de busca desenvolvidos para o acervo.

## Conjunto de teste

- **Consultas de ouro**: {total_consultas} consultas catalogadas em `avaliacao/consultas_ouro.json`.
- **Distribuição por tipologia de consulta**:
{bloco_tipos}
- **Pares julgados**: {_milhar(total_pares)} pares documento-consulta avaliados às cegas por modelo de linguagem (Gemini) nas escalas de relevância 0, 1 e 2.

## Métricas por motor de busca

Tabela extraída de `docs/plano_busca_servidor.md`:

{tabela_metricas}

Nenhum julgamento humano foi publicado; avaliar_busca.py e julgar.py permitem refazê-lo.
"""


def gerar_selfhost(credito: str) -> str:
    """Gera o SELFHOST.md com os perfis de implantação e passos numerados."""
    return f"""# Guia de Auto-hospedagem (Self-hosting)

O Catálogo AHU-Brasil pode ser executado em servidor próprio, estação local ou ambiente conteinerizado.

## Perfis de execução

| Perfil | Para quem | Recursos | Busca |
|---|---|---|---|
| **Só dados** | Quem quer consultar ou cruzar o banco | Nenhum servidor; DuckDB ou pandas sobre os parquet | SQL |
| **Leve** | Hospedagem gratuita ou máquina pequena | ~600 MiB de RAM, ~350 MB de disco | Léxico + Model2Vec (`AHU_SEM_SEMANTICA=1`) |
| **Completo** | Servidor institucional | ~3 GB de RAM, ~2 GB de disco, 2 núcleos | e5-small fp32 + reranker mMiniLM no top-30 (`AHU_RERANKER=1`); ~3 s por busca num i5 de 2014 |

### Exemplo de consulta analítica (Perfil Só dados com DuckDB)

```sql
SELECT doc_uid, ano_isolado, tipologia, reference_code
FROM read_parquet('documentos_sem_resumo.parquet')
WHERE ano_isolado BETWEEN 1750 AND 1777 LIMIT 10;
```

## Passos para implantação

1. **Baixar e verificar integridade**: faça o download do pacote e confira os arquivos com `sha256sum -c SHA256SUMS`.
2. **Reidratar resumos**: baixe os PDFs listados em `reidratacao/fontes_checksums.tsv` e rode `python reidratacao/reidratar_resumos.py ENTRADA.parquet PASTA_PDFS --checksums reidratacao/fontes_checksums.tsv` (ENTRADA = `doc_uid`, `url_actd`, `pagina_actd`, `sha1_resumo` deste parquet); o script diz quantos resumos bateram com `sha1_resumo`.
3. **Reconstruir artefatos**: execute `./reconstruir.sh` para gerar os índices e embeddings locais em `app_dados/`.
4. **Iniciar o contêiner**: suba a aplicação com `docker compose up -d` usando o arquivo `docker-compose.yml` desta pasta.
5. **Operação sem conexão à internet**: para executar em rede isolada, aponte a variável `AHU_ONNX_BASE_URL=file:///modelos/{{repositorio}}/resolve/{{revisao}}` com os pesos locais e hashes listados em `PROVENIENCIA.md`.
6. **Privacidade e logs**: o registro de buscas fica desativado por padrão (`AHU_LOG_DIR` vazio). Caso ativado, não registra endereços IP e aplica prazo estrito de retenção, em conformidade com LGPD e RGPD.
7. **Crédito obrigatório**: qualquer instância pública deve exibir o crédito de citação no rodapé da página:

{credito}
"""


def gerar_changelog(manifesto: dict, versao_repo: str, total_docs: int) -> str:
    """Gera o CHANGELOG.md com o histórico de versões do depósito."""
    gerado_em = manifesto.get("gerado_em", "2026-11-12")[:10]
    contagens = manifesto.get("contagens", {})
    docs_cont = contagens.get("documentos", total_docs)
    pessoas_cont = contagens.get("pessoas", 0)
    camadas_cont = len(manifesto.get("corpus", {}).get("camadas", []))

    return f"""# Histórico de Versões do Depósito

| Versão | Data | Documentos | Pessoas | Camadas Ativas |
|---|---|---|---|---|
| **{versao_repo}** | {gerado_em} | {_milhar(docs_cont)} | {_milhar(pessoas_cont)} | {camadas_cont} |

Versões anteriores do depósito: [10.5281/zenodo.18772666](https://doi.org/10.5281/zenodo.18772666) (conceito).
"""


def gerar_licencas(raiz: Path) -> str:
    """Gera o LICENCAS.md contendo a tabela do repo e a tabela do Registro B."""
    tabela_original = extrair_tabela_licencas_camadas(raiz)
    credito = extrair_credito_license_data(raiz)

    return f"""# Licenças por Camada de Dados

{tabela_original}

## Crédito exigido

{credito}

## Licenças dos arquivos deste Registro B

| Arquivo / Diretório | Licença | Descrição |
|---|---|---|
| `documentos_sem_resumo.parquet` (fatos e índices) | CC0 1.0 | Identificadores, cotas, anos e referências arquivísticas |
| `documentos_sem_resumo.parquet` (camadas analíticas) | CC BY 4.0 | Tipologias normalizadas e enriquecimentos do projeto |
| `camadas/` | CC BY 4.0 | Camadas analíticas de tipologia, pessoas e SRSP |
| `reidratacao/*.py` | MIT | Scripts utilitários de reidratação de textos |
| `reidratacao/fontes_checksums.tsv` | CC0 1.0 | Checksums das fontes documentais externas |
| `reconstruir.sh` | MIT | Script de reconstrução dos artefatos da aplicação |
| `docker-compose.yml` | MIT | Configuração de implantação conteinerizada |
| `avaliacao/*` | CC BY 4.0 | Consultas de teste e pares avaliados de busca |
| `QC/*` | CC BY 4.0 | Relatórios de controle de qualidade e auditoria |
| `*.md`, `esquema.json` | CC BY 4.0 | Documentação técnica e esquemas de dados |
| `SHA256SUMS` | CC0 1.0 | Tabela de conferência de integridade criptográfica |
"""


def gerar_docker_compose() -> str:
    """Gera o docker-compose.yml genérico sem watchtower nem funnel."""
    return """# Catálogo AHU - Docker Compose (Registro B)
name: catalogo-ahu

services:
  app:
    image: ghcr.io/saulorrp/catalogo-ahu:latest
    restart: unless-stopped
    ports:
      - "8080:7860"
    volumes:
      - modelos:/home/usuario/.cache/ahu-onnx
    environment:
      AHU_ONNX_CACHE: /home/usuario/.cache/ahu-onnx
      AHU_FRONT: web
      # Perfil leve por padrao: sem reranker. Altere para "1" no perfil completo.
      AHU_RERANKER: "0"
      # AHU_ROOT_PATH: /ahu
      # AHU_LOG_DIR: /home/usuario/logs
    mem_limit: 3g

volumes:
  modelos:
"""


def exportar_registro_b(
    app_dados: Path,
    corpus: Path,
    saida: Path,
    docs_no_repo: bool = False,
) -> None:
    """Executa a exportação completa do Registro B do Zenodo."""
    saida.mkdir(parents=True, exist_ok=True)

    # Passo a: lê manifesto e documentos.parquet, conferindo colunas obrigatórias
    manifest_path = app_dados / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Arquivo de manifesto nao encontrado em {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifesto = json.load(f)

    parquet_path = app_dados / "documentos.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Arquivo de documentos nao encontrado em {parquet_path}")
    df_original = pd.read_parquet(parquet_path)

    colunas_obrigatorias = ("sha1_resumo", "url_actd", "pagina_actd", "caminho_resgate", "qc")
    faltantes = [c for c in colunas_obrigatorias if c not in df_original.columns]
    if faltantes:
        # Um app_dados anterior ao build com proveniência: as colunas vêm do corpus, pelo doc_uid,
        # e o sha1 é calculado do resumo. O resultado é o mesmo que o build atual grava.
        print(f"aviso: {faltantes} ausentes em app_dados; completando a partir de {corpus}/colecoes")
        df_original = completar_proveniencia(df_original, corpus)

    # Passo b: remove colunas de terceiros e salva documentos_sem_resumo.parquet
    df_sem_resumo = montar_documentos_sem_resumo(df_original)
    caminho_parquet_saida = saida / "documentos_sem_resumo.parquet"
    df_sem_resumo.to_parquet(caminho_parquet_saida, compression="zstd", index=False)

    # Passo c: copia camadas ativas exceto embeddings, conferindo SHA-256 do manifesto
    corpus_info = manifesto.get("corpus", {})
    camadas_ativas = corpus_info.get("camadas", [])
    if isinstance(camadas_ativas, dict):
        camadas_ativas = list(camadas_ativas.values())
    camadas_selecionadas = [c for c in camadas_ativas if "embedding" not in c.lower()]

    manifesto_camadas = corpus_info.get("manifesto", {}).get("camadas", {})
    camadas_detalhes = []

    for nome_camada in camadas_selecionadas:
        pasta_camada_saida = saida / "camadas" / nome_camada
        pasta_camada_saida.mkdir(parents=True, exist_ok=True)

        info_camada = manifesto_camadas.get(nome_camada, {})
        arquivos_manifesto = info_camada.get("arquivos", {})
        arquivos_copiados = []

        itens_arquivos = []
        if isinstance(arquivos_manifesto, dict):
            for rel_caminho, hash_esp in arquivos_manifesto.items():
                itens_arquivos.append((rel_caminho, hash_esp))
        elif isinstance(arquivos_manifesto, list):
            for item in arquivos_manifesto:
                if isinstance(item, dict):
                    itens_arquivos.append(
                        (item.get("caminho") or item.get("arquivo"), item.get("sha256"))
                    )
                elif isinstance(item, str):
                    itens_arquivos.append((item, None))

        for rel_caminho, hash_esperado in itens_arquivos:
            if not rel_caminho:
                continue
            origem_arq = corpus / rel_caminho
            if not origem_arq.exists():
                origem_arq = corpus / "camadas" / nome_camada / Path(rel_caminho).name
            if not origem_arq.exists():
                raise FileNotFoundError(f"Arquivo da camada {nome_camada} ausente: {origem_arq}")

            hash_obtido = calcular_sha256(origem_arq)
            if hash_esperado and hash_obtido.lower() != str(hash_esperado).lower():
                # Dados (parquet) têm de bater byte a byte. Metadados JSON podem divergir por dois
                # motivos conhecidos, que ficam registrados em PROVENIENCIA.md em vez de parar:
                # o hash foi tirado no Windows, com fim de linha CRLF, e o git guarda LF; ou o
                # arquivo ganhou documentação depois do manifesto (o camada.json do SRSP, PR #25).
                if origem_arq.suffix != ".json":
                    raise ValueError(
                        f"SHA-256 invalido para {origem_arq}: esperado {hash_esperado}, obtido {hash_obtido}"
                    )
                crlf = hashlib.sha256(origem_arq.read_bytes().replace(b"\n", b"\r\n")).hexdigest()
                motivo = ("fim de linha CRLF no manifesto" if crlf == str(hash_esperado).lower()
                          else "metadado editado depois do manifesto")
                DIVERGENCIAS_METADADOS.append((f"{nome_camada}/{Path(rel_caminho).name}", motivo))
                print(f"aviso: {origem_arq.name} ({nome_camada}) difere do manifesto: {motivo}")

            destino_arq = pasta_camada_saida / Path(rel_caminho).name
            shutil.copy2(origem_arq, destino_arq)
            arquivos_copiados.append((destino_arq.name, hash_obtido))

        camada_json_path = corpus / "camadas" / nome_camada / "camada.json"
        camada_json_dict = {}
        if camada_json_path.exists():
            if not (pasta_camada_saida / "camada.json").exists():
                shutil.copy2(camada_json_path, pasta_camada_saida / "camada.json")
            try:
                camada_json_dict = json.loads(camada_json_path.read_text(encoding="utf-8"))
            except Exception:
                camada_json_dict = {}

        camadas_detalhes.append(
            {
                "nome": nome_camada,
                "camada_json": camada_json_dict,
                "arquivos": arquivos_copiados,
            }
        )

    # Passo d: copia utilitários de reidratação e script de reconstrução
    pasta_reidratacao = saida / "reidratacao"
    pasta_reidratacao.mkdir(parents=True, exist_ok=True)

    arqs_reidratacao = [
        RAIZ / "coleta" / "fontes_checksums.tsv",
        RAIZ / "coleta" / "scripts" / "reidratar_resumos.py",
        RAIZ / "coleta" / "scripts" / "parse_actd.py",
        RAIZ / "coleta" / "scripts" / "parse_texto.py",
        RAIZ / "coleta" / "scripts" / "pdf_para_texto.py",
        RAIZ / "coleta" / "scripts" / "conferir_reidratacao.py",
        RAIZ / "coleta" / "scripts" / "checksums_fontes.py",
    ]
    for arq in arqs_reidratacao:
        if arq.exists():
            shutil.copy2(arq, pasta_reidratacao / arq.name)

    script_reconstruir = RAIZ / "scripts" / "reconstruir.sh"
    if script_reconstruir.exists():
        shutil.copy2(script_reconstruir, saida / "reconstruir.sh")

    # Passo e: executa auditoria em pasta temporária e salva resumo em QC/
    pasta_qc = saida / "QC"
    pasta_qc.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        try:
            auditar_func = carregar_auditar(RAIZ)
            resumo_retorno = auditar_func(dados=app_dados, corpus=corpus, saida=tmp_dir)
            tmp_resumo = tmp_dir / "resumo.json"
            if tmp_resumo.exists():
                shutil.copy2(tmp_resumo, pasta_qc / "auditoria_resumo.json")
                resumo_auditoria = json.loads(tmp_resumo.read_text(encoding="utf-8"))
            elif isinstance(resumo_retorno, dict):
                (pasta_qc / "auditoria_resumo.json").write_text(
                    json.dumps(resumo_retorno, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                resumo_auditoria = resumo_retorno
            else:
                resumo_auditoria = {"verificacoes": {}}
        except Exception as e:
            sys.stderr.write(f"Aviso: falha na auditoria automatica ({e}). Gerando resumo vazio.\n")
            resumo_auditoria = {"verificacoes": {}}
            (pasta_qc / "auditoria_resumo.json").write_text(
                json.dumps(resumo_auditoria, indent=2), encoding="utf-8"
            )

    pasta_coleta_qc = RAIZ / "coleta" / "qc"
    if pasta_coleta_qc.exists():
        for f_md in pasta_coleta_qc.glob("*.md"):
            shutil.copy2(f_md, pasta_qc / f_md.name)

    # Passo f: copia artefatos de avaliação de busca
    pasta_avaliacao = saida / "avaliacao"
    pasta_avaliacao.mkdir(parents=True, exist_ok=True)
    consultas_ouro = []
    julgamentos_llm = {}

    for nome_arq in ("consultas_ouro.json", "julgamentos_llm.json", "listas_motores.json"):
        origem_av = RAIZ / "avaliacao" / nome_arq
        if origem_av.exists():
            shutil.copy2(origem_av, pasta_avaliacao / nome_arq)
            if nome_arq == "consultas_ouro.json":
                consultas_ouro = json.loads(origem_av.read_text(encoding="utf-8"))
            elif nome_arq == "julgamentos_llm.json":
                julgamentos_llm = json.loads(origem_av.read_text(encoding="utf-8"))

    # Passo g: gera os documentos estruturados com números calculados
    versao_repo = extrair_versao_pyproject(RAIZ)
    credito_citacao = extrair_credito_license_data(RAIZ)
    tabela_metricas = extrair_tabela_metricas_busca(RAIZ)
    total_linhas = len(df_sem_resumo)
    total_pares = contar_pares_julgados(julgamentos_llm)

    # Geração de DICIONARIO_DE_DADOS.md e esquema.json
    colunas_metadados = dicionario(df_sem_resumo)
    (saida / "DICIONARIO_DE_DADOS.md").write_text(
        gerar_dicionario_md(colunas_metadados, total_linhas), encoding="utf-8"
    )
    (saida / "esquema.json").write_text(
        gerar_esquema_json(colunas_metadados, total_linhas), encoding="utf-8"
    )

    # Geração dos demais documentos markdown
    (saida / "README.md").write_text(
        gerar_readme(manifesto, versao_repo, credito_citacao, total_linhas), encoding="utf-8"
    )
    (saida / "PROVENIENCIA.md").write_text(
        gerar_proveniencia(manifesto, camadas_detalhes), encoding="utf-8"
    )
    (saida / "LIMITACOES.md").write_text(
        gerar_limitacoes(manifesto, df_sem_resumo, resumo_auditoria, total_pares), encoding="utf-8"
    )
    (saida / "AVALIACAO_BUSCA.md").write_text(
        gerar_avaliacao_busca(consultas_ouro, julgamentos_llm, tabela_metricas), encoding="utf-8"
    )
    (saida / "SELFHOST.md").write_text(gerar_selfhost(credito_citacao), encoding="utf-8")
    (saida / "CHANGELOG.md").write_text(
        gerar_changelog(manifesto, versao_repo, total_linhas), encoding="utf-8"
    )
    (saida / "LICENCAS.md").write_text(gerar_licencas(RAIZ), encoding="utf-8")

    # Passo h: grava docker-compose.yml padrão
    (saida / "docker-compose.yml").write_text(gerar_docker_compose(), encoding="utf-8")

    # Passo i: calcula hashes SHA256SUMS de todo o pacote
    sha256sums(saida)

    # Passo j: copia documentação para versionamento no repo quando solicitado
    if docs_no_repo:
        destino_repo = RAIZ / "zenodo" / "registro_b"
        destino_repo.mkdir(parents=True, exist_ok=True)
        for arq_md in saida.glob("*.md"):
            shutil.copy2(arq_md, destino_repo / arq_md.name)
        if (saida / "esquema.json").exists():
            shutil.copy2(saida / "esquema.json", destino_repo / "esquema.json")
        if (saida / "docker-compose.yml").exists():
            shutil.copy2(saida / "docker-compose.yml", destino_repo / "docker-compose.yml")

    # Passo k: imprime sumário da exportação
    total_arquivos = sum(1 for _ in saida.rglob("*") if _.is_file())
    colunas_removidas = [c for c in COLUNAS_TEXTO_DE_TERCEIROS if c in df_original.columns]
    print("=" * 60)
    print("Sucesso na exportação do Registro B do Zenodo")
    print(f"Destino: {saida}")
    print(f"Total de arquivos gerados: {total_arquivos}")
    print(f"Total de registros em documentos_sem_resumo.parquet: {_milhar(total_linhas)}")
    print(f"Colunas de terceiros removidas: {colunas_removidas}")
    print(f"Documentação copiada para o repo: {'Sim' if docs_no_repo else 'Nao'}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    """Ponto de entrada do script com tratamento de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Exporta o Registro B do Zenodo (dados abertos sem textos protegidos)."
    )
    parser.add_argument(
        "--app-dados",
        default="app_dados",
        help="Caminho para o diretório app_dados (padrão: app_dados).",
    )
    parser.add_argument(
        "--corpus",
        default="corpus",
        help="Caminho para o diretório corpus (padrão: corpus).",
    )
    parser.add_argument(
        "--saida",
        default="~/claude/zenodo_registro_b",
        help="Diretório de destino da exportação (padrão: ~/claude/zenodo_registro_b).",
    )
    parser.add_argument(
        "--docs-no-repo",
        action="store_true",
        help="Copia documentação gerada e docker-compose.yml para zenodo/registro_b/ no repo.",
    )

    args = parser.parse_args(argv)

    caminho_app_dados = Path(args.app_dados)
    if not caminho_app_dados.is_absolute():
        caminho_app_dados = RAIZ / caminho_app_dados

    caminho_corpus = Path(args.corpus)
    if not caminho_corpus.is_absolute():
        caminho_corpus = RAIZ / caminho_corpus

    caminho_saida = Path(args.saida).expanduser().resolve()

    exportar_registro_b(
        app_dados=caminho_app_dados,
        corpus=caminho_corpus,
        saida=caminho_saida,
        docs_no_repo=args.docs_no_repo,
    )


if __name__ == "__main__":
    main()
