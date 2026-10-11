"""busca_v2.py — WP 0.3: Reproduzir a busca do app v2 como referência (spec §5.6, plano WP 0.3).

Copia fielmente a lógica de busca do app legado (`legado/app_v2.py`):
1. Normalização do catálogo inicial e tokenização BM25Okapi sobre:
   (description + sender_name + folder + reference_code + new_code + old_code).str.lower().split()
2. Normalização max para o escore léxico:
   lexical_normalized = lexical_scores_raw / max_lex
3. Normalização min-max para o escore semântico:
   semantic_normalized = (semantic_scores_raw - min_sem) / range_sem
4. Fusão ponderada de acordo com o comprimento da consulta:
   - len(tokens) <= 2: 75% léxico + 25% semântico
   - len(tokens) > 2:  35% léxico + 65% semântico
5. Exact Archival Code Booster: qualquer documento cujo reference_code, new_code ou old_code
   contenha a consulta (substring insensível a maiúsculas) recebe nota 1.0.
6. Filtro por limiar de relevância (default 0.30) e ordenação decrescente por semantic_score.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

_RE_PREFIX = re.compile(r"^\d+\.\s*")
_RE_YEAR = re.compile(r"\b(1[5-9]\d{2})\b")

_PAIS_MAP = {
    "Reino de Portugal": "Portugal",
    "Portugal (Reino de Portugal)": "Portugal",
    "Estado do Brasil (Império Português)": "Brasil",
    "Estado do Brasil (Monarquia Portuguesa)": "Brasil",
    "Reino de Portugal (América Portuguesa)": "Brasil",
}


def _extract_year(value: Any) -> Any:
    """Extrai o primeiro ano plausível (1500-1999) como em legado/app_v2.py."""
    if not isinstance(value, str) or not value:
        return pd.NA
    stripped = _RE_PREFIX.sub("", value)
    m = _RE_YEAR.search(stripped)
    return int(m.group(1)) if m else pd.NA


def _check_authorship(row: pd.Series) -> bool:
    """True se sender_name aparece na description (case-insensitive)."""
    sender = row.get("sender_name")
    desc = row.get("description")
    if not isinstance(sender, str) or not isinstance(desc, str):
        return False
    if not sender or not desc:
        return False
    return sender.lower() in desc.lower()


def localizar_arquivos_legado(
    catalogo_path: Path | str | None = None,
    indice_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Localiza o JSON do catálogo limpo e o PKL de embeddings da v2."""
    if catalogo_path is not None and indice_path is not None:
        return Path(catalogo_path), Path(indice_path)

    # Candidatos padrão a partir do repositório
    raiz = Path(__file__).resolve().parent.parent
    candidatos_cat = [
        raiz.parent / "legacy app" / "ahu_sul_catalog_limpo.json",
        raiz / "legado" / "ahu_sul_catalog_limpo.json",
        raiz / "ahu_sul_catalog_limpo.json",
    ]
    candidatos_idx = [
        raiz.parent / "legacy app" / "ahu_semantic_index.pkl",
        raiz / "legado" / "ahu_semantic_index.pkl",
        raiz / "ahu_semantic_index.pkl",
    ]

    cat_encontrado = None
    for p in candidatos_cat:
        if p.is_file():
            cat_encontrado = p
            break

    idx_encontrado = None
    for p in candidatos_idx:
        if p.is_file():
            idx_encontrado = p
            break

    if catalogo_path is not None:
        cat_encontrado = Path(catalogo_path)
    if indice_path is not None:
        idx_encontrado = Path(indice_path)

    if cat_encontrado is None or not cat_encontrado.is_file():
        raise FileNotFoundError(
            f"Catálogo legado ahu_sul_catalog_limpo.json não encontrado. Procurado em: {candidatos_cat}"
        )
    if idx_encontrado is None or not idx_encontrado.is_file():
        raise FileNotFoundError(
            f"Índice semântico ahu_semantic_index.pkl não encontrado. Procurado em: {candidatos_idx}"
        )

    return cat_encontrado, idx_encontrado


def carregar_dados_legado(catalogo_path: Path | str) -> tuple[pd.DataFrame, BM25Okapi]:
    """Carrega e enriquece o catálogo exatamente como load_data() em legado/app_v2.py."""
    with open(catalogo_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    df = pd.json_normalize(raw_data, sep="_")

    perplexity_renames = {
        "perplexity_search_data_cidade_nascimento": "cidade_nascimento",
        "perplexity_search_data_pais_normalizado": "pais_nascimento",
        "perplexity_search_data_confianca": "confianca",
        "perplexity_search_data_link_fonte": "link_fonte",
    }
    df.rename(columns={k: v for k, v in perplexity_renames.items() if k in df.columns}, inplace=True)

    if "perplexity_search_data_pais_nascimento" in df.columns and "pais_nascimento" in df.columns:
        df["pais_nascimento"] = df["pais_nascimento"].fillna(df["perplexity_search_data_pais_nascimento"])
    elif "perplexity_search_data_pais_nascimento" in df.columns and "pais_nascimento" not in df.columns:
        df["pais_nascimento"] = df["perplexity_search_data_pais_nascimento"]

    df["pais_normalizado"] = df["pais_nascimento"].map(
        lambda x: _PAIS_MAP.get(x, x) if isinstance(x, str) else x
    )
    df["pais_normalizado"] = df["pais_normalizado"].fillna("Desconhecido")
    df["cidade_nascimento"] = df["cidade_nascimento"].fillna("Desconhecida")
    df["confianca"] = df["confianca"].replace({"Media": "Média"}).fillna("Nula")
    df["folder"] = df["folder"].fillna("Sem região definida")

    if "vernacular_score" in df.columns:
        df["vernacular_score"] = pd.to_numeric(df["vernacular_score"], errors="coerce").fillna(0)
        df["vernacular_score"] = df["vernacular_score"] / 10.0
    else:
        df["vernacular_score"] = 0.0

    if "ano_isolado" not in df.columns:
        df["ano_isolado"] = df["document_id_and_date"].apply(_extract_year)
    df["ano_isolado"] = df["ano_isolado"].astype("Int64")

    if "autoria_verificada" not in df.columns:
        df["autoria_verificada"] = df.apply(_check_authorship, axis=1)

    corpus = (
        df["description"].fillna("") + " "
        + df["sender_name"].fillna("") + " "
        + df["folder"].fillna("") + " "
        + df["reference_code"].fillna("") + " "
        + df["new_code"].fillna("") + " "
        + df["old_code"].fillna("")
    ).str.lower().tolist()

    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return df, bm25


class MotorBuscaV2:
    """Motor que encapsula a lógica de recuperação exata do app v2."""

    def __init__(
        self,
        df: pd.DataFrame,
        bm25: BM25Okapi,
        embeddings: np.ndarray,
        reference_codes: list[str],
        encoder: Callable[[str], np.ndarray] | None = None,
        model_name: str = "intfloat/multilingual-e5-large",
    ) -> None:
        self.df = df
        self.bm25 = bm25
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.reference_codes = reference_codes
        self._encoder = encoder
        self.model_name = model_name
        self._sentence_transformer_model: Any = None

    @classmethod
    def carregar(
        cls,
        catalogo_path: Path | str | None = None,
        indice_path: Path | str | None = None,
        encoder: Callable[[str], np.ndarray] | None = None,
    ) -> MotorBuscaV2:
        """Carrega dados e índice a partir dos caminhos dados ou padrões."""
        cat_file, idx_file = localizar_arquivos_legado(catalogo_path, indice_path)
        df, bm25 = carregar_dados_legado(cat_file)

        with open(idx_file, "rb") as f:
            idx_data = pickle.load(f)

        embeddings = idx_data["embeddings"]
        ref_codes = idx_data.get("reference_codes", [])
        model_name = idx_data.get("model_used", "intfloat/multilingual-e5-large")

        return cls(
            df=df,
            bm25=bm25,
            embeddings=embeddings,
            reference_codes=ref_codes,
            encoder=encoder,
            model_name=model_name,
        )

    def _obter_encoder(self) -> Callable[[str], np.ndarray]:
        if self._encoder is not None:
            return self._encoder

        if self._sentence_transformer_model is None:
            from sentence_transformers import SentenceTransformer

            self._sentence_transformer_model = SentenceTransformer(self.model_name)

        model = self._sentence_transformer_model

        def _encode(texto: str) -> np.ndarray:
            vec = model.encode(f"query: {texto}", convert_to_tensor=False)
            return np.asarray(vec, dtype=np.float32)

        return _encode

    def buscar(
        self,
        query: str,
        limiar_semantico: float = 0.30,
        max_resultados: int | None = None,
    ) -> pd.DataFrame:
        """Executa a busca híbrida da v2 para a query dada."""
        if not query or not query.strip():
            res = self.df.sort_values(by="vernacular_score", ascending=False).copy()
            if max_resultados:
                res = res.head(max_resultados)
            return res

        df_work = self.df.copy()
        query_lower = query.lower().strip()
        tokenized_query = query_lower.split()

        # 1. Pontuação Léxica (BM25)
        lexical_scores_raw = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)
        max_lex = (
            np.max(lexical_scores_raw)
            if len(lexical_scores_raw) > 0 and np.max(lexical_scores_raw) > 0
            else 1.0
        )
        lexical_normalized = lexical_scores_raw / max_lex

        # 2. Pontuação Semântica (Dense Vectors)
        encoder = self._obter_encoder()
        query_emb = encoder(query)

        # Similaridade de cosseno (como util.cos_sim)
        norm_q = np.linalg.norm(query_emb)
        norm_corpus = np.linalg.norm(self.embeddings, axis=1)
        denom = norm_q * norm_corpus
        denom[denom == 0] = 1.0
        semantic_scores_raw = np.dot(self.embeddings, query_emb) / denom

        max_sem = np.max(semantic_scores_raw)
        min_sem = np.min(semantic_scores_raw)
        range_sem = (max_sem - min_sem) if (max_sem - min_sem) > 0 else 1.0
        semantic_normalized = (semantic_scores_raw - min_sem) / range_sem

        # 3. Fusão dinâmica de escore
        if len(tokenized_query) <= 2:
            lexical_weight = 0.75
            semantic_weight = 0.25
        else:
            lexical_weight = 0.35
            semantic_weight = 0.65

        final_hybrid_scores = (lexical_normalized * lexical_weight) + (semantic_normalized * semantic_weight)
        df_work["semantic_score"] = final_hybrid_scores

        # 4. Exact Archival Code Booster (legado: str.contains sem regex)
        exact_match_mask = (
            df_work["reference_code"].astype(str).str.contains(query, case=False, na=False, regex=False)
            | df_work["new_code"].astype(str).str.contains(query, case=False, na=False, regex=False)
            | df_work["old_code"].astype(str).str.contains(query, case=False, na=False, regex=False)
        )
        df_work.loc[exact_match_mask, "semantic_score"] = 1.0

        # 5. Filtragem por limiar e ordenação decrescente
        search_mask = df_work["semantic_score"] >= limiar_semantico
        results_df = df_work[search_mask].sort_values(by="semantic_score", ascending=False, kind="stable")

        if max_resultados is not None:
            results_df = results_df.head(max_resultados)

        return results_df
