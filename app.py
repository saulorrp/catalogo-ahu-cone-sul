import streamlit as st
import pandas as pd
import re
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pickle
import urllib.parse
from rank_bm25 import BM25Okapi
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Classificador AHU Sul", layout="wide")

# --- GLOBAL CSS ---
# Monospace font (EDSS/Terminal standard aesthetic)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Consolas', 'Courier New', monospace !important;
    }

    /* ── Panel header styling ── */
    .panel-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334477;
        border-radius: 10px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.5rem;
        color: #c8d0e8;
    }
    .panel-header h2 {
        color: #e0e0ff;
        margin: 0;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    .panel-header p {
        color: #8899bb;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* ── Section cards inside the panel ── */
    .section-card {
        background: #0e1117;
        border: 1px solid #262d3d;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .section-card h4 {
        color: #a0b0d0;
        margin: 0 0 0.3rem 0;
    }
    .section-card .meta {
        color: #667799;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)



# ==========================================
# 1. DATA ARCHITECTURE & CACHING
# ==========================================

# Pre-compiled regexes for date isolation (same logic as enriquecer_catalogo.py)
_RE_PREFIX = re.compile(r"^\d+\.\s*")
_RE_YEAR   = re.compile(r"\b(1[5-9]\d{2})\b")

# Normalization map for pais_nascimento variants from Perplexity
_PAIS_MAP = {
    'Reino de Portugal': 'Portugal',
    'Portugal (Reino de Portugal)': 'Portugal',
    'Estado do Brasil (Império Português)': 'Brasil',
    'Estado do Brasil (Monarquia Portuguesa)': 'Brasil',
    'Reino de Portugal (América Portuguesa)': 'Brasil',
}


def _extract_year(value):
    """Extract the first plausible year (1500-1999) from a document_id_and_date string."""
    if not isinstance(value, str) or not value:
        return pd.NA
    stripped = _RE_PREFIX.sub("", value)
    m = _RE_YEAR.search(stripped)
    return int(m.group(1)) if m else pd.NA


def _check_authorship(row):
    """Return True if sender_name appears inside description (case-insensitive)."""
    sender = row.get("sender_name")
    desc   = row.get("description")
    if not isinstance(sender, str) or not isinstance(desc, str):
        return False
    if not sender or not desc:
        return False
    return sender.lower() in desc.lower()


@st.cache_data
def load_data():
    """
    Loads the main JSON catalog, flattens perplexity_search_data into native
    columns, normalizes geographic metadata, computes enrichment columns
    (ano_isolado, autoria_verificada), and initializes the BM25 engine.
    """
    with open('ahu_sul_catalog_limpo.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Flatten nested perplexity_search_data into top-level columns
    df = pd.json_normalize(raw_data, sep='_')

    # Rename flattened perplexity columns to clean names
    perplexity_renames = {
        'perplexity_search_data_cidade_nascimento': 'cidade_nascimento',
        'perplexity_search_data_pais_normalizado': 'pais_nascimento', # We map the normalized one directly here!
        'perplexity_search_data_confianca': 'confianca',
        'perplexity_search_data_link_fonte': 'link_fonte',
    }
    df.rename(columns={k: v for k, v in perplexity_renames.items() if k in df.columns}, inplace=True)
    
    # Fill any missing pais_nascimento with the old field just in case
    if 'perplexity_search_data_pais_nascimento' in df.columns and 'pais_nascimento' in df.columns:
        df['pais_nascimento'] = df['pais_nascimento'].fillna(df['perplexity_search_data_pais_nascimento'])
    elif 'perplexity_search_data_pais_nascimento' in df.columns and 'pais_nascimento' not in df.columns:
        df['pais_nascimento'] = df['perplexity_search_data_pais_nascimento']

    # ── Geographic normalization (vectorized) ──
    df['pais_normalizado'] = df['pais_nascimento'].map(
        lambda x: _PAIS_MAP.get(x, x) if isinstance(x, str) else x
    )
    df['pais_normalizado'] = df['pais_normalizado'].fillna('Desconhecido')
    df['cidade_nascimento'] = df['cidade_nascimento'].fillna('Desconhecida')
    df['confianca'] = df['confianca'].replace({'Media': 'Média'}).fillna('Nula')

    # Fill missing folder data
    df['folder'] = df['folder'].fillna('Sem região definida')

    # Normalize Vernacular Score from 1-10 to 0.0-1.0
    if 'vernacular_score' in df.columns:
        df['vernacular_score'] = pd.to_numeric(df['vernacular_score'], errors='coerce').fillna(0)
        df['vernacular_score'] = df['vernacular_score'] / 10.0
    else:
        df['vernacular_score'] = 0.0

    # ── Enrichment: ano_isolado ──
    if 'ano_isolado' not in df.columns:
        df['ano_isolado'] = df['document_id_and_date'].apply(_extract_year)
    df['ano_isolado'] = df['ano_isolado'].astype('Int64')  # nullable int

    # ── Enrichment: autoria_verificada ──
    if 'autoria_verificada' not in df.columns:
        df['autoria_verificada'] = df.apply(_check_authorship, axis=1)

    # BM25 Lexical Engine
    corpus = (df['description'].fillna('') + " " +
              df['sender_name'].fillna('') + " " +
              df['folder'].fillna('') + " " +
              df['reference_code'].fillna('') + " " +
              df['new_code'].fillna('') + " " +
              df['old_code'].fillna('')).str.lower().tolist()

    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return df, bm25


@st.cache_data
def load_semantic_index():
    """Loads the pre-computed semantic embeddings (Dense Vectors) from a pickle file."""
    with open('ahu_semantic_index.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


@st.cache_resource
def load_semantic_model(model_name):
    """Loads the SentenceTransformer model for real-time query embedding."""
    return SentenceTransformer(model_name)


# Initialize Data and Models
df, bm25_engine = load_data()
semantic_index = load_semantic_index()

model_name = semantic_index.get('model_used', 'intfloat/multilingual-e5-large')
model = load_semantic_model(model_name)

device = model.device
corpus_embeddings = torch.tensor(semantic_index['embeddings']).to(device)
reference_codes_list = semantic_index['reference_codes']


# ==========================================
# 2. DIACHRONIC SAMPLES HELPER FUNCTIONS
# ==========================================

def get_verified_authors(source_df):
    """
    Returns a list of (author_name, doc_count) tuples for authors
    where autoria_verificada == True, sorted alphabetically by name.
    Operates on the (potentially filtered) source DataFrame so that
    sidebar filters affect which authors are available.
    """
    verified = source_df[source_df['autoria_verificada'] == True]
    counts = verified['sender_name'].value_counts()
    result = [(name, count) for name, count in counts.items()]
    result.sort(key=lambda x: x[0])
    return result


def get_author_docs(source_df, author_name):
    """
    Returns the chronologically sorted (by ano_isolado ASC) documents
    for a given author, filtered to autoria_verificada == True and ano_isolado not null.
    Operates on the (potentially filtered) source DataFrame.
    """
    mask = (
        (source_df['sender_name'] == author_name) &
        (source_df['autoria_verificada'] == True) &
        (source_df['ano_isolado'].notna())
    )
    return source_df[mask].sort_values('ano_isolado', ascending=True).reset_index(drop=True)


def build_collection_plan(author_name, author_df, batch_size=10):
    """
    Builds the JSON-serializable collection plan for a given author's documents.
    Includes geographic metadata from perplexity_search_data, prepared for
    future ingestion in morphosyntactic annotation environments (e.g. Tycho Brahe).
    """
    sections = []
    total = len(author_df)

    for i in range(0, total, batch_size):
        batch = author_df.iloc[i:i + batch_size]
        docs = []
        for _, row in batch.iterrows():
            crav_code = str(row.get('reference_code', 'N/A'))
            crav_link = None
            if crav_code.startswith("PT/AHU"):
                encoded_crav = urllib.parse.quote(crav_code, safe='')
                crav_link = f"https://digitarq.arquivos.pt/search?query={encoded_crav}&isAdvancedSearch=false"
            docs.append({
                "ano_isolado": int(row['ano_isolado']) if pd.notna(row['ano_isolado']) else None,
                "cota": str(row.get('new_code', 'N/A')),
                "referencia": crav_code,
                "crav_link": crav_link,
                "tipologia": str(row.get('extracted_typology', 'N/A')),
                "descricao": str(row.get('description', '')),
                "cidade_nascimento_autor": str(row.get('cidade_nascimento', 'Desconhecida')),
                "pais_nascimento_autor": str(row.get('pais_normalizado', 'Desconhecido')),
                "confianca_biografica": str(row.get('confianca', 'Nula')),
            })
        sections.append({
            "id_secao": (i // batch_size) + 1,
            "documentos": docs
        })

    years = author_df['ano_isolado'].dropna()
    plan = {
        "autor_alvo": author_name,
        "metadados_autor": {
            "cidade_nascimento": str(author_df['cidade_nascimento'].iloc[0]) if len(author_df) > 0 else None,
            "pais_nascimento": str(author_df['pais_normalizado'].iloc[0]) if len(author_df) > 0 else None,
            "confianca": str(author_df['confianca'].iloc[0]) if len(author_df) > 0 else None,
        },
        "recorte_temporal": {
            "inicio": int(years.min()) if len(years) > 0 else None,
            "fim": int(years.max()) if len(years) > 0 else None
        },
        "total_documentos": total,
        "secoes_coleta": sections
    }
    return plan


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================

if 'show_samples_panel' not in st.session_state:
    st.session_state.show_samples_panel = False


# ==========================================
# 4. UNIFIED SIDEBAR (renders once for all screens)
# ==========================================

with st.sidebar:
    # ── Navigation button (top of sidebar) ──
    if st.session_state.show_samples_panel:
        if st.button(":material/arrow_back: Voltar ao Motor de Busca", use_container_width=True, key="btn_voltar"):
            st.session_state.show_samples_panel = False
            st.rerun()
    else:
        if st.button(":material/history_edu: Ferramenta de Amostras Diacrônicas", use_container_width=True, key="btn_amostras"):
            st.session_state.show_samples_panel = True
            st.rerun()

    st.divider()

    # ── Predefined Search Profiles ──
    st.header(":material/tune: Perfis de Busca Predefinidos")
    lente = st.radio(
        "Selecione uma lente metodológica:",
        ["Busca Livre (Personalizada)",
         "Vozes Marginalizadas & História Social",
         "Sintaxe Diacrônica (Alto SRSP)",
         "Máquina Administrativa (Top-Down)"]
    )

    st.divider()

    # ── AHU Section Filter ──
    st.header(":material/filter_alt: Filtro de Seções do AHU")
    todas_regioes = df['folder'].unique().tolist()
    regioes_selecionadas = st.multiselect("Regiões/Capitanias:", todas_regioes, default=todas_regioes)

    # ── Sociolinguistic Filters ──
    st.header(":material/groups_2: Filtros Sociolinguísticos")

    min_score = 0.0
    max_score = 1.0
    vetor_padrao = ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"]
    categorias_disponiveis = df['sender_category'].fillna('Unknown').unique().tolist()
    remetente_padrao = categorias_disponiveis.copy()

    if lente == "Vozes Marginalizadas & História Social":
        vetor_padrao = ["Bottom-Up"]
        remetente_padrao = [c for c in ["Commoner", "Marginalized", "Low Military"] if c in categorias_disponiveis]
    elif lente == "Sintaxe Diacrônica (Alto SRSP)":
        min_score = 0.7
    elif lente == "Máquina Administrativa (Top-Down)":
        vetor_padrao = ["Top-Down", "Horizontal"]
        remetente_padrao = [c for c in ["Metropolitan Elite", "Local Elite"] if c in categorias_disponiveis]

    score_range = st.slider(
        "Score de Relevância Sociolinguística Potencial (SRSP):",
        0.0, 1.0, (min_score, max_score), step=0.1
    )

    st.markdown("""
        <div style="display: flex; text-align: center; font-size: 0.75em; color: gray; margin-top: -15px; margin-bottom: 15px;">
            <div style="flex: 0.35; border-right: 2px solid #555;">0.0 - 0.3<br>Formulaico</div>
            <div style="flex: 0.30; border-right: 2px solid #555;">0.4 - 0.6<br>Moderado</div>
            <div style="flex: 0.35;">0.7 - 1.0<br>Potencial</div>
        </div>
    """, unsafe_allow_html=True)

    vetores = st.multiselect(
        "Direção da Comunicação:",
        ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"],
        default=vetor_padrao
    )
    categorias = st.multiselect(
        "Perfil Social do Remetente:",
        categorias_disponiveis,
        default=remetente_padrao
    )

    st.divider()

    # ── Diatopic & Biographical Filters (NEW) ──
    st.header(":material/public: Filtros Diatópicos e Biográficos")

    todos_paises = sorted(df['pais_normalizado'].unique().tolist())
    paises_selecionados = st.multiselect(
        "País de Nascimento do Remetente:",
        todos_paises,
        default=todos_paises
    )

    # Dynamic city filter: options depend on selected countries
    cidades_do_pais = sorted(
        df[df['pais_normalizado'].isin(paises_selecionados)]['cidade_nascimento'].unique().tolist()
    ) if paises_selecionados else []
    cidades_selecionadas = st.multiselect(
        "Cidade/Região de Nascimento:",
        cidades_do_pais,
        default=[],
        help="Deixe vazio para incluir todas as cidades dos países selecionados."
    )

    todos_niveis_confianca = sorted(df['confianca'].unique().tolist())
    confianca_selecionada = st.multiselect(
        "Nível de Confiança Biográfica:",
        todos_niveis_confianca,
        default=todos_niveis_confianca,
        help="Filtra pelo grau de confiança da pesquisa biográfica do Perplexity."
    )


# ==========================================
# 5. GLOBAL METADATA FILTERING
# ==========================================

df_filter = df.copy()
df_filter['vector'] = df_filter['vector'].fillna('Unknown')
df_filter['sender_category'] = df_filter['sender_category'].fillna('Unknown')

metadata_mask = (
    (df_filter['folder'].isin(regioes_selecionadas)) &
    (df_filter['vector'].isin(vetores)) &
    (df_filter['sender_category'].isin(categorias)) &
    (df_filter['vernacular_score'] >= score_range[0]) &
    (df_filter['vernacular_score'] <= score_range[1]) &
    (df_filter['pais_normalizado'].isin(paises_selecionados)) &
    (df_filter['confianca'].isin(confianca_selecionada))
)

# City filter is optional — only applied if user selects specific cities
if cidades_selecionadas:
    metadata_mask = metadata_mask & (df_filter['cidade_nascimento'].isin(cidades_selecionadas))

df_filtered = df_filter[metadata_mask]


# ==========================================
# 6. UI FLOW — CONDITIONAL ROUTING
# ==========================================

if st.session_state.show_samples_panel:
    # ─────────────────────────────────────────
    #  PANEL: DIACHRONIC SAMPLES TOOL
    # ─────────────────────────────────────────

    # Panel header
    st.markdown("""
    <div class="panel-header">
        <h2>Amostras Diacrônicas & Exportação</h2>
        <p>Ferramenta de seleção de corpus para Linguística Histórica. Os filtros da barra lateral afetam os autores disponíveis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader(":material/history_edu: Ferramenta de Criação de Amostras Diacrônicas")
    st.markdown("*Selecione um autor com autoria verificada no catálogo para gerar um plano de coleta organizado cronologicamente.*")

    # Author selectbox (uses globally filtered DataFrame)
    authors = get_verified_authors(df_filtered)

    if not authors:
        st.warning("Nenhum autor com autoria verificada encontrado com os filtros atuais. Tente ampliar os filtros na barra lateral.")
    else:
        author_options = [f"{name}  ({count} docs)" for name, count in authors]
        selected_label = st.selectbox(
            "Selecione o autor/remetente:",
            author_options,
            index=0,
            key="author_select"
        )

        # Extract the actual author name (strip the count suffix)
        selected_author = authors[author_options.index(selected_label)][0]

        # Get filtered & sorted documents
        author_docs = get_author_docs(df_filtered, selected_author)

        if author_docs.empty:
            st.info("Nenhum documento com data válida encontrado para este autor.")
        else:
            # Summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            years = author_docs['ano_isolado'].dropna()
            col_m1.metric("Total de Documentos", len(author_docs))
            col_m2.metric("Ano Mais Antigo", int(years.min()) if len(years) > 0 else "N/A")
            col_m3.metric("Ano Mais Recente", int(years.max()) if len(years) > 0 else "N/A")
            col_m4.metric("Amplitude (anos)", int(years.max() - years.min()) if len(years) > 1 else 0)

            # Author geographic metadata
            geo_col1, geo_col2, geo_col3 = st.columns(3)
            geo_col1.metric("País de Nascimento", author_docs['pais_normalizado'].iloc[0])
            geo_col2.metric("Cidade de Nascimento", author_docs['cidade_nascimento'].iloc[0])
            geo_col3.metric("Confiança Biográfica", author_docs['confianca'].iloc[0])

            st.markdown("---")

            # Paginated sections (batches of 10)
            BATCH_SIZE = 10
            total_docs = len(author_docs)
            num_sections = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

            for sec_idx in range(num_sections):
                start = sec_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, total_docs)
                batch = author_docs.iloc[start:end]

                year_start = batch['ano_isolado'].iloc[0]
                year_end   = batch['ano_isolado'].iloc[-1]

                with st.expander(
                    f"Seção {sec_idx + 1} de {num_sections}  |  Docs {start + 1}-{end}  |  {int(year_start)}-{int(year_end)}",
                    expanded=(sec_idx == 0)
                ):
                    display_df = batch[['ano_isolado', 'reference_code', 'extracted_typology',
                                        'cidade_nascimento', 'pais_normalizado', 'description']].copy()
                    display_df.columns = ['Ano', 'Referência CRAV', 'Tipologia',
                                          'Cidade Nascimento', 'País Nascimento', 'Resumo']
                    # Build CRAV links for each document
                    display_df['Digitarq'] = display_df['Referência CRAV'].apply(
                        lambda c: f"https://digitarq.arquivos.pt/search?query={urllib.parse.quote(str(c), safe='')}&isAdvancedSearch=false"
                        if isinstance(c, str) and c.startswith('PT/AHU') else None
                    )
                    # Truncate long descriptions for readability
                    display_df['Resumo'] = display_df['Resumo'].apply(
                        lambda x: (x[:150] + '...') if isinstance(x, str) and len(x) > 150 else x
                    )
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Digitarq": st.column_config.LinkColumn(
                                "Digitarq",
                                display_text="Abrir",
                            )
                        }
                    )

            st.markdown("---")

            # JSON export (with geographic metadata for Tycho Brahe ingest)
            plan = build_collection_plan(selected_author, author_docs, BATCH_SIZE)
            plan_json = json.dumps(plan, ensure_ascii=False, indent=4)

            safe_filename = re.sub(r'[^\w\s-]', '', selected_author).strip().replace(' ', '_')
            st.download_button(
                label=f":material/download: Baixar plano_coleta_{safe_filename}.json",
                data=plan_json.encode('utf-8'),
                file_name=f"plano_coleta_{safe_filename}.json",
                mime="application/json",
                key="json_download_plan"
            )

else:
    # ─────────────────────────────────────────
    #  MAIN SCREEN WITH TABS
    # ─────────────────────────────────────────

    st.title("Classificador de Documentos do Catálogo do AHU para a Macrorregião Sul do Brasil")

    st.markdown("""
    ### Sobre esta ferramenta
    **1. Motor de Busca e Triagem:** Este sistema não contém as imagens digitalizadas dos manuscritos originais. Ele funciona como um classificador para os resumos do catálogo do **Arquivo Histórico Ultramarino (AHU)**. O objetivo é permitir que pesquisadores cruzem recortes geográficos, temas históricos e variáveis sociolinguísticas para obter as **cotas arquivísticas** (ex: *PT/AHU/CU/...*) antes de acessar o arquivo físico ou o Projeto Resgate.

    **2. Score de Relevância Sociolinguística Potencial (SRSP):** Cada documento teve sua descrição processada pelo DeepSeek para a atribuição de um valor numérico indicativo da probabilidade de o documento conter indícios de vernacularidade. Esse valor varia entre **0 e 1**.
    * Um **Score próximo a 0** indica baixa probabilidade (fórmulas diplomáticas rígidas, linguagem erudita metropolitana ou forte padronização de notários).
    * Um **Score próximo a 1** indica alta probabilidade de que o manuscrito original contenha marcas de oralidade, inovações sintáticas e vazamento do português brasileiro colonial.

    **3. O Corte de Relevância (Rigor da Busca):** Este parâmetro define o limite matemático exigido para que o motor considere um documento pertinente à sua consulta. Ele cruza o sentido do texto com a correspondência exata das palavras.
    * **Relevância próxima a 0** amplia o escopo da pesquisa e relaxa o filtro para incluir documentos com uma relação conceitual mais distante, periférica ou apenas tangencial ao termo inserido.
    * **Relevância próxima a 1** exige uma correspondência extremamente estrita com o tema pesquisado.

    **4. Ferramenta de Amostras Diacrônicas:** Acesse pela barra lateral para selecionar autores com autoria verificada e gerar planos de coleta cronológicos em JSON, preparados para ingestão na Plataforma Tycho Brahe.

    **5. Acesso Direto aos Acervos:** A ferramenta gera automaticamente links para as plataformas oficiais. Ao expandir um resultado na tela, você pode usar o Código de Referência para abrir a ficha de controle arquivístico no DigitArq, ou usar o Código Atual para buscar as imagens microfilmadas no portal do Projeto Resgate (Biblioteca Nacional).
    """)
    st.divider()

    # ── TABS ──
    tab_busca, tab_diatopico = st.tabs([
        ":material/search: Motor de Busca",
        ":material/public: Análise Diatópica e Autoral"
    ])

    # ── TAB 1: SEARCH ENGINE ──
    with tab_busca:
        st.subheader(":material/search: Busca Semântica/Lexical")
        st.markdown("*Digite um conceito, tema, evento histórico ou **cota arquivística**.*")

        query = st.text_input("Ex: 'conflitos de terra', 'deserção de soldados', ou 'PT/AHU/CU/021/0006/00390':")

        col_segura, col_vazia = st.columns([2, 8])

        with col_segura:
            limiar_str = st.text_input(
                "Valor de Corte de Relevância:",
                value="30",
                max_chars=2,
                help="Preencha as casas decimais, se digitar 5, será lido como 0.50."
            )

        try:
            limiar_limpo = limiar_str.strip()
            if not limiar_limpo: limiar_limpo = "50"
            elif len(limiar_limpo) == 1: limiar_limpo += "0"
            limiar_semantico = float(f"0.{limiar_limpo}")
        except ValueError:
            limiar_semantico = 0.30
            st.error("Por favor, digite apenas números. Retornando ao rigor padrão (0.30).")

        st.divider()

        # ==========================================
        # HYBRID SEARCH ENGINE (ENSEMBLE RETRIEVAL)
        # ==========================================

        if query:
            query_lower = query.lower().strip()
            tokenized_query = query_lower.split()

            # Lexical Scoring (BM25)
            lexical_scores_raw = bm25_engine.get_scores(tokenized_query)
            max_lex = np.max(lexical_scores_raw) if np.max(lexical_scores_raw) > 0 else 1
            lexical_normalized = lexical_scores_raw / max_lex

            # Semantic Scoring (Dense Vector Embeddings)
            e5_query = f"query: {query}"
            query_embedding = model.encode(e5_query, convert_to_tensor=True)
            semantic_scores_raw = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()

            max_sem = np.max(semantic_scores_raw)
            min_sem = np.min(semantic_scores_raw)
            range_sem = (max_sem - min_sem) if (max_sem - min_sem) > 0 else 1
            semantic_normalized = (semantic_scores_raw - min_sem) / range_sem

            # Dynamic Score Fusion
            if len(tokenized_query) <= 2:
                lexical_weight = 0.75
                semantic_weight = 0.25
            else:
                lexical_weight = 0.35
                semantic_weight = 0.65

            final_hybrid_scores = (lexical_normalized * lexical_weight) + (semantic_normalized * semantic_weight)
            df_filter['semantic_score'] = final_hybrid_scores

            # Exact Archival Code Booster
            exact_match_mask = (
                df_filter['reference_code'].str.contains(query, case=False, na=False, regex=False) |
                df_filter['new_code'].str.contains(query, case=False, na=False, regex=False) |
                df_filter['old_code'].str.contains(query, case=False, na=False, regex=False)
            )
            df_filter.loc[exact_match_mask, 'semantic_score'] = 1.0

            search_mask = metadata_mask & (df_filter['semantic_score'] >= limiar_semantico)
            results_df = df_filter[search_mask].sort_values(by='semantic_score', ascending=False)
        else:
            results_df = df_filtered.sort_values(by='vernacular_score', ascending=False)

        # ==========================================
        # RENDER RESULTS
        # ==========================================

        st.subheader(f"Resultados Encontrados: {len(results_df)} documentos")

        if not results_df.empty:
            for idx, row in results_df.head(50).iterrows():
                score = row.get('vernacular_score', 0.0)
                date_id = row.get('document_id_and_date', 'Sem Data')
                folder = row.get('folder', 'Local Desconhecido')

                if query:
                    sem_score = row.get('semantic_score', 0.0)
                    expander_title = f"Relevância: {sem_score:.2f} | SRSP: {score:.1f} | {date_id} | {folder} "
                else:
                    expander_title = f" {date_id} | {folder} | SRSP: {score:.1f}"

                with st.expander(expander_title):
                    crav_code = row.get('reference_code', 'Sem Cota CRAV')
                    new_code = row.get('new_code', 'N/A')
                    resgate_url = "https://resgate.bn.gov.br/"

                    if crav_code.startswith("PT/AHU"):
                        encoded_crav = urllib.parse.quote(crav_code, safe='')
                        crav_url = f"https://digitarq.arquivos.pt/search?query={encoded_crav}&isAdvancedSearch=false"
                        st.markdown(f"**Código de Referência:** {crav_code} ([Busca no Digitarq]({crav_url}))")
                    else:
                        st.markdown(f"**Código de Referência:** {crav_code}")

                    st.markdown(f"**Código Atual:** {new_code} ([Busca no Projeto Resgate]({resgate_url}))")

                    st.markdown(f"**Tipologia:** {row.get('extracted_typology', 'N/A')}")
                    st.markdown(f"**Remetente:** {row.get('sender_name', 'N/A')} *(Classe: {row.get('sender_category', 'N/A')})*")
                    st.markdown(f"**Hierarquia:** {row.get('vector', 'N/A')}")

                    # Geographic metadata from Perplexity
                    pais_r = row.get('pais_normalizado', 'Desconhecido')
                    cidade_r = row.get('cidade_nascimento', 'Desconhecida')
                    conf_r = row.get('confianca', 'Nula')
                    st.markdown(f"**Origem do Remetente:** {cidade_r}, {pais_r} *(Confiança: {conf_r})*")

                    st.markdown("---")
                    st.markdown(f"**Resumo do Arquivo (de autoria do AHU):**\n{row.get('description', '')}")
                    st.markdown("---")
                    reasoning = row.get('sociolinguistic_reasoning_by_deepseek_v3', '')
                    st.markdown(f"**Justificativa Analítica para o Score (LLM):**\n*{reasoning}*")

            if len(results_df) > 50:
                st.info(f"Mostrando os 50 resultados mais relevantes no navegador de um total de {len(results_df)}. Acesse a Ferramenta de Amostras pela barra lateral para selecionar autores e gerar planos de coleta.")
        else:
            st.warning("Nenhum documento encontrado com os filtros atuais. Experimente diminuir o valor de corte da relevância ou aumentar o intervalo do Score.")

    # ── TAB 2: DIATOPIC & AUTHORIAL DASHBOARD ──
    with tab_diatopico:
        st.subheader(":material/public: Dashboard de Análise Diatópica e Autoral")
        st.markdown(f"*Insights visuais sobre o corpus filtrado. Exibindo **{len(df_filtered)}** documentos com os filtros atuais da barra lateral.*")

        # ── Chart A: Temporal distribution by country of birth ──
        st.markdown("#### Distribuição Temporal por País de Nascimento do Remetente")

        df_chart = df_filtered.dropna(subset=['ano_isolado']).copy()

        if not df_chart.empty:
            df_chart['decada'] = (df_chart['ano_isolado'].astype(int) // 10) * 10

            grouped = (
                df_chart.groupby(['decada', 'pais_normalizado'])
                .size()
                .reset_index(name='contagem')
            )

            fig = px.bar(
                grouped,
                x='decada',
                y='contagem',
                color='pais_normalizado',
                labels={
                    'decada': 'Década',
                    'contagem': 'Nº de Documentos',
                    'pais_normalizado': 'País de Nascimento'
                },
                template='plotly_dark',
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                xaxis_title="Década",
                yaxis_title="Documentos",
                legend_title="País de Nascimento",
                bargap=0.15,
                height=500,
                font=dict(family="Consolas, Courier New, monospace"),
            )
            
            event = st.plotly_chart(
                fig, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="points"
            )
            
            selected_countries = set()
            if event and "selection" in event and "points" in event.selection:
                for pt in event.selection["points"]:
                    if "curveNumber" in pt:
                        trace_name = fig.data[pt["curveNumber"]].name
                        selected_countries.add(trace_name)
        else:
            st.info("Nenhum documento com data válida encontrado para gerar o gráfico temporal.")
            selected_countries = set()

        st.divider()

        # ── Table B: Most prolific authors in the current filter ──
        st.markdown("#### Autores Mais Prolíficos no Recorte Atual")
        st.markdown("*Top 50 remetentes por volume de documentos, com metadados sociais e geográficos.*")

        if not df_filtered.empty:
            df_table = df_filtered
            if selected_countries:
                df_table = df_table[df_table['pais_normalizado'].isin(selected_countries)]
                st.info(f"Filtro ativo no gráfico acima. Mostrando autores de: **{', '.join(sorted(selected_countries))}**")

            author_stats = (
                df_table.groupby('sender_name')
                .agg(
                    total_docs=('sender_name', 'size'),
                    categoria_social=('sender_category', 'first'),
                    pais=('pais_normalizado', 'first'),
                    cidade=('cidade_nascimento', 'first'),
                    confianca_bio=('confianca', 'first'),
                    conferir=('link_fonte', 'first'),
                )
                .sort_values(by='total_docs', ascending=False)
                .head(50)
                .reset_index()
            )
            
            author_stats['conferir'] = author_stats['conferir'].apply(
                lambda x: x if pd.notnull(x) and str(x).strip() != "" else None
            )

            author_stats.columns = [
                'Autor/Remetente', 'Nº de Documentos', 'Categoria Social',
                'País de Nascimento', 'Cidade de Nascimento', 'Confiança', 'Conferir'
            ]

            st.dataframe(
                author_stats,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nº de Documentos": st.column_config.NumberColumn(format="%d"),
                    "Conferir": st.column_config.LinkColumn("Conferir", display_text="link"),
                }
            )
        else:
            st.info("Nenhum documento encontrado com os filtros atuais para gerar a tabela de autores.")


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        O presente trabalho foi realizado com apoio da Coordenação de 
        Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES).<br><br>
        Desenvolvido por Saulo R. Em caso de dúvidas ou erros, contatar: 
        <a href="mailto:saulorrp@gmail.com" style="color: gray; text-decoration: underline;">eu@saulo.ru</a>
    </div>
    """,
    unsafe_allow_html=True
)
