import streamlit as st
import pandas as pd
import re
import json
from sentence_transformers import SentenceTransformer, util
import torch
from fpdf import FPDF
import numpy as np
import pickle
import urllib.parse
from rank_bm25 import BM25Okapi

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Classificador AHU Sul", layout="wide")

# --- GLOBAL CSS ---
# Monospace font (EDSS/Terminal standard aesthetic) +
# Floating Action Button (FAB) pinned to the bottom-left corner
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Consolas', 'Courier New', monospace !important;
    }

    /* ── Floating Action Button (bottom-left) ──
       Targets the container wrapping the very last button in the app
       which we render at the end of the page with a unique key. ── */
    div.fab-fixed-wrapper {
        position: fixed;
        bottom: 2rem;
        left: 2rem;
        z-index: 999999;
    }
    div.fab-fixed-wrapper button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        color: #e0e0ff !important;
        border: 1px solid #334477 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.4rem !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 0.85rem !important;
        font-weight: bold !important;
        cursor: pointer !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    div.fab-fixed-wrapper button:hover {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a6e 100%) !important;
        box-shadow: 0 6px 28px rgba(15, 52, 96, 0.7) !important;
        transform: translateY(-2px) !important;
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
    Loads the main JSON catalog, computes enrichment columns
    (ano_isolado, autoria_verificada), and initializes the BM25 engine.
    """
    df = pd.read_json('ahu_sul_catalog.json')

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
# 2. PDF GENERATOR (DOCUMENT DOSSIER)
# ==========================================

class PDF(FPDF):
    """Custom FPDF class to handle headers, footers, and native hyperlinking."""
    def header(self):
        self.set_font('Courier', 'B', 14)
        self.cell(0, 10, 'Dossiê Documental: Catálogo do AHU para a macrorregião Sul', ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Courier', 'I', 10)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')


def create_pdf(dataframe, search_params):
    """
    Generates an ABNT-compliant PDF report containing the filtered documents,
    their sociolinguistic metadata, and clickable URLs to the official archives.
    """
    pdf = PDF()
    pdf.set_margins(left=30, top=30, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    line_height = 6

    def safe_write(text, style='', size=11):
        """Helper to handle Latin-1 encoding issues with Portuguese characters."""
        pdf.set_x(30)
        pdf.set_font('Courier', style, size)
        cleaned = str(text).replace('\n', ' ').strip()
        encoded = cleaned.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, line_height, encoded)

    def write_link_line(label, code, link_text, url):
        """Helper to create clickable hyperlinks within the PDF."""
        pdf.set_x(30)
        pdf.set_font('Courier', 'B', 11)
        label_encoded = f"{label}: {code} ".encode('latin-1', 'replace').decode('latin-1')
        pdf.write(line_height, label_encoded)

        pdf.set_text_color(0, 0, 255)
        pdf.set_font('Courier', 'U', 11)
        link_encoded = link_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.write(line_height, link_encoded, url)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Courier', '', 11)
        pdf.ln(line_height + 2)

    # --- PDF Header & Context ---
    safe_write("Sobre a Elaboração deste Dossiê", style='B')
    intro_text = (
        "Este dossiê foi gerado automaticamente pelo Classificador de Obras do Catálogo do Arquivo "
        "Histórico Ultramarino (AHU) para a Macrorregião Sul do Brasil. O sistema utiliza extração de "
        "metadados e processamento de linguagem natural (DeepSeek) para analisar os resumos arquivísticos. "
        "Os documentos são classificados por tipologia, hierarquia comunicativa e um Score de Relevância Sociolinguística Potencial (SRSP), "
        "que estima a probabilidade de o texto original conter evidências de sintaxe diacrônica e oralidade do "
        "português brasileiro colonial."
    )
    safe_write(intro_text)
    pdf.ln(5)

    # --- Print Search Parameters for Reproducibility ---
    safe_write("Parâmetros de Busca Utilizados:", style='B')
    safe_write(f"- Busca Semântica: {search_params['query']}", size=10)
    safe_write(f"- Perfil (Lente): {search_params['lente']}", size=10)
    safe_write(f"- Regiões: {search_params['regioes']}", size=10)
    safe_write(f"- Relevância Sociolinguística Potencial: {search_params['sv_range']}", size=10)
    safe_write(f"- Direção da Comunicação: {search_params['vetores']}", size=10)
    safe_write(f"- Categoria do Remetente: {search_params['categorias']}", size=10)
    safe_write(f"- Rigor Semântico (Corte): {search_params['limiar']}", size=10)
    pdf.ln(10)

    # --- Document Iteration ---
    for idx, row in dataframe.iterrows():
        crav_code = row.get('reference_code', 'Sem Cota CRAV')
        new_code = row.get('new_code', 'Sem Cota')
        old_code = row.get('old_code', 'N/A')
        typology = row.get('extracted_typology', 'N/A')
        folder = row.get('folder', 'Local Desconhecido')
        description = row.get('description', '')
        sender_name = row.get('sender_name', 'N/A')
        sender_category = row.get('sender_category', 'N/A')
        recipient_name = row.get('recipient_name', 'N/A')
        vector = row.get('vector', 'N/A')
        score = row.get('vernacular_score', 0.0)
        reasoning = row.get('sociolinguistic_reasoning_by_deepseek_v3', '')

        scribe_raw = row.get('scribe_mediation_likely', False)
        scribe_text = "Provável" if str(scribe_raw).lower() in ['true', '1', 'sim'] else "Pouco provável"

        safe_write("> Referência", style='B')

        if crav_code.startswith("PT/AHU"):
            encoded_crav = urllib.parse.quote(crav_code, safe='')
            crav_url = f"https://digitarq.arquivos.pt/search?query={encoded_crav}&isAdvancedSearch=false"
            write_link_line("Código de Referência", crav_code, "Busca no Digitarq", crav_url)
        else:
            safe_write(f"Código de Referência: {crav_code}", style='B')

        resgate_url = "https://resgate.bn.gov.br/"
        write_link_line("Código Atual", new_code, "Busca no Projeto Resgate", resgate_url)

        safe_write(f"Código Antigo: {old_code}", style='B')
        safe_write(f"Tipologia: {typology}")
        pdf.ln(2)

        safe_write("> Localização no Arquivo", style='B')
        safe_write(f"Pasta: {folder}")
        safe_write(f"Descrição: {description}")
        pdf.ln(2)

        safe_write("> Protagonistas", style='B')
        safe_write(f"Remetente: {sender_name}")
        safe_write(f"Categoria do Remetente: {sender_category}")
        safe_write(f"Destinatário: {recipient_name}")
        pdf.ln(2)

        safe_write("> Análise Sociolinguística Automatizada", style='B')
        safe_write(f"Vetor de Comunicação: {vector}")
        safe_write(f"Mediação por Escrivão: {scribe_text}")
        safe_write(f"Score de Probabilidade de Vernacularidade: {score:.1f}")
        safe_write(f"Justificativa do Score: {reasoning}")

        pdf.ln(5)
        pdf.set_x(30)
        pdf.set_font('Courier', 'B', 12)
        pdf.cell(0, line_height, "-" * 50, ln=1, align='C')
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1')


# ==========================================
# 3. DIACHRONIC SAMPLES HELPER FUNCTIONS
# ==========================================

@st.cache_data
def get_verified_authors(_df):
    """
    Returns a list of (author_name, doc_count) tuples for authors
    where autoria_verificada == True, sorted descending by count.
    """
    verified = _df[_df['autoria_verificada'] == True]
    counts = verified['sender_name'].value_counts()
    return [(name, count) for name, count in counts.items()]


@st.cache_data
def get_author_docs(_df, author_name):
    """
    Returns the chronologically sorted (by ano_isolado ASC) documents
    for a given author, filtered to autoria_verificada == True and ano_isolado not null.
    """
    mask = (
        (_df['sender_name'] == author_name) &
        (_df['autoria_verificada'] == True) &
        (_df['ano_isolado'].notna())
    )
    return _df[mask].sort_values('ano_isolado', ascending=True).reset_index(drop=True)


def build_collection_plan(author_name, author_df, batch_size=10):
    """
    Builds the JSON-serializable collection plan for a given author's documents.
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
                "descricao": str(row.get('description', ''))
            })
        sections.append({
            "id_secao": (i // batch_size) + 1,
            "documentos": docs
        })

    years = author_df['ano_isolado'].dropna()
    plan = {
        "autor_alvo": author_name,
        "recorte_temporal": {
            "inicio": int(years.min()) if len(years) > 0 else None,
            "fim": int(years.max()) if len(years) > 0 else None
        },
        "total_documentos": total,
        "secoes_coleta": sections
    }
    return plan


# ==========================================
# 4. SESSION STATE INITIALIZATION
# ==========================================

if 'show_samples_panel' not in st.session_state:
    st.session_state.show_samples_panel = False


# ==========================================
# 5. UI FLOW — CONDITIONAL ROUTING
# ==========================================

if st.session_state.show_samples_panel:
    # ─────────────────────────────────────────
    #  PANEL: DIACHRONIC SAMPLES + PDF EXPORT
    # ─────────────────────────────────────────

    # Back button
    if st.button(":material/arrow_back: Voltar ao Motor de Busca"):
        st.session_state.show_samples_panel = False
        st.rerun()

    # Panel header
    st.markdown("""
    <div class="panel-header">
        <h2>Amostras Diacronicas & Exportacao</h2>
        <p>Ferramentas de selecao de corpus para Linguistica Historica e exportacao de dossie documental.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── 5A. PDF EXPORT (relocated from main screen) ──
    st.subheader(":material/picture_as_pdf: Exportar Dossiê Documental (PDF)")
    st.markdown("*Retorne ao motor de busca para aplicar filtros. Aqui, exporte o resultado da última busca ativa.*")

    # We need the filtered results_df — compute it here too
    # (duplicated filter logic to keep both paths self-contained)
    with st.sidebar:
        st.header(":material/tune: Perfis de Busca Predefinidos")
        lente = st.radio(
            "Selecione uma lente metodológica:",
            ["Busca Livre (Personalizada)",
             "Vozes Marginalizadas & História Social",
             "Sintaxe Diacrônica (Alto SRSP)",
             "Máquina Administrativa (Top-Down)"],
            key="lente_panel"
        )

        st.divider()
        st.header(":material/filter_alt: Filtro de Seções do AHU")
        todas_regioes = df['folder'].unique().tolist()
        regioes_selecionadas = st.multiselect("Regiões/Capitanias:", todas_regioes, default=todas_regioes, key="regioes_panel")

        st.header(":material/groups_2: Filtros Sociolinguísticos")

        min_score = 0.0
        max_score = 1.0
        vetor_padrao = ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"]
        categorias_disponiveis = df['sender_category'].fillna('Unknown').unique().tolist()
        remetente_padrao = categorias_disponiveis.copy()

        if lente == "Vozes Marginalizadas & História Social":
            vetor_padrao = ["Bottom-Up"]
            remetente_padrao = ["Commoner", "Marginalized", "Low Military"]
        elif lente == "Sintaxe Diacrônica (Alto SRSP)":
            min_score = 0.7
        elif lente == "Máquina Administrativa (Top-Down)":
            vetor_padrao = ["Top-Down", "Horizontal"]
            remetente_padrao = ["Metropolitan Elite", "Local Elite"]

        score_range = st.slider("Score de Relevância Sociolinguística Potencial (SRSP):", 0.0, 1.0, (min_score, max_score), step=0.1, key="srsp_panel")

        st.markdown("""
            <div style="display: flex; text-align: center; font-size: 0.75em; color: gray; margin-top: -15px; margin-bottom: 15px;">
                <div style="flex: 0.35; border-right: 2px solid #555;">0.0 - 0.3<br>Formulaico</div>
                <div style="flex: 0.30; border-right: 2px solid #555;">0.4 - 0.6<br>Moderado</div>
                <div style="flex: 0.35;">0.7 - 1.0<br>Potencial</div>
            </div>
        """, unsafe_allow_html=True)

        vetores = st.multiselect("Direção da Comunicação:", ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"], default=vetor_padrao, key="vetores_panel")
        categorias = st.multiselect("Perfil Social do Remetente:", categorias_disponiveis, default=remetente_padrao, key="categorias_panel")

    # Apply filters
    df_filter = df.copy()
    df_filter['vector'] = df_filter['vector'].fillna('Unknown')
    df_filter['sender_category'] = df_filter['sender_category'].fillna('Unknown')

    mask_panel = (
        (df_filter['folder'].isin(regioes_selecionadas)) &
        (df_filter['vector'].isin(vetores)) &
        (df_filter['sender_category'].isin(categorias)) &
        (df_filter['vernacular_score'] >= score_range[0]) &
        (df_filter['vernacular_score'] <= score_range[1])
    )
    results_df_panel = df_filter[mask_panel].sort_values(by='vernacular_score', ascending=False)

    if not results_df_panel.empty:
        col_exp, _ = st.columns([2, 8])
        with col_exp:
            limite_str = st.text_input("Quantidade a exportar:", value="50", max_chars=4, key="limite_panel")

        try:
            limite_exportacao = int(limite_str.strip())
            if limite_exportacao <= 0:
                limite_exportacao = 50
        except ValueError:
            limite_exportacao = 50
            st.error("Digite apenas inteiros. Retornando ao padrão (50).")

        export_df = results_df_panel.head(limite_exportacao)

        regioes_str = ", ".join(regioes_selecionadas) if regioes_selecionadas else "Nenhuma"
        if len(regioes_selecionadas) == len(todas_regioes):
            regioes_str = "Todas"

        current_params = {
            "query": "Nenhuma restrição semântica (painel de amostras)",
            "lente": lente,
            "regioes": regioes_str,
            "sv_range": f"{score_range[0]:.1f} a {score_range[1]:.1f}",
            "vetores": ", ".join(vetores) if vetores else "Nenhum",
            "categorias": ", ".join(categorias) if categorias else "Nenhuma",
            "limiar": "N/A (filtro por metadados)"
        }

        pdf_bytes = create_pdf(export_df, current_params)
        st.download_button(
            label=f":material/download: Baixar Dossiê (Top {len(export_df)} documentos)",
            data=pdf_bytes,
            file_name="Dossie_AHU.pdf",
            mime="application/pdf",
            key="pdf_download_panel"
        )
    else:
        st.warning("Nenhum documento encontrado com os filtros atuais da sidebar.")

    st.divider()

    # ── 5B. DIACHRONIC SAMPLES TOOL ──
    st.subheader(":material/history_edu: Ferramenta de Criação de Amostras Diacrônicas")
    st.markdown("*Selecione um autor com autoria verificada no catálogo para gerar um plano de coleta organizado cronologicamente.*")

    # Author selectbox
    authors = get_verified_authors(df)

    if not authors:
        st.warning("Nenhum autor com autoria verificada encontrado no catálogo.")
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
        author_docs = get_author_docs(df, selected_author)

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
                    f"Secao {sec_idx + 1} de {num_sections}  |  Docs {start + 1}-{end}  |  {int(year_start)}-{int(year_end)}",
                    expanded=(sec_idx == 0)
                ):
                    display_df = batch[['ano_isolado', 'reference_code', 'extracted_typology', 'description']].copy()
                    display_df.columns = ['Ano', 'Referência CRAV', 'Tipologia', 'Resumo']
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

            # JSON export
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
    #  MAIN SCREEN: SEARCH ENGINE (unchanged logic)
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

    **4. A Geração de Dossiê Documental (Exportar PDF):** Acesse o painel de Amostras Diacrônicas (botão no canto inferior esquerdo) para gerar e baixar o dossiê em PDF.

    **5. Acesso Direto aos Acervos:** A ferramenta gera automaticamente links para as plataformas oficiais. Ao expandir um resultado na tela, você pode usar o Código de Referência para abrir a ficha de controle arquivístico no DigitArq, ou usar o Código Atual para buscar as imagens microfilmadas no portal do Projeto Resgate (Biblioteca Nacional).
    """)
    st.divider()

    # --- Search Bar & Semantic Rigor ---
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

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header(":material/tune: Perfis de Busca Predefinidos")
        lente = st.radio(
            "Selecione uma lente metodológica:",
            ["Busca Livre (Personalizada)",
             "Vozes Marginalizadas & História Social",
             "Sintaxe Diacrônica (Alto SRSP)",
             "Máquina Administrativa (Top-Down)"]
        )

        st.divider()
        st.header(":material/filter_alt: Filtro de Seções do AHU")
        todas_regioes = df['folder'].unique().tolist()
        regioes_selecionadas = st.multiselect("Regiões/Capitanias:", todas_regioes, default=todas_regioes)

        st.header(":material/groups_2: Filtros Sociolinguísticos")

        min_score = 0.0
        max_score = 1.0
        vetor_padrao = ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"]
        categorias_disponiveis = df['sender_category'].fillna('Unknown').unique().tolist()
        remetente_padrao = categorias_disponiveis.copy()

        if lente == "Vozes Marginalizadas & História Social":
            vetor_padrao = ["Bottom-Up"]
            remetente_padrao = ["Commoner", "Marginalized", "Low Military"]
        elif lente == "Sintaxe Diacrônica (Alto SRSP)":
            min_score = 0.7
        elif lente == "Máquina Administrativa (Top-Down)":
            vetor_padrao = ["Top-Down", "Horizontal"]
            remetente_padrao = ["Metropolitan Elite", "Local Elite"]

        # Slider limpo e minimalista apenas com os números
        score_range = st.slider("Score de Relevância Sociolinguística Potencial (SRSP):", 0.0, 1.0, (min_score, max_score), step=0.1)

        # Barra visual criando os "vincos" de separação entre as categorias
        st.markdown("""
            <div style="display: flex; text-align: center; font-size: 0.75em; color: gray; margin-top: -15px; margin-bottom: 15px;">
                <div style="flex: 0.35; border-right: 2px solid #555;">0.0 - 0.3<br>Formulaico</div>
                <div style="flex: 0.30; border-right: 2px solid #555;">0.4 - 0.6<br>Moderado</div>
                <div style="flex: 0.35;">0.7 - 1.0<br>Potencial</div>
            </div>
        """, unsafe_allow_html=True)

        vetores = st.multiselect("Direção da Comunicação:", ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"], default=vetor_padrao)
        categorias = st.multiselect("Perfil Social do Remetente:", categorias_disponiveis, default=remetente_padrao)

    # Apply Metadata Filters
    df_filter = df.copy()
    df_filter['vector'] = df_filter['vector'].fillna('Unknown')
    df_filter['sender_category'] = df_filter['sender_category'].fillna('Unknown')

    mask = (
        (df_filter['folder'].isin(regioes_selecionadas)) &
        (df_filter['vector'].isin(vetores)) &
        (df_filter['sender_category'].isin(categorias)) &
        (df_filter['vernacular_score'] >= score_range[0]) &
        (df_filter['vernacular_score'] <= score_range[1])
    )

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

        mask = mask & (df_filter['semantic_score'] >= limiar_semantico)
        results_df = df_filter[mask].sort_values(by='semantic_score', ascending=False)
    else:
        results_df = df_filter[mask].sort_values(by='vernacular_score', ascending=False)

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
                st.markdown("---")
                st.markdown(f"**Resumo do Arquivo (de autoria do AHU):**\n{row.get('description', '')}")
                st.markdown("---")
                reasoning = row.get('sociolinguistic_reasoning_by_deepseek_v3', '')
                st.markdown(f"**Justificativa Analítica para o Score (LLM):**\n*{reasoning}*")

        if len(results_df) > 50:
            st.info(f"Mostrando os 50 resultados mais relevantes no navegador de um total de {len(results_df)}. Acesse o painel de Amostras para exportar mais em PDF.")
    else:
        st.warning("Nenhum documento encontrado com os filtros atuais. Experimente diminuir o valor de corte da relevância ou aumentar o intervalo do Score.")


# ==========================================
# 6. FOOTER & FLOATING ACTION BUTTON
# ==========================================

st.divider()

# Footer
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

# ── Floating Action Button ──
# We render a normal st.button, then inject JS to move its container into
# a fixed-position wrapper div. This preserves full Streamlit interactivity.
if st.button("Amostras Diacronicas", key="fab_amostras"):
    st.session_state.show_samples_panel = True
    st.rerun()

st.markdown("""
<script>
(function() {
    // Find the FAB button by its inner text
    const buttons = window.parent.document.querySelectorAll('button[kind="secondary"]');
    for (const btn of buttons) {
        if (btn.textContent.trim() === 'Amostras Diacronicas') {
            // Walk up to the stElementContainer
            let container = btn.closest('[data-testid="stElementContainer"]');
            if (container && !container.parentElement.classList.contains('fab-fixed-wrapper')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'fab-fixed-wrapper';
                container.parentElement.insertBefore(wrapper, container);
                wrapper.appendChild(container);
            }
            break;
        }
    }
})();
</script>
""", unsafe_allow_html=True)