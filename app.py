import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from fpdf import FPDF
import numpy as np

# ==========================================
# PAGE CONFIGURATION & ACADEMIC CSS
# ==========================================
st.set_page_config(page_title="Catálogo do AHU para o Cone Sul+", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🚀 ULTRA-FAST DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_parquet('mar_do_sul.parquet')
    
    # Prevenção extra para garantir que a pasta nunca seja nula
    df['folder'] = df['folder'].fillna('Sem região definida')
    
    if 'vernacular_score' in df.columns:
        df['vernacular_score'] = pd.to_numeric(df['vernacular_score'], errors='coerce').fillna(0)
        df['vernacular_score'] = df['vernacular_score'] / 10.0
    else:
        df['vernacular_score'] = 0.0
    return df

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_precomputed_embeddings():
    return torch.load('mar_do_sul_embeddings.pt', map_location=torch.device('cpu'), weights_only=True)

df = load_data()
model = load_semantic_model()
corpus_embeddings = load_precomputed_embeddings()

# ==========================================
# 📄 PDF GENERATOR CLASS (Normas ABNT)
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 14)
        self.cell(0, 10, 'Dossiê Documental: Catálogo do AHU para o Cone Sul+', ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15) # Margem inferior do rodapé
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

def create_pdf(dataframe, search_params):
    pdf = PDF()
    # MARGENS ABNT EM MILÍMETROS: Esquerda 30, Cima 30, Direita 20.
    pdf.set_margins(left=30, top=30, right=20)
    # Margem inferior de 20mm para quebra de página
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    line_height = 6 
    
    def safe_write(text, style='', size=12):
        """Escreve texto com segurança, tratando caracteres especiais."""
        pdf.set_x(30) # Força o retorno para a margem esquerda de 30mm (ABNT)
        pdf.set_font('Times', style, size)
        cleaned = str(text).replace('\n', ' ').strip()
        encoded = cleaned.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, line_height, encoded)

    # 1. DEFAULT EXPLANATION
    safe_write("Sobre a Elaboração deste Dossiê", style='B')
    intro_text = (
        "Este dossiê foi gerado automaticamente pelo Classificador de Obras do Catálogo do Arquivo "
        "Histórico Ultramarino (AHU) para a Macro Região Sul do Brasil. O sistema utiliza extração de "
        "metadados e processamento de linguagem natural (DeepSeek v3) para analisar os resumos arquivísticos. "
        "Os documentos são classificados por tipologia, hierarquia comunicativa e um Score de Vernacularidade (SV), "
        "que estima a probabilidade de o texto original conter evidências de sintaxe diacrônica e oralidade do "
        "português brasileiro colonial."
    )
    safe_write(intro_text)
    pdf.ln(5)

    # 2. SEARCH PARAMETERS
    safe_write("Parâmetros de Busca Utilizados:", style='B')
    safe_write(f"- Busca Semântica: {search_params['query']}", size=11)
    safe_write(f"- Perfil (Lente): {search_params['lente']}", size=11)
    safe_write(f"- Regiões: {search_params['regioes']}", size=11)
    safe_write(f"- Score de Vernacularidade (SV): {search_params['sv_range']}", size=11)
    safe_write(f"- Direção da Comunicação: {search_params['vetores']}", size=11)
    safe_write(f"- Categoria do Remetente: {search_params['categorias']}", size=11)
    pdf.ln(10)

    # 3. DOCUMENT LIST
    for idx, row in dataframe.iterrows():
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
        if isinstance(scribe_raw, bool):
            scribe_text = "Provável" if scribe_raw else "Pouco provável"
        else:
            scribe_text = "Provável" if str(scribe_raw).lower() in ['true', '1', 'sim'] else "Pouco provável"

        # Trocado "•" por ">" para evitar conflito de codificação latin-1 no PDF
        safe_write("> Referência", style='B')
        safe_write(f"Código: {new_code}", style='B')
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
        safe_write(f"Score de vernacularidade: {score:.1f}")
        safe_write(f"Justificativa do Score: {reasoning}")
        
        pdf.ln(5)
        pdf.set_x(30)
        pdf.set_font('Times', 'B', 12)
        pdf.cell(0, line_height, "-" * 50, ln=1, align='C')
        pdf.ln(5)
        
    return bytes(pdf.output())

# ==========================================
# THE PRESENTATION 
# ==========================================
st.title("Classificador de Obras do Catálogo do AHU para Documentos da Macro Região Sul do Brasil")

st.markdown("""
### Sobre esta ferramenta
**1. Motor de Busca e Triagem:** Este sistema não contém as imagens digitalizadas dos manuscritos originais. Ele funciona como um classificador avançado para os resumos do catálogo do **Arquivo Histórico Ultramarino (AHU)**. O objetivo é permitir que pesquisadores cruzem recortes geográficos, temas históricos e variáveis sociolinguísticas para obter as **cotas arquivísticas exatas** (ex: *AHU_ACL_CU_...*) antes de acessar o arquivo físico ou o Projeto Resgate.

**2. O Score de Vernacularidade (SV):** Cada documento teve sua descrição processada pelo DeepSeek (v3) para a atribuição de um valor numérico indicativo da probabilidade de o documento conter indícios de vernacularidade. Esse valor varia entre **0.0 a 1.0**.
* Um **SV próximo a 1.0** indica alta probabilidade de que o manuscrito original contenha marcas de oralidade, inovações sintáticas e vazamento do português vernáculo brasileiro colonial.
* Um **SV próximo a 0.0** indica baixa probabilidade (fórmulas diplomáticas rígidas, linguagem erudita metropolitana ou forte padronização de notários).
""")

st.divider()

# ==========================================
# THE SEMANTIC ENGINE
# ==========================================
st.subheader("Busca Semântica Inteligente")
st.markdown("*Digite um conceito, tema ou evento histórico. O motor buscará documentos pelo significado contextual.*")

query = st.text_input("Ex: 'conflitos de terra', 'deserção de soldados', 'escassez de farinha':")

st.divider()

# ==========================================
# THE SIDEBAR (LENSES & FILTERS)
# ==========================================
with st.sidebar:
    st.header("Perfis de Busca (Lentes)")
    lente = st.radio(
        "Selecione uma lente metodológica:",
        ["Busca Livre (Personalizada)", 
         "Vozes Marginalizadas & História Social", 
         "Sintaxe Diacrônica (Alto SV)", 
         "Máquina Administrativa (Top-Down)"]
    )
    
    st.divider()
    st.header("Filtros Manuais")
    
    st.subheader("Contexto Espacial")
    todas_regioes = df['folder'].unique().tolist()
    regioes_selecionadas = st.multiselect("Regiões/Capitanias:", todas_regioes, default=todas_regioes)
    
    st.subheader("Variáveis Sociolinguísticas")
    
    min_score = 0.0
    max_score = 1.0
    vetor_padrao = ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"]
    
    categorias_disponiveis = df['sender_category'].fillna('Unknown').unique().tolist()
    remetente_padrao = categorias_disponiveis.copy()
    
    if lente == "Vozes Marginalizadas & História Social":
        vetor_padrao = ["Bottom-Up"]
        remetente_padrao = ["Commoner", "Marginalized", "Low Military"]
    elif lente == "Sintaxe Diacrônica (Alto SV)":
        min_score = 0.7
    elif lente == "Máquina Administrativa (Top-Down)":
        vetor_padrao = ["Top-Down", "Horizontal"]
        remetente_padrao = ["Metropolitan Elite", "Local Elite"]
        
    score_range = st.slider("Score de Vernacularidade (SV):", 0.0, 1.0, (min_score, max_score), step=0.1)
    
    vetores = st.multiselect("Direção da Comunicação:", 
                             ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"], 
                             default=vetor_padrao)
    
    categorias = st.multiselect("Perfil Social do Remetente:", 
                                categorias_disponiveis, 
                                default=remetente_padrao)

# ==========================================
# FILTERING & SEARCH LOGIC
# ==========================================
df_filter = df.copy()
df_filter['vector'] = df_filter['vector'].fillna('Unknown')
df_filter['sender_category'] = df_filter['sender_category'].fillna('Unknown')

# MÁSCARA BASE (Filtros Laterais)
mask = (
    (df_filter['folder'].isin(regioes_selecionadas)) &
    (df_filter['vector'].isin(vetores)) &
    (df_filter['sender_category'].isin(categorias)) &
    (df_filter['vernacular_score'] >= score_range[0]) &
    (df_filter['vernacular_score'] <= score_range[1])
)

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    df_filter['semantic_score'] = cosine_scores.cpu().numpy()
    
    # LIMIAR DE CORTE DA BUSCA SEMÂNTICA (0.20 = 20% de alinhamento contextual mínimo)
    mask = mask & (df_filter['semantic_score'] >= 0.20)
    
    results_df = df_filter[mask].sort_values(by='semantic_score', ascending=False)
else:
    results_df = df_filter[mask].sort_values(by='vernacular_score', ascending=False)

# ==========================================
# EXPORT MODULE
# ==========================================
st.subheader("📄 Exportar PDF com o Dossiê Documental")
st.markdown("*Use os filtros e a busca para isolar um conjunto de documentos. Em seguida, escolha a quantidade e clique abaixo para baixar um PDF formatado (Normas ABNT).*")

if not results_df.empty:
    # Seletor de quantidade para exportação
    limite_exportacao = st.selectbox(
        "Quantidade de documentos para exportar:",
        [10, 30, 60, 100],
        index=1 # Padrão é 30
    )
    
    export_df = results_df.head(limite_exportacao)
    
    regioes_str = ", ".join(regioes_selecionadas) if regioes_selecionadas else "Nenhuma"
    if len(regioes_selecionadas) == len(todas_regioes):
        regioes_str = "Todas"
    
    current_params = {
        "query": query if query else "Nenhuma restrição semântica",
        "lente": lente,
        "regioes": regioes_str,
        "sv_range": f"{score_range[0]:.1f} a {score_range[1]:.1f}",
        "vetores": ", ".join(vetores) if vetores else "Nenhum",
        "categorias": ", ".join(categorias) if categorias else "Nenhuma"
    }
    
    pdf_bytes = create_pdf(export_df, current_params)
    
    st.download_button(
        label=f"Baixar Dossiê (Top {len(export_df)} documentos)",
        data=pdf_bytes,
        file_name="Dossie_AHU.pdf",
        mime="application/pdf"
    )
else:
    st.warning("Nenhum documento encontrado para exportar com os filtros atuais.")

st.divider()

# ==========================================
# RESULTS DISPLAY
# ==========================================
st.subheader(f"Resultados Encontrados: {len(results_df)} documentos")

if not results_df.empty:
    for idx, row in results_df.head(20).iterrows():
        score = row.get('vernacular_score', 0.0)
        date_id = row.get('document_id_and_date', 'Sem Data')
        folder = row.get('folder', 'Local Desconhecido')
        
        # Opcional: mostrar o score semântico no título se a busca estiver ativa
        if query:
            sem_score = row.get('semantic_score', 0.0)
            expander_title = f"Relevância: {sem_score:.2f} | SV: {score:.1f} | {date_id} | {folder}"
        else:
            expander_title = f"SV: {score:.1f} | {date_id} | {folder}"
        
        with st.expander(expander_title):
            st.markdown(f"**Cota:** {row.get('new_code', 'N/A')}")
            st.markdown(f"**Tipologia:** {row.get('extracted_typology', 'N/A')}")
            st.markdown(f"**Remetente:** {row.get('sender_name', 'N/A')} *(Classe: {row.get('sender_category', 'N/A')})*")
            st.markdown(f"**Hierarquia:** {row.get('vector', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**Resumo do Arquivo:**\n{row.get('description', '')}")
            st.markdown("---")
            reasoning = row.get('sociolinguistic_reasoning_by_deepseek_v3', '')
            st.markdown(f"**Análise Sociolinguística:**\n*{reasoning}*")
            
    if len(results_df) > 20:
        st.info(f"Mostrando os 20 resultados mais relevantes no navegador de um total de {len(results_df)}. Ajuste o seletor acima para incluir mais no PDF.")