import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from fpdf import FPDF
import numpy as np
import pickle
import urllib.parse
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Catálogo do AHU para a macrorregião Sul", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 1. NOVA ARQUITETURA DE DADOS ---
@st.cache_data
def load_data():
    df = pd.read_json('ahu_documents_phase2_by_deepseek_v7_crav.json')
    
    df['folder'] = df['folder'].fillna('Sem região definida')
    
    if 'vernacular_score' in df.columns:
        df['vernacular_score'] = pd.to_numeric(df['vernacular_score'], errors='coerce').fillna(0)
        df['vernacular_score'] = df['vernacular_score'] / 10.0
    else:
        df['vernacular_score'] = 0.0
        
    corpus = (df['description'].fillna('') + " " + 
              df['sender_name'].fillna('') + " " + 
              df['folder'].fillna('')).str.lower().tolist()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return df, bm25

@st.cache_data
def load_semantic_index():
    with open('ahu_semantic_index.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

@st.cache_resource
def load_semantic_model(model_name):
    return SentenceTransformer(model_name)

df, bm25_engine = load_data()
semantic_index = load_semantic_index()

model_name = semantic_index.get('model_used', 'intfloat/multilingual-e5-large')
model = load_semantic_model(model_name)

device = model.device
corpus_embeddings = torch.tensor(semantic_index['embeddings']).to(device)
reference_codes_list = semantic_index['reference_codes']


# --- 2. GERADOR DE PDF COM LINKS NATIVOS ---
class PDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 14)
        self.cell(0, 10, 'Dossiê Documental: Catálogo do AHU para a macrorregião Sul', ln=1, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

def create_pdf(dataframe, search_params):
    pdf = PDF()
    pdf.set_margins(left=30, top=30, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    line_height = 6 
    
    def safe_write(text, style='', size=12):
        pdf.set_x(30) 
        pdf.set_font('Times', style, size)
        cleaned = str(text).replace('\n', ' ').strip()
        encoded = cleaned.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, line_height, encoded)

    def write_link_line(label, code, link_text, url):
        """Escreve uma linha híbrida com texto em negrito e link clicável em azul no PDF."""
        pdf.set_x(30)
        # Parte em negrito (O Código em si)
        pdf.set_font('Times', 'B', 12)
        label_encoded = f"{label}: {code} ".encode('latin-1', 'replace').decode('latin-1')
        pdf.write(line_height, label_encoded)
        
        # Parte do Link (Azul e sublinhado)
        pdf.set_text_color(0, 0, 255)
        pdf.set_font('Times', 'U', 12)
        link_encoded = link_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.write(line_height, link_encoded, url)
        
        # Reseta as cores e fonte para as próximas linhas
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Times', '', 12)
        pdf.ln(line_height + 2)

    safe_write("Sobre a Elaboração deste Dossiê", style='B')
    intro_text = (
        "Este dossiê foi gerado automaticamente pelo Classificador de Obras do Catálogo do Arquivo "
        "Histórico Ultramarino (AHU) para a Macrorregião Sul do Brasil. O sistema utiliza extração de "
        "metadados e processamento de linguagem natural (DeepSeek v3) para analisar os resumos arquivísticos. "
        "Os documentos são classificados por tipologia, hierarquia comunicativa e um Score de Probabilidade de Vernacularidade (SPV), "
        "que estima a probabilidade de o texto original conter evidências de sintaxe diacrônica e oralidade do "
        "português brasileiro colonial."
    )
    safe_write(intro_text)
    pdf.ln(5)

    safe_write("Parâmetros de Busca Utilizados:", style='B')
    safe_write(f"- Busca Semântica: {search_params['query']}", size=11)
    safe_write(f"- Perfil (Lente): {search_params['lente']}", size=11)
    safe_write(f"- Regiões: {search_params['regioes']}", size=11)
    safe_write(f"- Score de Probabilidade de Vernacularidade (SPV): {search_params['sv_range']}", size=11)
    safe_write(f"- Direção da Comunicação: {search_params['vetores']}", size=11)
    safe_write(f"- Categoria do Remetente: {search_params['categorias']}", size=11)
    safe_write(f"- Rigor Semântico (Corte): {search_params['limiar']}", size=11)
    pdf.ln(10)

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
        
        # Geração dos Links Nativos no PDF
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
        pdf.set_font('Times', 'B', 12)
        pdf.cell(0, line_height, "-" * 50, ln=1, align='C')
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')


# --- 3. INTERFACE ---
st.title("Classificador Semântico e Preditor de Vernacularidade para o Catálogo AHU da Macrorregião Sul do Brasil ")

st.markdown("""
### Sobre esta ferramenta
**1. Motor de Busca e Triagem:** Este sistema não contém as imagens digitalizadas dos manuscritos originais. Ele funciona como um classificador para os resumos do catálogo do **Arquivo Histórico Ultramarino (AHU)**. O objetivo é permitir que pesquisadores cruzem recortes geográficos, temas históricos e variáveis sociolinguísticas para obter as **cotas arquivísticas** (ex: *AHU_ACL_CU_...*) antes de acessar o arquivo físico ou o Projeto Resgate.

**2. O Score de Probabilidade de Vernacularidade (SPV):** Cada documento teve sua descrição processada pelo DeepSeek (v3) para a atribuição de um valor numérico indicativo da probabilidade de o documento conter indícios de vernacularidade. Esse valor varia entre **0 e 1**.
* Um **SPV próximo a 0** indica que o LLM que avaliou a descrição indicou baixa probabilidade (fórmulas diplomáticas rígidas, linguagem erudita metropolitana ou forte padronização de notários).
* Um **SPV próximo a 1** indica que o LLM que avaliou a descrição indicou alta probabilidade de que o manuscrito original contenha marcas de oralidade, inovações sintáticas e vazamento do português vernáculo brasileiro colonial.

**3. O Corte de Relevância (Rigor da Busca):** Este parâmetro define o limite matemático exigido para que o motor considere um documento pertinente à sua consulta. Ele cruza o sentido do texto com a correspondência exata das palavras.
* **Relevância próxima a 0** amplia o escopo da pesquisa e relaxa o filtro para incluir documentos com uma relação conceitual mais distante, periférica ou apenas tangencial ao termo inserido.
* **Relevância próxima a 1** exige uma correspondência extremamente estrita com o tema pesquisado.

**4. A Geração de Dossiê Documental (Exportar PDF):** Concluída a aplicação dos filtros de busca, a ferramenta permite compilar os resultados num dossiê exportável. Este documento apresenta os metadados, resumos dos manuscritos selecionados, assim como, em seu cabeçalho, todos os filtros que originaram aquele recorte.

**5. Acesso Direto aos Acervos:** A ferramenta gera automaticamente links para as plataformas oficiais. Ao expandir um resultado na tela, você pode usar o Código de Refência para abrir a ficha de controle arquivístico no DigitArq, ou usar o Código Atual para buscar as imagens microfilmadas no portal do Projeto Resgate (Biblioteca Nacional). Essa funcionalidade de redirecionamento também é preservada nos Dossiês em PDF exportados""")
st.divider()

st.subheader(":material/search: Busca Semântica/Lexical")
st.markdown("*Digite um conceito, tema ou evento histórico. O motor buscará documentos pelo significado contextual e pela correspondência literal.*")

query = st.text_input("Ex: 'conflitos de terra', 'deserção de soldados', 'escassez de farinha', 'pesca de baleias':")

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

with st.sidebar:
    st.header(":material/tune: Perfis de Busca Predefinidos")
    lente = st.radio(
        "Selecione uma lente metodológica:",
        ["Busca Livre (Personalizada)", 
         "Vozes Marginalizadas & História Social", 
         "Sintaxe Diacrônica (Alto SV)", 
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
    elif lente == "Sintaxe Diacrônica (Alto SV)":
        min_score = 0.7
    elif lente == "Máquina Administrativa (Top-Down)":
        vetor_padrao = ["Top-Down", "Horizontal"]
        remetente_padrao = ["Metropolitan Elite", "Local Elite"]
        
    score_range = st.slider("Score de Probabilidade de Vernacularidade (SPV):", 0.0, 1.0, (min_score, max_score), step=0.1)
    vetores = st.multiselect("Direção da Comunicação:", ["Bottom-Up", "Horizontal", "Top-Down", "Unknown"], default=vetor_padrao)
    categorias = st.multiselect("Perfil Social do Remetente:", categorias_disponiveis, default=remetente_padrao)

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

# --- 4. MOTOR ENSEMBLE RAG ---
if query:
    query_lower = query.lower().strip()
    tokenized_query = query_lower.split()
    
    lexical_scores_raw = bm25_engine.get_scores(tokenized_query)
    max_lex = np.max(lexical_scores_raw) if np.max(lexical_scores_raw) > 0 else 1
    lexical_normalized = lexical_scores_raw / max_lex
    
    e5_query = f"query: {query}"
    query_embedding = model.encode(e5_query, convert_to_tensor=True)
    semantic_scores_raw = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    
    max_sem = np.max(semantic_scores_raw)
    min_sem = np.min(semantic_scores_raw)
    range_sem = (max_sem - min_sem) if (max_sem - min_sem) > 0 else 1 
    semantic_normalized = (semantic_scores_raw - min_sem) / range_sem
    
    if len(tokenized_query) <= 2:
        lexical_weight = 0.75
        semantic_weight = 0.25
    else:
        lexical_weight = 0.35
        semantic_weight = 0.65
        
    final_hybrid_scores = (lexical_normalized * lexical_weight) + (semantic_normalized * semantic_weight)
    df_filter['semantic_score'] = final_hybrid_scores
    
    mask = mask & (df_filter['semantic_score'] >= limiar_semantico)
    results_df = df_filter[mask].sort_values(by='semantic_score', ascending=False)
else:
    results_df = df_filter[mask].sort_values(by='vernacular_score', ascending=False)


# --- 5. RESULTADOS E PDF ---
st.subheader(":material/picture_as_pdf: Exportar PDF com o Dossiê Documental")
st.markdown("*Use os filtros e a busca para isolar um conjunto de documentos. Em seguida, escolha a quantidade e clique abaixo para baixar um PDF formatado (Normas ABNT).*")

if not results_df.empty:
    col_exp_segura, col_exp_vazia = st.columns([2, 8])
    with col_exp_segura:
        limite_str = st.text_input("Quantidade a exportar:", value="50", max_chars=4)
        
    try:
        limite_exportacao = int(limite_str.strip())
        if limite_exportacao <= 0: limite_exportacao = 50
    except ValueError:
        limite_exportacao = 50
        st.error("Digite apenas inteiros. Retornando ao padrão (50).")
    
    export_df = results_df.head(limite_exportacao)
    
    regioes_str = ", ".join(regioes_selecionadas) if regioes_selecionadas else "Nenhuma"
    if len(regioes_selecionadas) == len(todas_regioes): regioes_str = "Todas"
    
    current_params = {
        "query": query if query else "Nenhuma restrição semântica",
        "lente": lente,
        "regioes": regioes_str,
        "sv_range": f"{score_range[0]:.1f} a {score_range[1]:.1f}",
        "vetores": ", ".join(vetores) if vetores else "Nenhum",
        "categorias": ", ".join(categorias) if categorias else "Nenhuma",
        "limiar": f"{limiar_semantico:.2f}"
    }
    
    pdf_bytes = create_pdf(export_df, current_params)
    st.download_button(label=f"Baixar Dossiê (Top {len(export_df)} documentos)", data=pdf_bytes, file_name="Dossie_AHU.pdf", mime="application/pdf")
else:
    st.warning("Nenhum documento encontrado com os filtros atuais, experimente diminuir o valor de corte da relevância ou aumentar o intervalo de SPV.")

st.divider()

st.subheader(f"Resultados Encontrados: {len(results_df)} documentos")

if not results_df.empty:
    for idx, row in results_df.head(50).iterrows():
        score = row.get('vernacular_score', 0.0)
        date_id = row.get('document_id_and_date', 'Sem Data')
        folder = row.get('folder', 'Local Desconhecido')
        
        if query:
            sem_score = row.get('semantic_score', 0.0)
            expander_title = f"Relevância: {sem_score:.2f} | SPV: {score:.1f} | {date_id} | {folder}"
        else:
            expander_title = f"SPV: {score:.1f} | {date_id} | {folder}"
        
        with st.expander(expander_title):
            crav_code = row.get('reference_code', 'Sem Cota CRAV')
            new_code = row.get('new_code', 'N/A')
            resgate_url = "https://resgate.bn.gov.br/"
            
            # Formatação do Digitarq
            if crav_code.startswith("PT/AHU"):
                encoded_crav = urllib.parse.quote(crav_code, safe='')
                crav_url = f"https://digitarq.arquivos.pt/search?query={encoded_crav}&isAdvancedSearch=false"
                st.markdown(f"**Código de Referência:** {crav_code} ([Busca no Digitarq]({crav_url}))")
            else:
                st.markdown(f"**Código de Referência:** {crav_code}")
            
            # Formatação do Projeto Resgate
            st.markdown(f"**Código Atual:** {new_code} ([Busca no Projeto Resgate]({resgate_url}))")
            
            st.markdown(f"**Tipologia:** {row.get('extracted_typology', 'N/A')}")
            st.markdown(f"**Remetente:** {row.get('sender_name', 'N/A')} *(Classe: {row.get('sender_category', 'N/A')})*")
            st.markdown(f"**Hierarquia:** {row.get('vector', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**Resumo do Arquivo (de autoria do AHU):**\n{row.get('description', '')}")
            st.markdown("---")
            reasoning = row.get('sociolinguistic_reasoning_by_deepseek_v3', '')
            st.markdown(f"**Justificativa do DeepSeek para o SPV:**\n*{reasoning}*")
            
    if len(results_df) > 50:
        st.info(f"Mostrando os 50 resultados mais relevantes no navegador de um total de {len(results_df)}. Ajuste o seletor acima para incluir mais no PDF.")

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
