# Notas da Versão — v3.0.0

Registro consolidado das transformações arquiteturais, metodológicas e de interface implementadas na versão **3.0.0** do Classificador AHU-Sul, com base na especificação técnica e na transição do preprint (SciELO Preprints, doi:10.1590/SciELOPreprints.16461):

---

## 1. Dados e Arquitetura Modular (F1)

- **Corpus em camadas desacopladas (`corpus/`):**
  - Base documental com 7.051 registros catalográficos do AHU (coleção Sul), dotados de identificadores únicos estáveis e reprodutíveis (`doc_uid`).
  - Camadas independentes para tipologia diplomática (`tipologia_v1`), sociolinguística computacional (`sociolinguistica_deepseek_v3`), entidades e biografias (`biografias_v2`), e índices densos de vetores.
  - Artefatos de execução otimizados gerados em `app_dados/` (formato Apache Parquet e NumPy `.f16`).
- **Pipeline de Biografias v2 e Filtragem Estrita (D2):**
  - Resolução de entidades enriquecida;
  - Exibição estritamente restrita a entidades com confiança **ALTA** ou **MÉDIA** (biografias rejeitadas ou de baixa confiança expurgadas da interface e filtros).

---

## 2. Motor de Busca e Recuperação de Informação (F2, R7)

- **Busca Híbrida em 3 Vias:**
  1. *Rota Exata de Cotas:* Reconhecimento automático dos formatos arquivísticos (CRAV `PT/AHU/CU/...`, Cota Atual `AHU_ACL_CU_...` e busca textual livre desprovida de pontuação), eliminando buscas por correspondência solta em números parciais.
  2. *Motor Léxico BM25 por Campos:* Ponderação diferenciada entre pessoas (2,0), tipologia (1,5), destinatários (1,5), descrição (1,0) e região/pasta (0,5).
  3. *Motor Semântico com ONNX int8:* Quantização int8 do modelo `intfloat/multilingual-e5-small`. Eliminação da importação do PyTorch em tempo de execução, reduzindo o consumo de pico de 948 MiB para **630 MiB**, permitindo estabilidade nos limites do Streamlit Community Cloud (690 MiB).
- **Fusão Recíproca por Posição (RRF):**
  - Substituição de normalizações lineares frágeis pela soma de posições relativas `1 / (60 + rank)`, com alternância de pesos adaptativa para consultas curtas vs. longas.
- **Rigor por Margem Relativa:**
  - Limiar de rigor baseado na distância em relação ao topo da própria consulta, contornando a distorção de cosseno absoluto típica de encoders multilíngues.

---

## 3. Interface e Experiência do Usuário (v3.1)

- **Descarte de IDs Internos na Interface (D10):**
  - O pesquisador opera exclusivamente com cotas, referências, datas, nomes e resumos históricos.
- **Ressalva Metodológica e Auditoria em Popover (D8, R5):**
  - Popover "Saiba mais" explicando a mediação notarial/escrivães do século XVIII e relatando os parâmetros de auditoria do DeepSeek v3 (MAE 1,13; correlação r = 0,48; viés +0,59; N = 705).
- **PDF da Lista de Consulta no AHU (D7, R3):**
  - Documento de trabalho em folha A4 paisagem, tipografia Noto Serif embutida, caixas de conferência física, cota CRAV e link direto para o catálogo DigitArq.
- **Amostragem e Integração EINS Lite (D4):**
  - Exportação de planos de coleta estruturados em JSON validando contra o esquema `eins-plano-coleta/1`.
- **Identidade Visual EINS (R6):**
  - Paleta com acento bordô, suporte a temas claro e escuro, e resumos documentais em tipografia serifada.

---

## 4. Avaliação, Reprodutibilidade e CI (F4)

- **Suíte de Avaliação de Recuperação (WP 0.3, WP 4.1):**
  - Conjunto de teste com 30 consultas calibradas (temas, pessoas, lugares e cotas arquivísticas).
  - Reprodução do motor de busca da versão v2 (`avaliacao/busca_v2.py`) para permitir avaliação cega pareada.
  - Métricas formais de recuperação de informação: nDCG@10, Recall@20 e MRR.
- **Integração Contínua (WP 4.3):**
  - Pipeline GitHub Actions automatizado rodando linters (`ruff`) e suíte de testes de regressão (`pytest`).
