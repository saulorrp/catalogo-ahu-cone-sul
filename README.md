# AHU Catalog Document Classifier for the Southern Macro-Region of Brazil

## Visão Geral (Overview)
Este conjunto de dados (*dataset*) e o aplicativo web que o acompanha fornecem um corpus legível por máquina e semanticamente anotado de resumos de catálogos do Arquivo Histórico Ultramarino (AHU). Focado na "Macrorregião Sul do Brasil" durante o final do período colonial (1737–1828), estes arquivos são o produto de uma *pipeline* que utilizou Grandes Modelos de Linguagem (LLMs), Motores de Busca Lexical e *embeddings* de vetores densos para quantificar variáveis sociolinguísticas. 

O projeto foi desenhado para apoiar pesquisas em Linguística Diacrônica, especificamente com o objetivo de mapear a competição de gramáticas e isolar a probabilidade de "Vazamento Vernáculo" (sintaxe do português oral/brasileiro) em documentação administrativa histórica. Além disso, atua como um instrumento de pesquisa avançado, gerando hiperlinks diretos e clicáveis para os registros dos manuscritos nas bases de dados do DigitArq (Portugal) e do Projeto Resgate (Brasil).

## Arquivos Incluídos neste Dataset
O dataset e o aplicativo são distribuídos como um repositório contendo os seguintes arquivos e pastas principais:

* `ahu_sul_catalog.json`: O recurso estruturado principal (formato JSON). Ele conecta o texto arquivístico bruto a metadados geográficos, dados temporais, códigos de referência arquivística modernizados e anotações sociolinguísticas algorítmicas.
  * **Chaves Principais (Core Keys):**
    * `reference_code`: Código de referência modernizado no padrão CRAV/DigitArq (ex: PT/AHU/CU/023-001/0006/00631).
    * `new_code`: Código padronizado/intermediário ainda utilizado pelo Projeto Resgate.
    * `description`: Descrição em texto integral extraída do catálogo do AHU.
    * `vernacular_score`: Um número decimal (0.0 a 1.0) que representa quantitativamente o Índice de Relevância Sociolinguística Potencial (IRSP), indicando a probabilidade de ocorrência de sintaxe vernácula/localizada no documento original.
    * `vector`: Direcionalidade do documento (ex: Top-Down, Bottom-Up).
    * `extracted_typology`: Raiz singular normalizada do tipo de documento (ex: CARTA, REQUERIMENTO).
    * `sociolinguistic_reasoning_by_deepseek_v3`: A justificativa analítica gerada pela IA para o Índice de Relevância Sociolinguística Potencial (IRSP) atribuído.
* `ahu_semantic_index.pkl`: Um dicionário Python serializado contendo os *embeddings* de vetores densos das descrições arquivísticas. Gerados através do modelo `intfloat/multilingual-e5-large`, esses *embeddings* são rigidamente ancorados aos códigos de referência dos documentos para garantir sincronização perfeita durante a recuperação semântica.
* `auditoria/`: Pasta contendo a documentação metodológica e os dados da validação empírica manual (Auditoria de Consistência Semântica) realizada sobre uma amostra aleatória de 10% do corpus (705 documentos). Inclui os dados utilizados para o cálculo do Erro Médio Absoluto, Correlação de Pearson e Viés Sistemático da IA.
* `app.py`: O script do Aplicativo Streamlit (Interface Gráfica). Ele gerencia a Busca Híbrida Dinâmica (*Ensemble Search*, fundindo a pontuação Lexical do BM25 com a Similaridade Semântica do E5), a filtragem sociolinguística ("Lentes"), o roteamento dinâmico de URLs para os arquivos digitais e a geração programática de dossiês em PDF formatados via FPDF.
* `requirements.txt`: Lista de dependências do ambiente Python necessárias para executar a *pipeline* e o aplicativo web (ex: pandas, streamlit, sentence-transformers, rank-bm25, torch, fpdf).

## Escopo Incluído
* **Recorte Temporal:** 1737 a 1828.
* **Foco Geográfico:** Macrorregião Sul do Brasil.
* **Fonte dos Dados:** Resumos e metadados extraídos dos catálogos do Arquivo Histórico Ultramarino (AHU).

## Fonte dos Dados Arquivísticos
Os resumos textuais não estruturados originais usados para construir este corpus foram obtidos a partir dos esforços públicos de catalogação do Arquivo Histórico Ultramarino e do Projeto Resgate. Agradecemos imensamente o trabalho contínuo destas instituições na preservação e indexação do patrimônio documental colonial. Os códigos de referência modernizados gerados por este aplicativo fazem interface direta com a infraestrutura do Arquivo de Portugal (DigitArq).

## Metodologia
* **Extração e Modernização:** Análise por Expressões Regulares (RegEx) do texto bruto do catálogo do AHU para isolar IDs de documentos, datas e tipologias, seguida de preenchimento algorítmico de zeros (*zero-padding*) para fazer engenharia reversa e gerar códigos de referência modernos compatíveis com o padrão CRAV.
* **Anotação:** Inferência *Zero-shot* por LLM (DeepSeek) para categorizar o remetente/destinatário, vetor de comunicação e calcular o Índice de Relevância Sociolinguística Potencial (IRSP) junto com uma justificativa analítica por escrito.
* **Vetorização (Semântica):** Passagens enriquecidas com contexto transformadas em *embeddings* usando o modelo robusto `intfloat/multilingual-e5-large` da HuggingFace.
* **Tokenização (Lexical):** Indexação por frequência de termo-inverso da frequência nos documentos via algoritmo `BM25Okapi` para garantir a recuperação precisa de nomes próprios e localidades.
* **Fusão de Busca (Search Fusion):** Recuperação de Conjunto Dinâmica (*Dynamic Ensemble Retrieval*) calculando uma pontuação híbrida ponderada entre significado semântico e exatidão lexical.

## Critérios Metodológicos
Para uma descrição detalhada da *pipeline* computacional, parâmetros de *prompting* do LLM, auditoria empírica de viés sistemático, lógica do "Modernizador de Referência CRAV", algoritmos da Busca Híbrida e regras de implantação de UI/UX, consulte o *data paper* a ser publicado que será associado a este repositório.

## Citação
Atualmente, este dataset e aplicativo estão publicados como um recurso computacional independente. Se você utilizar este corpus, metodologia ou código, por favor, cite este repositório diretamente usando o seu DOI do Zenodo:

> Rocha, Saulo Rogério Pacheco. (2026). AHU Catalog Document Classifier for the Southern Macro-Region of Brazil (Version 2.0) [Data set and Software]. Zenodo. https://doi.org/10.5281/zenodo.18772667

## Licença
Este dataset e o código associado são disponibilizados sob a licença **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

## Repositório Oficial
O código-fonte, as atualizações e a documentação completa podem ser encontrados no GitHub:
🔗 [https://github.com/saulorrp/catalogo-ahu-cone-sul.git](https://github.com/saulorrp/catalogo-ahu-cone-sul.git)
