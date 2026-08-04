# Classificador de Documentos do Catálogo do AHU para a Macrorregião Sul do Brasil

## Visão Geral (Overview)
Este conjunto de dados (*dataset*) e o aplicativo web que o acompanha fornecem um corpus legível por máquina e semanticamente anotado de resumos de catálogos do Arquivo Histórico Ultramarino (AHU). Focado na "Macrorregião Sul do Brasil" durante o final do período colonial (1737–1828), estes arquivos são o produto de uma *pipeline* que utilizou Grandes Modelos de Linguagem (LLMs), Motores de Busca Lexical, *embeddings* de vetores densos e **Agentes de Pesquisa Autônomos** para quantificar variáveis sociolinguísticas e mapear origens diatópicas. 

O projeto foi desenhado para apoiar pesquisas em Linguística Diacrônica, especificamente com o objetivo de mapear a competição de gramáticas e isolar a probabilidade de sintaxe do português brasileiro em documentação administrativa histórica. Além disso, atua como um instrumento de pesquisa avançado, gerando hiperlinks diretos e clicáveis para os registros dos manuscritos nas bases de dados do DigitArq (Portugal) e do Projeto Resgate (Brasil), e orquestrando a criação de amostras estruturadas para o trabalho de campo em arquivos físicos.

---

## Arquivos Incluídos neste Dataset
O dataset e o aplicativo são distribuídos como um repositório contendo os seguintes arquivos e pastas principais:

* `ahu_sul_catalog_final.json`: O recurso estruturado principal (formato JSON). Ele conecta o texto arquivístico bruto a metadados geográficos, dados temporais limpos, códigos de referência arquivística modernizados e anotações sociolinguísticas algorítmicas e biográficas.
  * **Chaves Principais (Core Keys):**
    * `reference_code` e `new_code`: Códigos de referência modernizados no padrão CRAV/DigitArq e Projeto Resgate.
    * `description`: Descrição em texto integral extraída do catálogo do AHU.
    * `ano_isolado`: Ano exato do documento (formato numérico) higienizado via RegEx estrutural, garantindo ordenação cronológica estrita livre de ruídos de códigos de tombo.
    * `autoria_verificada`: Booleano de validação cruzada confirmando a correspondência exata entre o nome do remetente e a estrutura diplomática do resumo.
    * `perplexity_search_data`: Objeto aninhado contendo metadados biográficos e diatópicos do autor (extraídos via IA agêntica baseada em fontes primárias/enciclopédicas), incluindo `cidade_nascimento`, `pais_nascimento`, grau de `confianca` e `link_fonte`.
    * `vernacular_score`: Um número decimal (0.0 a 1.0) que representa quantitativamente o Índice de Relevância Sociolinguística Potencial (IRSP).
    * `vector` e `extracted_typology`: Direcionalidade do documento e raiz singular normalizada do tipo documental.
    * `sociolinguistic_reasoning_by_deepseek_v3`: A justificativa analítica gerada pela IA para o IRSP atribuído.

* `ahu_semantic_index.pkl`: Um dicionário Python serializado contendo os embeddings de vetores densos das descrições arquivísticas (modelo `intfloat/multilingual-e5-large`), ancorados aos códigos de referência para recuperação semântica.

* `app.py`: O script do Aplicativo Streamlit (Interface Gráfica). Além de gerenciar a Busca Híbrida Dinâmica (BM25 + Dense Vectors) e a filtragem sociolinguística, esta versão inclui:
  * **Dashboard de Análise Diatópica e Autoral:** Ambiente visual analítico para o cruzamento do perfil geográfico do escrevente com a linha do tempo do corpus.
  * **Ferramenta de Criação de Amostras Diacrônicas:** Módulo projetado para o trabalho de arquivo físico. Filtra autores validados, ordena documentos estritamente pela diacronia e agrupa-os em lotes de paginação (máx. 10 documentos), exportando um plano de coleta estruturado em JSON para integração com softwares de aquisição de imagens (ex: Projeto EINS).
  * **Filtros Globais Interconectados:** Segmentação profunda cruzando categorias sociais do remetente com seus locais de nascimento mapeados.

* `auditoria/`: Pasta contendo scripts e dados de higienização do corpus, documentação metodológica, auditoria empírica de viés da IA, e heurísticas de desambiguação de entidades nomeadas para filtragem de anacronismos históricos.

* `requirements.txt`: Lista de dependências do ambiente Python (ex: pandas, streamlit, sentence-transformers, rank-bm25, torch, fpdf, openai).

---

## Escopo Incluído
* **Recorte Temporal:** 1737 a 1828.
* **Foco Geográfico:** Macrorregião Sul do Brasil (Contexto de Escrita) e Império Português (Origens Diatópicas).
* **Fonte dos Dados:** Resumos e metadados extraídos dos catálogos do Arquivo Histórico Ultramarino (AHU).

---

## Metodologia Computacional
* **Limpeza e Engenharia Reversa:** Extração precisa e isolamento do eixo diacrônico (`ano_isolado`) a partir de strings arquivísticas mistas via RegEx, e modernização de códigos de referência para o padrão CRAV.
* **Inferência Sociolinguística (Zero-shot LLM):** Categorização automatizada da hierarquia comunicacional e cálculo do IRSP (probabilidade de sintaxe vernácula/localizada) com o modelo DeepSeek v3.
* **Enriquecimento Biográfico Agêntico:** Integração com a API do Perplexity (modelo `sonar-pro`) para pesquisa web autônoma. O agente extraiu o local de nascimento dos remetentes priorizando a Wikipédia Lusófona e dicionários histórico-biográficos, consolidando os dados geográficos e suas fontes de comprovação.
* **Auditoria de Entidades:** Aplicação de heurísticas pós-processamento para identificar e isolar em quarentena alucinações de desambiguação de entidades nomeadas (anacronismos).
* **Busca Híbrida (Search Fusion):** Fusão dinâmica de indexação léxica (`BM25Okapi`) para precisão de termos e vetorização semântica densa (`multilingual-e5-large`) para alcance conceitual.

---

## Fonte dos Dados Arquivísticos
Os resumos textuais não estruturados originais usados para construir este corpus foram obtidos a partir dos esforços públicos de catalogação do Arquivo Histórico Ultramarino e do Projeto Resgate. Agradecemos imensamente o trabalho contínuo destas instituições na preservação e indexação do patrimônio documental colonial. Os códigos de referência modernizados gerados por este aplicativo fazem interface direta com a infraestrutura do Arquivo de Portugal (DigitArq).

## Critérios Metodológicos
Para uma descrição detalhada da *pipeline* computacional, parâmetros de prompting, auditoria empírica de viés, algoritmos de integração agêntica e arquitetura da interface focada em linguística de corpus, consulte o *data paper* a ser publicado que será associado a este repositório.

## Citação
Atualmente, este dataset e aplicativo estão publicados como um recurso computacional independente. Se você utilizar este corpus, metodologia ou código, por favor, cite este repositório diretamente usando o seu DOI do Zenodo:

> Pacheco Rocha, Saulo Rogério. (2026). AHU Catalog Document Classifier for the Southern Macro-Region of Brazil (Version 2.0) [Data set and Software]. Zenodo. https://doi.org/10.5281/zenodo.18772667

## Licença
Este dataset e o código associado são disponibilizados sob a licença **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

## Repositório
O código-fonte, as atualizações e a documentação completa podem ser encontrados no GitHub:
[https://github.com/saulorrp/catalogo-ahu-cone-sul.git](https://github.com/saulorrp/catalogo-ahu-cone-sul.git)
