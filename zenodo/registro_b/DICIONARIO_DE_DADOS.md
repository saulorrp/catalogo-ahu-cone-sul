# Dicionário de Dados — documentos_sem_resumo.parquet

Total de registros: **167.344**  
Total de colunas: **26**

| Coluna | Tipo Arrow | Origem | Licença | Vazios | Exemplo | Regra / Descrição |
|---|---|---|---|---|---|---|
| `doc_uid` | `string` | Gerado no build do corpus | CC0 1.0 (fatos não são protegidos) | 0 (0.0%) | `ahu-sul:e7fd41bfb3` | Identificador interno estável: ahu-sul vem de corpus/colecoes/ahu-sul/registro_ids.json; ahu-brasil é sha1 da chave da cota (com #n quando a cota se repete), em coleta/scripts/juntar.py. |
| `colecao` | `string` | Corpus | CC0 1.0 (fatos não são protegidos) | 0 (0.0%) | `ahu-sul` | Coleção de origem: ahu-sul ou ahu-brasil. |
| `folder` | `string` | Catálogo do AHU (ACTD) / Projeto Resgate | CC0 1.0 (fatos não são protegidos) | 0 (0.0%) | `Brasil Geral` | Série ou capitania (pasta) do verbete, como no catálogo. |
| `old_code` | `string` | Catálogo do AHU (ACTD) / Projeto Resgate | CC0 1.0 (fatos não são protegidos) | 75573 (45.16%) | `AHU-Maranhão.` | Cota antiga (arrumação anterior do AHU), como impressa. |
| `new_code` | `string` | Catálogo do AHU (ACTD) / Projeto Resgate | CC0 1.0 (fatos não são protegidos) | 5 (0.0%) | `AHU_ACL_CU_003, Cx. 1, D. 100.` | Cota atual no formato do Resgate (AHU_ACL_CU_<série>, Cx., D.), como impressa. |
| `reference_code` | `string` | ahu/conversor.py | CC0 1.0 (fatos não são protegidos) | 9769 (5.84%) | `PT/AHU/CU/003/0001/00100` | Código de referência PT/AHU/CU/<série>/<caixa>/<documento>, escrito pelo conversor a partir da cota. |
| `document_id_and_date` | `string` | Catálogo do AHU (ACTD) / Projeto Resgate | CC0 1.0 (fatos não são protegidos) | 0 (0.0%) | `2673. 1686, Fevereiro, 12, Lisboa` | Número do verbete, data e local, como no catálogo. |
| `ano_isolado` | `int64` | build/cotas.py | CC0 1.0 (fatos não são protegidos) | 2390 (1.43%) | `1686` | Ano extraído da data do verbete; vazio quando não há ano legível. |
| `recipient_name` | `string` | Camada do ahu-sul (pipeline de biografias) | CC0 1.0 (fatos não são protegidos) | 160293 (95.79%) | `D. Pedro II` | Destinatário; só existe na coleção ahu-sul. |
| `sender_name` | `string` | Camada do ahu-sul (pipeline de biografias) | CC0 1.0 (fatos não são protegidos) | 160293 (95.79%) | `Conselho Ultramarino` | Remetente; só existe na coleção ahu-sul. |
| `referencia_digitarq` | `string` | DigitArq do AHU (coleta/scripts/varrer_digitarq.py) | CC0 1.0 (fatos não são protegidos) | 9991 (5.97%) | `PT/AHU/CU/003/0001/00100` | Código de referência do registro do DigitArq que contém o documento. |
| `resgate_docreader` | `string` | Projeto Resgate, DocReader da BN | CC0 1.0 (fatos não são protegidos) | 151561 (90.57%) | `019_RS#1` | Identificador do documento no DocReader (imagens digitalizadas). |
| `duplicata_de` | `string` | corpus/duplicatas.ignore | CC0 1.0 (fatos não são protegidos) | 167137 (99.88%) | `ahu-sul:f0d170b4ca` | doc_uid do verbete de que este é repetição; a busca ignora o repetido. |
| `tipologia` | `string` | Camada tipologia_v1 | CC BY 4.0 | 3974 (2.37%) | `CONSULTA` | Primeira palavra em maiúsculas do resumo, normalizada e corrigida por corpus/correcoes_tipologia.json. |
| `tipologia_chave` | `string` | Camada tipologia_v1 | CC BY 4.0 | 3974 (2.37%) | `CONSULTA` | A tipologia sem acentos, para agrupar e filtrar. |
| `typology_status` | `bool` | Camada tipologia_v1 | CC BY 4.0 | 0 (0.0%) | `True` | Falso quando a tipologia ficou vazia depois da normalização. |
| `sender_category` | `string` | Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2) | CC BY 4.0 | 160293 (95.79%) | `Metropolitan Elite` | Categoria social do remetente; só ahu-sul. |
| `vector` | `string` | Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2) | CC BY 4.0 | 160293 (95.79%) | `Top-Down` | Direção da comunicação (ex.: Top-Down); só ahu-sul. |
| `typology_impact` | `string` | Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2) | CC BY 4.0 | 160293 (95.79%) | `Highly Formulaic` | Quanto a tipologia fixa a forma do texto (ex.: Highly Formulaic); só ahu-sul. |
| `vernacular_score` | `int64` | Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2) | CC BY 4.0 | 160293 (95.79%) | `1` | SRSP, inteiro de 1 a 9; exploratório (ver LIMITACOES.md); só ahu-sul. |
| `justificativa` | `string` | Camada sociolinguistica_deepseek_v3 (DeepSeek, preprint §2) | CC BY 4.0 | 160311 (95.8%) | `This is a 'Consulta' from the Overseas Council to the king, representing the hig` | Justificativa do modelo para a classificação, em inglês; só ahu-sul. |
| `url_actd` | `string` | ACTD/IICT | CC0 1.0 (fatos não são protegidos) | 10055 (6.01%) | `https://actd.iict.pt/eserv/actd:CUc004/CU-Alagoas.pdf` | URL do PDF do catálogo da série no ACTD de onde veio o resumo. |
| `pagina_actd` | `int32` | ACTD/IICT | CC0 1.0 (fatos não são protegidos) | 10055 (6.01%) | `3` | Página do PDF (contada pelo PyMuPDF, a partir de 1) onde está o verbete. |
| `caminho_resgate` | `string` | Projeto Resgate (gov.br) | CC0 1.0 (fatos não são protegidos) | 121295 (72.48%) | `https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/novos-instrumen` | Página do instrumento de pesquisa do Projeto Resgate que traz o verbete. |
| `qc` | `string` | coleta/scripts/qc_banco.py | CC0 1.0 (fatos não são protegidos) | 163170 (97.51%) | `cota_repetida` | Marcas de controle de qualidade, separadas por ponto e vírgula (ex.: sem_data, cota_repetida). |
| `sha1_resumo` | `string` | build/construir_app.py | CC0 1.0 (fatos não são protegidos) | 30 (0.02%) | `397d2146ee3b9e2a93e5eff2ed0db3583d603c93` | SHA-1 do resumo (UTF-8, como na coluna retirada); confere a reidratação a partir do ACTD. |
