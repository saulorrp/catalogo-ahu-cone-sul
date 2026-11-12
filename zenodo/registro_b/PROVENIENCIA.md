# Proveniência dos Dados e Camadas Analíticas

Este documento detalha a proveniência dos dados, a genealogia de cada camada analítica e os modelos de inteligência artificial empregados no pipeline de processamento.

As contribuições de modelos de inteligência artificial na extração, classificação e geração de artefatos estão descritas nesta seção e nas camadas correspondentes, não configurando autoria dos dados.

## Camada: tipologia_v1

### Metadados de camada.json

| Chave | Valor |
|---|---|
| `nome` | tipologia_v1 |
| `tipo` | tipologia |
| `versao` | 1 |
| `colecoes` | ["ahu-brasil", "ahu-sul"] |
| `origem_do_campo` | extracted_typology e typology_status do catálogo (preprint §1, Bellotto) |
| `regra_normalizacao` | maiúsculas, NFC, espaços colapsados, pontuação e aspas das pontas removidas; depois a tabela de correções de corpus/correcoes_tipologia.json, casada pela forma normalizada com acentos (para que a tabela possa corrigir acento trocado). Valor vazio -> tipologia nula e typology_status falso. |
| `regra_status` | typology_status do catálogo, ou a tipologia corrigida já ser canônica em outro documento. O vocabulário canônico sai dos dados. |
| `correcoes` | {"arquivo": "correcoes_tipologia.json", "tabela": {"COSIJLTA": "CONSULTA", "OFICÍO": "OFÍCIO", "RELÁTORIO": "RELATÓRIO", "UNKNOWN": ""}, "aplicadas": 6} |
| `tipologias_distintas` | 671 |
| `canonicas` | 151669 |
| `a_revisar` | 15675 |
| `contagem_por_tipologia` | {"ABAIXO ASSINADO": 5, "ABAIXO-ASSINADO": 4, "ABSTRATO": 1, "ACTA": 4, "ACÓRDÃO": 9, "ACÓRDÃOS": 2, "ADVERTÊNCIA": 1, "ADVERTÊNCIAS": 1, "ALEGAÇÃO": 2, "ALVARA": 1, "ALVARA DE LEI": 1, "ALVARÁ": 477, "ALVARÁ COM FORÇA DE LEI": 1, "ALVARÁ DE SESMARIA": 1, "ALVARÁS": 59, "ALVÁRA": 1, "ANOTAÇÃO": 1, "A |
| `processo` | {"script": "build/importar_camadas.py", "versao_script": "1.0.0", "commit": "fe76256"} |
| `entrada` | {"arquivo": "documentos.parquet", "sha256": "5cd0c7b5a5256add39fe2651848d2569077f5033ef66d299d04cafffae6d5163"} |
| `documentos` | 167344 |
| `esquema` | {"doc_uid": "object", "tipologia_bruta": "object", "tipologia": "object", "tipologia_chave": "object", "typology_status": "bool", "tipologia_corrigida": "bool"} |
| `data` | 2026-10-22T23:09:26+00:00 |

### Arquivos da camada e integridade

| Arquivo | SHA-256 |
|---|---|
| `tipologia.parquet` | `e2dde1a6c3acad454b346317e2fb1b73e3d61495e408c1be28ab6309e8d17f81` |
| `camada.json` | `ca1d4d0b1433c306d0ea96135e69fbc88675069422364d3116e8ced83f844dbc` |

## Camada: sociolinguistica_deepseek_v3

### Metadados de camada.json

| Chave | Valor |
|---|---|
| `nome` | sociolinguistica_deepseek_v3 |
| `tipo` | sociolinguistica |
| `versao` | 3 |
| `colecoes` | ["ahu-sul"] |
| `modelo` | deepseek-v3 |
| `prompt` | You are an expert in Historical Sociolinguistics, Paleography, and the Diachronic Syntax of Colonial Brazilian Portuguese. You are analyzing archival descriptions from the Southern Brazil region (Santa Catarina, Rio Grande do Sul, Uruguay, cities and regions like Ilha de Santa Catarina, Laguna, Rio Grande, rivers like 'Araranguá', 'Tramandaí', 'Tubarão', etc.) between 1737 and 1828.
Your task is to extract historical metadata and assign a 'vernacular_score' (1 to 10). This score represents the probability that the ORIGINAL physical document contains colloquial syntax, linguistic innovation (like Brazilian clitic placement), or non-standard grammar, avoiding rigid metropolitan formulaicism.
EVALUATE THE FOLLOWING DIMENSIONS TO CALCULATE THE SCORE:
1. Typology & Degree of Monitoring:
- HIGHEST (8-10): Devassas, Testemunhos, Interrogatórios (Transcripts of semi-spontaneous speech with low grammatical monitoring).
- MODERATE (4-7): Requerimentos, Petições, Cartas Particulares (Highly formulaic openings/closings, but the central narrative often leaks vernacular syntax).
- LOWEST (1-3): Provisões, Decretos, Alvarás, Consultas, ofícios militares de alta patente (Rigid, heavily monitored Coimbra-standard syntax).

2. Social Rank, Gender & The "Scribe Filter" (scriptor/Amanuense):
- Marginalized actors (índios, escravizados, pardos) and commoners could not write. Their petitions were dictated to local scribes. This "Scribe Filter" results in a HIGH score (7-9) due to the local scribe's hypercorrection or failure to master metropolitan syntax.
- Women generally lacked formal grammatical schooling. Intersect gender with class: A poor widow scores VERY HIGH (8-10); an elite woman scores MODERATE (5-6).
- Local Elites (Capitães-mores, vereadores) score MODERATE (often Brazilian-born, less formal education than European peers).
- Metropolitan Elites (Ouvidores, Governadores, Bispos) score LOWEST (1-2) due to strict adherence to classical grammar.

3. Communication Vector & Geography:
- Bottom-Up (Periphery to Metropole): HIGH probability of vernacular features.
- Deep Periphery (Frontier camps, sertão, aldeias) scores higher than administrative capitals (Desterro, Rio Grande city).
- Top-Down (Metropole to Periphery): LOW probability.

You MUST respond strictly with a raw JSON object exactly in this format:
{
"sender_name": "Name or Title",
"sender_category": "Metropolitan Elite / Local Elite / Low Military / Commoner / Marginalized / Clergy / Unknown","recipient_name": "Name or Title",
"vector": "Bottom-Up / Top-Down / Horizontal / Unknown",
"typology_impact": "Highly Formulaic / Narrative / Speech Transcript",
"scribe_mediation_likely": true or false,
"vernacular_score": 8,
"sociolinguistic_reasoning_by_deepseek_v3": "A highly specific 3-sentence linguistic explanation evaluating the intersection of typology, the scribe filter, and the communication vector."
} |
| `campos_de_origem` | {"sender_category": "sender_category", "vector": "vector", "typology_impact": "typology_impact", "vernacular_score": "vernacular_score", "scribe_mediation_likely": "scribe_mediation_likely", "sociolinguistic_reasoning_by_deepseek_v3": "justificativa"} |
| `escala_srsp` | [1, 9] |
| `escala_srsp_nota` | inteira de 1 a 9, como no dado; nunca dividida por 10. É esta escala que a importação exige: outra classificação é outra camada, com a sua própria escala. |
| `auditoria` | {"escala": "inteira 1-9 (SRSP)", "secao_preprint": "3.1", "mae": 1.13, "linha_base_constante_2_mae": 0.97, "r": 0.48, "spearman": 0.37, "qwk": 0.44, "vies": 0.59, "n": 716, "avaliadores": 1, "estatuto": "exploratória", "nota": "Prever sempre \"2\" daria MAE 0,97, menor que o do modelo: o SRSP não ac |
| `uso_restrito` | {"scribe_mediation_likely": "só o dossiê em PDF (D1); nunca interface, filtros, CSV ou JSON"} |
| `contagem_por_srsp` | {"1": 624, "2": 4842, "3": 359, "4": 94, "5": 138, "6": 853, "7": 126, "8": 6, "9": 9} |
| `contagem_por_categoria` | {"Clergy": 43, "Commoner": 371, "Local Elite": 831, "Local Elite / Low Military": 1, "Local Elite / Marginalized": 1, "Low Military": 257, "Low Military / Commoner": 2, "Marginalized": 7, "Metropolitan Elite": 5331, "Metropolitan Elite / Commoner": 1, "Metropolitan Elite / Local Elite": 3, "Mixed (L |
| `contagem_por_vetor` | {"Bottom-Up": 2830, "Horizontal": 178, "Mixed (Bottom-Up, Top-Down, Horizontal)": 1, "Top-Down": 3865, "Top-Down / Bottom-Up": 1, "Top-Down / Horizontal": 1, "Unknown": 175} |
| `mediacao_provavel` | 1080 |
| `processo` | {"script": "build/importar_camadas.py", "versao_script": "1.0.0", "commit": "1ee2ba6"} |
| `entrada` | {"arquivo": "ahu_sul_catalog_biografias.json", "sha256": "996feb761829f84863db528c44a9a8d85d33f1c9afd95b1d92ebed37576bf40d"} |
| `documentos` | 7051 |
| `esquema` | {"doc_uid": "object", "sender_category": "object", "vector": "object", "typology_impact": "object", "vernacular_score": "Int64", "scribe_mediation_likely": "bool", "justificativa": "object"} |
| `data` | 2026-10-08T12:05:35+00:00 |

### Arquivos da camada e integridade

| Arquivo | SHA-256 |
|---|---|
| `sociolinguistica.parquet` | `836164036eac56a8fae83d23918785796936360917ddce6e92b520ec48b64643` |
| `camada.json` | `486ba49cc9a85bc3256b0cebc3bd9e1fd937a49d64b9e6d8311d785588e60456` |

## Camada: entidades_v2

### Metadados de camada.json

| Chave | Valor |
|---|---|
| `nome` | entidades_v2 |
| `tipo` | entidades |
| `versao` | 2 |
| `colecoes` | ["ahu-sul"] |
| `origem` | pipeline de biografias v2, etapas p1 (extrair entidades) e p2 (consolidar) |
| `regra_pessoa_uid` | assinatura = chave_nome(nome_principal) | chaves dos títulos em ordem | ano_min-ano_max; pessoa_uid = 'pes:' + sha1(assinatura)[:8] |
| `assinatura_versao` | 1 |
| `documentos` | 7050 |
| `entidades` | 7229 |
| `pessoas` | 1359 |
| `contagem_por_tipo` | {"coletivo": 170, "instituicao": 957, "nao_identificado": 233, "pessoa": 5834, "pessoa_nome_incompleto": 19, "titulo": 16} |
| `entidades_com_pessoa_uid` | 5850 |
| `processo` | {"script": "build/importar_biografias_v2.py", "versao_script": "1.0.0", "commit": "58d6d34"} |
| `entradas` | {"ahu_sul_catalog_biografias.json": "996feb761829f84863db528c44a9a8d85d33f1c9afd95b1d92ebed37576bf40d", "entidades_por_remetente.json": "510f097e3f921aa19aec2a9bfb9a0b735890be0ccf83c9c2245f1c3000933316", "doc_pessoas.json": "3140c1c9f01dd5527ac211414b977ddbdf9b4eed5036a30720fd205de4f69583", "pessoas |
| `esquema` | {"entidades.parquet": {"doc_uid": "object", "posicao": "int16", "nome": "object", "tipo": "object", "pessoa_uid": "object", "titulos": "object", "cargos": "object", "autoria_verificada": "bool"}, "pessoas.parquet": {"pessoa_uid": "object", "tipo": "object", "nome_principal": "object", "nome_exibicao |
| `data` | 2026-10-08T11:44:11+00:00 |

### Arquivos da camada e integridade

| Arquivo | SHA-256 |
|---|---|
| `entidades.parquet` | `2e6dc27cf45d9a93e72676ef86582bde1bad3c67ffa8bcde89988518e8d003dd` |
| `pessoas.parquet` | `2d1dcd965c8fad9f6ba10ee762bb439afcc9e438bb243db4e93d9a31ee067ec9` |
| `registro_pessoas.json` | `8d73659d3a0bc79b9c6253a8e26cbd9fee849b4ac6e0c22a4494c6d5ae26b64e` |
| `camada.json` | `17cbf905242890ec44942b5037ceace2e79c745859d44ee162272b856523d70d` |

## Camada: biografias_v2

### Metadados de camada.json

| Chave | Valor |
|---|---|
| `nome` | biografias_v2 |
| `tipo` | biografias |
| `versao` | 2 |
| `colecoes` | ["ahu-sul"] |
| `origem` | pipeline de biografias v2, etapas p3 (buscar), p4 (validar, --wikidata) e p5 |
| `documentos` | 7050 |
| `pessoas` | 1359 |
| `exibiveis` | 253 |
| `regra_exibicao` | D2: nível ALTA ou MÉDIA, ou decisao_manual = aceitar; decisao_manual = rejeitar nunca é exibida |
| `rotulos` | {"ALTA": "Nascimento identificado (confiança alta)", "MEDIA": "Nascimento identificado (confiança média)", "BAIXA": "Nascimento não confirmado", "REJEITADO": "Não identificado", "IDENTIFICADO_SEM_LOCAL": "Identificado, local não encontrado", "NAO_ENCONTRADO": "Não identificado"} |
| `contagem_por_nivel` | {"ALTA": 192, "BAIXA": 36, "IDENTIFICADO_SEM_LOCAL": 274, "MEDIA": 61, "NAO_ENCONTRADO": 788, "REJEITADO": 8} |
| `contagem_por_decisao` | {} |
| `com_wikidata` | 139 |
| `processo` | {"script": "build/importar_biografias_v2.py", "versao_script": "1.0.0", "commit": "58d6d34"} |
| `entradas` | {"ahu_sul_catalog_biografias.json": "996feb761829f84863db528c44a9a8d85d33f1c9afd95b1d92ebed37576bf40d", "entidades_por_remetente.json": "510f097e3f921aa19aec2a9bfb9a0b735890be0ccf83c9c2245f1c3000933316", "doc_pessoas.json": "3140c1c9f01dd5527ac211414b977ddbdf9b4eed5036a30720fd205de4f69583", "pessoas |
| `esquema` | {"biografias.parquet": {"pessoa_uid": "object", "nivel": "object", "decisao_manual": "object", "exibivel": "bool", "rotulo": "object", "nome_na_fonte": "object", "ano_nascimento": "Int64", "ano_morte": "Int64", "cidade_nascimento": "object", "regiao_nascimento": "object", "pais_nascimento_epoca": "o |
| `data` | 2026-10-08T11:44:11+00:00 |

### Arquivos da camada e integridade

| Arquivo | SHA-256 |
|---|---|
| `biografias.parquet` | `86d23564f5aedfa416a1e212a9c2254c3241faaa2dba33a108c939feded3db02` |
| `camada.json` | `575fb7dc40adbd08d4f324df2c3cc437f7dea04e8bd9aed3f0feddb97761c581` |

## Metadados que divergem do manifesto do corpus

Os dados (parquet) batem byte a byte com `corpus/manifest.json`. Estes metadados não:

| Arquivo | Motivo |
|---|---|
| `sociolinguistica_deepseek_v3/camada.json` | metadado editado depois do manifesto |
| `entidades_v2/registro_pessoas.json` | fim de linha CRLF no manifesto |
| `entidades_v2/camada.json` | fim de linha CRLF no manifesto |
| `biografias_v2/camada.json` | fim de linha CRLF no manifesto |

## Modelos

| Repositório | Revisão | Arquivo | SHA-256 |
|---|---|---|---|
| `Xenova/multilingual-e5-small` | `761b726dd34fb83930e26aab4e9ac3899aa1fa78` | `onnx/model.onnx` | `4aa845c27760e06e9a686b9d8b5d440eae4b6612cd09e5b522b716d3941f77ff` |
| `Xenova/multilingual-e5-small` | `761b726dd34fb83930e26aab4e9ac3899aa1fa78` | `onnx/model_quantized.onnx` | `f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193` |
| `Xenova/multilingual-e5-small` | `761b726dd34fb83930e26aab4e9ac3899aa1fa78` | `tokenizer.json` | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` |
| `Xenova/multilingual-e5-base` | `1ec9243030a27d1a115d5c340572074c125b58b2` | `onnx/model.onnx` | `84a4d426f7e87a6bf5bf195f0bae2c4a7d15f675b23ca96f42fab8326d7a77aa` |
| `Xenova/multilingual-e5-base` | `1ec9243030a27d1a115d5c340572074c125b58b2` | `onnx/model_quantized.onnx` | `df7a9a29309e3ad491e1783adf8baee710262cc06079c7cbab63c630277fac94` |
| `Xenova/multilingual-e5-base` | `1ec9243030a27d1a115d5c340572074c125b58b2` | `tokenizer.json` | `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
| `Xenova/multilingual-e5-large` | `00fc3aeb3dbb95842de2ac1961d33c6319acf57b` | `onnx/model.onnx` | `8e17fbd06c54c2a53b467f5b127230f798d4fb69c5344ef9a9d6fa463862c453` |
| `Xenova/multilingual-e5-large` | `00fc3aeb3dbb95842de2ac1961d33c6319acf57b` | `onnx/model.onnx_data` | `0cf1883fee81c63819a44e2ba0efa51d4043d9759685a4ebebbde97e0623d15c` |
| `Xenova/multilingual-e5-large` | `00fc3aeb3dbb95842de2ac1961d33c6319acf57b` | `onnx/model_quantized.onnx` | `0a8d65db9a36f810ba5da15249f13145fcdc7890e6656f1fd38cd8b7c4db1fca` |
| `Xenova/multilingual-e5-large` | `00fc3aeb3dbb95842de2ac1961d33c6319acf57b` | `tokenizer.json` | `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | `onnx/model.onnx` | `3e9a03ed1e966f7c5288dd4230e3d6a9bf5e3a170a06f1f4241c5bca12c6487c` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | `onnx/model_quint8_avx2.onnx` | `6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b` |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | `tokenizer.json` | `62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626` |
