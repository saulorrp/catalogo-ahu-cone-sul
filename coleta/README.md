# Coleta do catálogo AHU-Brasil

Os scripts que montaram a coleção `ahu-brasil`, publicados como rodaram, para que a coleta possa ser
refeita e conferida. O método está em [`../docs/METODO_COLETA.md`](../docs/METODO_COLETA.md), e as
fontes, com endereços e data de acesso, em [`FONTES.md`](FONTES.md).

Os scripts não fazem parte do app nem passam pelo `ruff` e pelos testes do repositório. Eles leem e
escrevem numa pasta de trabalho (`R`, o primeiro argumento), fora do repositório.

## Ordem

| Passo | Script | O que faz |
|---|---|---|
| 1 | `mapear_arvore.py`, `mapear_faltantes.py` | Rastreiam as páginas do Projeto Resgate no gov.br e gravam a árvore de endereços |
| 2 | `colher_html.py`, `colher_faltantes.py` | Baixam as páginas e PDFs e os guardam como vieram (`raw/`, `fontes/`) |
| 3 | `parse_verbetes.py`, `parse_texto.py`, `parse_sergipe.py`, `parse_codices.py` | Separam os verbetes de cada formato do Resgate (tabela HTML, texto de PDF, Sergipe, Códices) |
| 4 | `pdf_para_texto.py`, `parse_actd.py` | Extraem o texto dos catálogos do ACTD (lista em `series_actd.tsv`) e separam os verbetes |
| 5 | `juntar.py` | Junta as fontes por chave de cota, atribui `doc_uid` estável e liga à coleção ahu-sul (`juntar_v1.py` é a versão anterior, sem o ACTD) |
| 6 | `qc_banco.py` | Marca os verbetes suspeitos e escreve `qc/RELATORIO_QC.md` |
| 6b | `conferir_ocr.py` | Compara o OCR dos catálogos escaneados com o texto do ACTD, pela cota |
| 7 | `exportar_colecao.py` | Gera o parquet da coleção com as cotas calculadas por `ahu/conversor.py` |
| 8 | `../build/importar_colecao.py` | Importa a coleção no corpus do app |

## OCR dos catálogos escaneados

`baixar_docvirt.py` e `ocr/scripts/baixar_paginas.py` baixam as imagens das páginas (docvirt), e
`ocr/scripts/extrair_bn.py` as dos PDFs da BN. As instruções de transcrição dadas ao Gemini Flash
estão em `ocr/PROMPT_OCR.md`. Em `ocr/scripts/`:

| Script | O que faz |
|---|---|
| `ocr_gemini.py` | Transcrição pelo Gemini (gratuito, em fila, com pausa quando a cota acaba) |
| `ocr_batch.py` | Transcrição pela Batch API paga do Gemini, com teto de gasto e pausa por arquivo |
| `tesseract_estoque.py`, `qualidade_tesseract.py` | Tesseract 5.4 nas páginas e medida da sua qualidade |
| `corrigir_tesseract.py` | Correção do texto do Tesseract pelo Gemini, olhando a imagem (páginas recusadas) |
| `auditar_saidas.py`, `painel.py` | Conferência das saídas e painel de andamento |
| `teste_correcao_texto.py`, `teste_deepseek.py` | Testes de qualidade usados para escolher os modelos |

As chaves de API vêm de variáveis de ambiente; nenhuma está nos arquivos.

O texto transcrito serve de conferência do ACTD, não de fonte. `scripts/conferir_ocr.py` faz essa
conferência, e o resultado está em [`qc/RELATORIO_CONFERENCIA_OCR.md`](qc/RELATORIO_CONFERENCIA_OCR.md).
