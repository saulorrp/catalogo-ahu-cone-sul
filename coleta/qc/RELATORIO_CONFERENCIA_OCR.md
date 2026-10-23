# Conferência do OCR com o texto do ACTD

Cada verbete transcrito por OCR dos catálogos escaneados do Projeto Resgate foi comparado, pela cota, com o verbete do mesmo documento no banco (texto do ACTD). Similaridade de texto normalizado (sem acentos, caixa e pontuação); confere a partir de 0.9.

Verbetes do OCR com cota no banco: 74696. Conferem: 62314 (83.4%). Ano igual: 67621.

## Por volume

| Volume | Páginas | Verbetes | Sem cota | Sem par | Com par | Mediana | Conferem |
|---|---|---|---|---|---|---|---|
| 75_sao_paulo_v1 | 311 | 1388 | 2 | 4 | 1384 | 1.0 | 1368 |
| 76_sao_paulo_v2 | 772 | 4896 | 8 | 76 | 4820 | 1.0 | 4707 |
| 79_para_v1 | 438 | 5959 | 1 | 15 | 5944 | 0.958 | 5758 |
| 80_para_v2 | 490 | 6521 | 0 | 6 | 6515 | 0.964 | 6427 |
| 81_para_v3 | 262 | 0 | 0 | 0 | 0 | None | 0 |
| 84_codices_cu | 175 | 810 | 810 | 810 | 0 | None | 0 |
| 86_bahia_v1 | 708 | 11947 | 4 | 10 | 11937 | 1.0 | 11864 |
| 87_bahia_v2 | 470 | 7500 | 1 | 15 | 7485 | 1.0 | 7464 |
| 88_ultramar_brasil_geral | 195 | 2849 | 1 | 12 | 2837 | 0.998 | 2615 |
| 89_cartografia | 30 | 0 | 0 | 0 | 0 | None | 0 |
| 91_minas_v1 | 320 | 0 | 0 | 0 | 0 | None | 0 |
| 92_minas_v2 | 356 | 0 | 0 | 0 | 0 | None | 0 |
| 93_minas_v3 | 360 | 0 | 0 | 0 | 0 | None | 0 |
| colonia | 312 | 990 | 1 | 5 | 985 | 0.963 | 899 |
| maranhao | 673 | 13249 | 0 | 31 | 13218 | 0.149 | 2590 |
| pe_v1 | 416 | 7003 | 14 | 33 | 6970 | 1.0 | 6912 |
| pe_v2 | 445 | 6938 | 53 | 94 | 6844 | 1.0 | 6633 |
| pe_v3 | 398 | 5845 | 53 | 88 | 5757 | 0.99 | 5077 |
| pe_v4 | 581 | 0 | 0 | 0 | 0 | None | 0 |

## Verbetes marcados pelo QC que o OCR alcança

232 verbetes marcados têm par no OCR.

| Marca | Com par no OCR | Texto confere |
|---|---|---|
| cota_repetida | 92 | 87 |
| desc_curta | 13 | 1 |
| desc_longa | 5 | 3 |
| fora_de_sequencia | 15 | 15 |
| sem_data | 102 | 99 |
| sem_descricao | 8 | 0 |

Dos 102 marcados `sem_data` com par, o OCR traz um ano em 49.

## Leitura

- Volumes com 0 verbetes são índices (81_para_v3, pe_v4), ainda sem OCR completo (89_cartografia) ou com cota em outro formato (Minas Gerais, que imprime a cota antiga).
- Os Códices (84_codices_cu) não estão no ACTD: o OCR é a única fonte de texto deles, e por isso não há com que comparar.
- O catálogo do Maranhão impresso abrevia o texto (REQ., cap.), enquanto o ACTD o escreve por extenso. A similaridade baixa ali mede a diferença de edição, não erro.
- O ano do OCR é o primeiro número entre 1400 e 1899 no início do verbete. Quando o número do verbete cai nessa faixa ele é lido como ano, então a taxa de ano igual é um piso.

Fontes de cada página, por ordem de preferência: ocr_corrigido, ocr_api, ocr, ocr_lite, ocr_agy_medium, tesseract.
Detalhe por verbete: `conferencia_ocr.jsonl`.
