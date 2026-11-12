# Limitações Conhecidas e Qualidade dos Dados

Todos os indicadores abaixo foram calculados diretamente a partir dos dados do catálogo e dos relatórios de auditoria automatizada.

1. **Cotas não curadas manualmente**: 160.293 documentos da coleção `ahu-brasil` preservam as cotas extraídas de catálogos históricos sem revisão manual exaustiva item a item.
2. **Intervalos absurdos**: 19 registros apresentam intervalos numéricos inconsistentes detectados pela verificação `intervalo_absurdo`.
3. **Cotas ilegíveis ou vazias**: 227 documentos possuem cota ilegível (222) ou cota vazia (5).
4. **Códices transcritos exclusivamente por OCR**: 2.125 documentos agrupados na série `Códices` derivam de reconhecimento óptico de caracteres sem conferência paleográfica exaustiva.
5. **Tipologias com erro de extração (`PARA`)**: 687 documentos possuem tipologia atribuída como `PARA`, reflexo de falso positivo de extração sintática nos Códices.
6. **Desempenho de SRSP**: na avaliação da camada SRSP (n = 716), obteve-se MAE = 1.13, r = 0.48 e viés = 0.59. A linha de base de prever sempre o valor 2 atinge MAE 0,97 segundo o preprint §3.1 (Rocha, 2026).
7. **Avaliação da busca restrita a modelo avaliador**: avaliação conduzida pelo juiz Gemini sobre 960 pares consulta-documento às cegas (notas 0, 1 e 2). Nenhum julgamento humano foi publicado ainda.
8. **Divergência binária em `lexico.sqlite`**: o banco SQLite reconstruído reproduz exatamente o conteúdo lógico e as tabelas, mas pode divergir em bytes devido à alocação interna de páginas de diferentes versões do motor SQLite.
9. **Documentos sem resumo**: 30 registros não dispõem de texto de resumo no catálogo de origem.
10. **Documentos sem datação de ano**: 2.390 registros não possuem ano atribuível na coluna `ano_isolado`.
