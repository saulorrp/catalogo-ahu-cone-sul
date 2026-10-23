# Como o banco foi montado: coleta, transcrição, conferência e cura das cotas

Este documento descreve, passo a passo, como os 167.344 documentos do Catálogo AHU-Brasil
chegaram ao app. A ideia é que qualquer pessoa possa refazer o caminho, conferir uma decisão ou
apontar um erro. Os números são os de 22–23/10/2026.

## 1. Duas coleções com estatutos diferentes

| Coleção | Documentos | O que traz | Como foi feita |
|---|---|---|---|
| **ahu-sul** | 7.051 | Resumo, data, cota e também pessoas identificadas, tipologia revisada e variáveis sociolinguísticas (SRSP e outras) | Transcrição e enriquecimento descritos no preprint (Rocha, 2026) |
| **ahu-brasil** | 160.293 | Resumo, data, local, série, cota e links para as fontes | Coleta automatizada dos catálogos publicados, descrita aqui |

Um documento que está nas duas fontes entra no app **uma vez só**, na versão da ahu-sul, que é a
mais rica. A coleta encontrou 7.031 dos 7.051 documentos da ahu-sul pela cota. Os 20 restantes
estão listados no arquivo de junção.

## 2. Fontes

A lista completa, com endereços e data de acesso, está em `FONTES.md`, na pasta da coleta.

1. **Fonte principal: catálogos do próprio AHU no ACTD/IICT** (Arquivo Científico Tropical
   Digital, Universidade de Lisboa). São 35 séries do fundo Conselho Ultramarino relativas ao
   Brasil, em PDF com texto nativo, um por série
   (`https://actd.iict.pt/eserv/actd:CUc<série>/CU-<Nome>.pdf`). Daqui vêm o resumo, a data e a
   cota de quase todos os documentos.
2. **Projeto Resgate "Barão do Rio Branco" (Biblioteca Nacional, gov.br).** Instrumentos de
   pesquisa em HTML (13 capitanias, Série Reino, Códices) e PDFs de texto do Rio de Janeiro.
   Servem para três coisas: dar o link de cada documento no site do Resgate, conferir o texto do
   ACTD e cobrir o que o ACTD não tem (Códices, parte da Série Reino, Sergipe).
3. **Catálogos temáticos do Resgate** (mulheres, escravidão, irmandades, povos indígenas,
   semiárido, Rio da Prata). Marcam só o campo `temas`. Viram documento próprio apenas quando a
   cota não aparece em nenhuma outra fonte.

## 3. Coleta (scraping)

- **Mapa do site.** Um rastreador percorreu as páginas do Projeto Resgate no gov.br e gravou a
  árvore de endereços (`resgate_arvore.json`). Um segundo passe achou um ramo que o primeiro
  perdeu ("catálogo de verbetes, códices, avulsos e séries").
- **Páginas e PDFs.** Cada página HTML foi salva como veio (`raw/`), antes de qualquer
  processamento, para que a leitura possa ser refeita. Os PDFs do ACTD e do Rio de Janeiro também
  foram guardados.
- **Leitura.** Há um leitor (parser) para cada formato: tabela HTML, texto corrido de PDF,
  Sergipe, Códices e catálogos do ACTD. Cada um separa os verbetes e extrai a data e o local, o
  resumo, os anexos, a cota antiga e a cota atual.
- **Ritmo.** Os servidores foram consultados devagar, com retomada de downloads interrompidos.
  Nenhuma área restrita foi acessada.

## 4. Catálogos que só existiam como livro escaneado

Bahia, São Paulo, Minas Gerais e Pará, no Resgate, só estão como imagens das páginas dos
catálogos impressos (biblioteca digital docvirt). Pernambuco, Maranhão e Colônia do Sacramento
estão como PDFs escaneados na BN (objdigital), cuja camada de texto embutida é pouco confiável.
Para esses volumes montou-se uma esteira de **transcrição automatizada**:

1. **Download das páginas** (cerca de 4.600 no docvirt, mais os volumes da BN).
2. **Transcrição com Gemini Flash**, com instruções fixas (`PROMPT_OCR.md`): transcrever
   literalmente, sem corrigir ortografia nem desdobrar abreviaturas, ler as colunas na ordem
   certa, juntar palavras hifenizadas e separar os verbetes por linha em branco.
3. **Páginas recusadas pelo modelo** (cerca de 10%, por filtro de recitação): passaram primeiro
   pelo Tesseract 5.4, e o Gemini corrigiu o texto do Tesseract olhando a imagem.
4. O texto transcrito passa pelo mesmo leitor dos outros catálogos.

**Como isso entrou no banco.** No meio do trabalho achou-se que o próprio AHU publica o texto
desses catálogos no ACTD. Como o texto nativo do arquivo é mais confiável que qualquer OCR, ele
virou a fonte do resumo e da cota, e a transcrição automatizada passou a servir de
**conferência**: ela não alimenta o texto da coleção. A parte paga foi interrompida por decisão de
custo. O OCR gratuito dos Códices do Conselho Ultramarino, que não estão no ACTD, segue em
andamento: para eles o OCR é a única fonte de texto.

**Resultado da conferência (23/10/2026).** O script `coleta/scripts/conferir_ocr.py` comparou,
pela cota, 74.696 verbetes transcritos com o verbete do mesmo documento no banco. Fora o Maranhão,
cujo catálogo impresso abrevia o texto (REQ., cap.) e por isso não se compara palavra a palavra,
são 61.478 pares, e o texto confere (similaridade ≥ 0,9) em 97,1% deles. Nos volumes da Bahia,
de Pernambuco e de São Paulo a mediana fica entre 0,99 e 1,000; no Pará e na Colônia, perto de
0,96. O ano confere em 99,9% dos verbetes da Bahia e entre 81% e 100% nos outros volumes. É uma
estimativa por baixo, porque o número do verbete às vezes é lido como ano. Os 232 verbetes marcados pelo
QC que o OCR alcança confirmam, em geral, o texto do ACTD, e o OCR traz o ano de 49 dos 102 marcados
"sem data". O relatório completo está em `coleta/qc/RELATORIO_CONFERENCIA_OCR.md`.

## 5. Junção e identificadores

- Cada verbete recebe uma **chave de cota** `série|caixa|documento`, normalizada (ver §6).
- O ACTD é a fonte principal. Quando o gov.br tem a mesma chave, acrescenta o link do Resgate e
  entra na conferência de texto.
- Cada documento recebe um `doc_uid` (`ahu-brasil:…`) **estável**: refazer a coleta mantém o
  mesmo identificador para o mesmo documento. Dos 51.947 uids da primeira versão, 16 mudaram, todos
  por correção da cota.
- Cada documento carrega a sua proveniência: `fonte_texto` (citação), `url_actd` e `pagina_actd`
  (o PDF e a página de onde veio) e `caminho_resgate` (a página no Resgate).

## 6. Cura das cotas

As cotas aparecem escritas de muitos jeitos nas fontes. O conversor do app (`ahu/conversor.py`)
reconhece todas e gera as três formas a partir de uma só:

- **Cota do Resgate:** `AHU_ACL_CU_017, Cx. 69, D. 6420`
- **Código de referência do AHU/DigitArq:** `PT/AHU/CU/017/0069/06420`
- **Cota antiga:** só nas séries que guardam a numeração atual (017-01, 023-01).

Os códigos de referência foram **calculados** pelo conversor, não copiados do DigitArq.

Variantes curadas:
- `AHU_CU_` sem `ACL`, como no ACTD, vira `AHU_ACL_CU_`;
- subsérie com zero à esquerda: `0005-01` vira `005-01`;
- pontuação trocada: `Cx. 33. D.` vira `Cx. 33, D.`;
- separador de milhar no número do documento: `D. 20.836` vira `D. 20836` (Bahia, Castro e
  Almeida);
- pastas com a marca da coleção (`Bahia-CA`, `Bahia-LF`) são agrupadas sob o nome da série;
  grafias como `Espirito Santo` são acertadas.

A cifra das séries (número → capitania ou série) cobre agora todas as séries do Brasil: 003 a
023, 005-01, 005-02, 017-01, 023-01, 030, 035, 050, 059, 065, 066, 071, 081, 084 e 089.

**O que não foi curado, de propósito.** Cerca de 849 cotas aparecem repetidas e 310 verbetes não
têm cota. Corrigi-los exigiria adivinhar o número a partir da sequência do catálogo. Preferiu-se
deixá-los como estão na fonte, com uma marca de controle de qualidade visível, a inventar uma cota.
A Série Reino e os Códices não têm cota no formato série/caixa/documento e ficam fora do
conversor.

## 7. Controle de qualidade

Um script de conferência marca cada documento suspeito, sem alterar o texto. As marcas ficam no
campo `qc`. Dos 167.324 verbetes coletados, 4.424 (2,6%) têm alguma marca:

| Marca | Verbetes | O que quer dizer |
|---|---|---|
| sem_data | 1.959 | Nenhum ano entre 1400 e 1899 nem "s.d." na linha da data |
| cota_repetida | 1.771 | Outra entrada tem a mesma chave de cota |
| diverge_resgate | 450 | O texto do ACTD e o do Resgate, para a mesma cota, têm similaridade abaixo de 0,6 |
| sem_cota | 310 | Nenhuma cota reconhecível |
| fora_de_sequencia | 133 | O número do documento destoa muito dos vizinhos no mesmo catálogo |
| desc_curta / desc_longa / sem_descricao | 85 | Resumo vazio, curto demais ou longo demais (provável junção de verbetes) |

Nos 49.173 pares em que ACTD e Resgate descrevem a mesma cota, a similaridade mediana do texto é
1,000, e 97,2% dos pares ficam acima de 0,9.

## 8. Do banco ao app

- **Importação** (`build/importar_colecao.py`): a coleção entra no corpus com o hash do arquivo de
  origem e o commit do código registrados em `camada.json`.
- **Tipologia:** classificação automática por regras, marcada como tal (`typology_status`).
- **Busca por palavras:** índice BM25 gravado em disco, com resultado idêntico ao da biblioteca de
  referência (`rank_bm25`), conferido por teste.
- **Busca por sentido:** vetores `multilingual-e5-small` de todos os documentos. Na versão gratuita
  do app (Streamlit Cloud) essa busca fica desligada por falta de memória. A versão completa roda em
  servidor próprio.

## 9. Ciência aberta: o que está disponível e o que falta

- **No repositório:** o código do app, o conversor de cotas, a importação, os índices, os testes e
  os dados derivados (`corpus/`, `app_dados/`, CC BY 4.0).
- **Coleta:** os scripts de coleta, os leitores dos catálogos, o controle de qualidade, a
  conferência do OCR e os scripts de OCR estão em [`coleta/`](../coleta/), publicados como rodaram.
- **Ainda fora do repositório:** as páginas baixadas, as imagens e o texto do OCR, por tamanho.
  Podem ser refeitos com os scripts.
- **Direitos:** resumos e cotas pertencem ao Arquivo Histórico Ultramarino e ao Projeto Resgate,
  que são creditados em cada documento. O que este projeto acrescenta é a organização, a
  normalização das cotas, as marcas de qualidade e, na ahu-sul, as camadas de análise.
- **Uso de IA:** modelos de linguagem foram usados para transcrever páginas escaneadas (Gemini
  Flash, só para conferência), para escrever e revisar o código da coleta e, na ahu-sul, para as
  estimativas descritas no preprint. Nenhum resumo do catálogo foi escrito ou reescrito por IA.

Erros encontrados podem ser comunicados como *issue* no repositório. Cada correção fica registrada
no histórico.
