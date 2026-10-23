# Fontes do catálogo AHU-Brasil

Todas as fontes foram acessadas em 22/10/2026. Cada documento da coleção (`colecao_ahu_brasil/documentos.parquet`) traz a sua própria fonte, em quatro campos:
- `fonte_texto`: a citação completa da fonte de onde vieram o resumo e a cota.
- `fontes`: as fontes que o confirmam. `actd` é o catálogo do AHU no ACTD; `resgate` são as páginas e PDFs do Projeto Resgate.
- `url_actd`: o PDF do ACTD e a página (`pagina_actd`).
- `caminho_resgate`: a página do Projeto Resgate.

## 1. Fonte principal: catálogos do Arquivo Histórico Ultramarino no ACTD/IICT
Os catálogos dos documentos avulsos do fundo Conselho Ultramarino (AHU, Lisboa), série por série, em PDF com texto nativo. Estão publicados no **Arquivo Científico Tropical Digital (ACTD)**, do antigo Instituto de Investigação Científica Tropical (IICT, hoje Universidade de Lisboa).

- **Ficha de cada série:** `https://actd.iict.pt/view/actd:CUc<série>` (ex.: `actd:CUc015` = Pernambuco).
- **PDF de cada série:** `https://actd.iict.pt/eserv/actd:CUc<série>/CU-<Nome>.pdf`.
- **Lista completa usada:** `actd/series_actd.tsv`, com 35 séries. São as capitanias do Brasil (003–023), Bahia Castro e Almeida (005-01) e Luísa da Fonseca (005-02), Rio de Janeiro Castro e Almeida (017-01) e São Paulo Mendes Gouveia (023-01). Entram também Serviço de Partes (030), Ultramar (035), Contratos do Sal (050), Brasil-Limites (059), Montevidéu (065), Buenos Aires (066), Paraguai (071), Reino (076), Visita do Ouro (084) e Conselho Ultramarino (089).
- **Cópias locais:** `actd/*.pdf`. Parser: `scripts/parse_actd.py`.

## 2. Biblioteca Nacional, Projeto Resgate Barão do Rio Branco (gov.br)
- **Instrumentos de pesquisa (HTML):** https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/novos-instrumentos-de-pesquisa. Cobre 13 capitanias, a Série Reino e os Códices 235, 255, 267, 277 e 586.
- **Códices, avulsos e séries:** https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/catalogo-de-verbetes-codices-avulsos-e-series. Cobre os Códices 274, 584 e 610, e os Contratos por capitania.
- **Catálogos temáticos:** https://www.gov.br/bn/pt-br/central-de-conteudos/projeto-resgate/catalogos-tematicos. Cobre mulheres, escravidão, irmandades, povos indígenas, semiárido e Caminhos/Rio da Prata. São usados só para marcar o campo `temas`, e entram como registro próprio apenas quando a cota não existe em outra fonte.
- **PDFs de texto do Rio de Janeiro** (Avulsos e Castro e Almeida), no mesmo site: `fontes/*.pdf`.
- **Páginas salvas:** `raw/`. Árvore de URLs: `resgate_arvore.json`.

## 3. Fontes usadas só para conferência (não alimentam o texto da coleção)
- **Imagens dos catálogos impressos na biblioteca digital docvirt (`crv_guiafontes`):** Bahia, São Paulo, Minas Gerais, Pará, Ultramar/Brasil-Geral, Códices do CU e Cartografia. A transcrição foi feita por OCR com Gemini e Tesseract, no PC Hermerico (`resgate_ocr\`).
- **PDFs escaneados da BN (objdigital.bn.br):** Pernambuco (mss1700651–mss1700654), Maranhão (mss1069572) e Colônia (mss1054717). Também com OCR.
- **Castro e Almeida, "Inventario dos documentos relativos ao Brasil existentes no Archivo de Marinha e Ultramar de Lisboa":** publicado nos Anais da Biblioteca Nacional (vols. 31, 32, 34, 36, 37, 39, 46, 50 e 71). Separatas no archive.org e PDFs em `hemeroteca/`. Detalhes em `hemeroteca/PLANO_DOWNLOAD.md`.

## 4. Códigos de referência
`reference_code`, `cota_resgate` e a cota antiga das séries de numeração igual (017-01, 023-01) são gerados pelo conversor do app (`ahu/conversor.py`), a partir da cota. Não foram copiados do DigitArq.
