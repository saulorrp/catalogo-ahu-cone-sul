# Registro detalhado — Varredura do DocReader, endurecimento web e publicação (`14/11/2026`, PR #35 / `9f6d9ff`)

Registro técnico consolidado das alterações integradas ao repositório `saulorrp/catalogo-ahu-cone-sul` no commit [`9f6d9ff`](https://github.com/saulorrp/catalogo-ahu-cone-sul/commit/9f6d9ff) (PR `#35`, substituindo o PR `#34`) e implantadas em produção no servidor `agiulfo` (<https://agiulfo.tailc278e6.ts.net/ahu/>).

---

## 1. Resumo dos números

| Indicador | Antes (`main` anterior) | Depois (`9f6d9ff` em produção) | Fonte / verificação |
|---|---:|---:|---|
| Documentos no catálogo (`app_dados/documentos.parquet`) | 167.344 | 167.344 | `app_dados/manifest.json` (`contagens.documentos`) |
| Documentos com quadro no DocReader (`resgate_com_quadro`) | 15.783 (9,4%) | **141.298 (84,4%)** | `app_dados/manifest.json` (`acertos.resgate_com_quadro`) |
| Entradas no índice `corpus/resgate_docreader.tsv.gz` | ~15.723 | **142.057** | Contagem direta de linhas em `corpus/resgate_docreader.tsv.gz` |
| Linhas brutas mapeadas em `coleta/docreader/*.tsv.gz` | 0 | **127.520** (127.029 de 19 rolos + 491 de Sergipe gov.br) | Contagem direta nos 20 `.tsv.gz` |
| Documentos ainda sem quadro (`resgate_docreader` vazio) | 151.561 | **26.046** | Diferença `167.344 - 141.298` |
| Suíte `scripts/gate.py` (`ruff` + `pytest`) | — | **595 passed, 6 skipped, 1 warning** (`PASS`) | Execução integral antes do PR `#35` |

---

## 2. Varredura do DocReader e reconciliação em `corpus/`

### 2.1 Coleta nos 19 rolos do DocReader (`coleta/scripts/varrer_docreader.py`)

A busca nativa `Pesq=` do DocReader da Biblioteca Nacional falha para números de um dígito e aponta documentos errados. Cada página HTML do visualizador (`DocReader.aspx?bib=<rolo>&hf=resgate.bn.gov.br&pagfis=<n>`) expõe, porém, a etiqueta do documento (ex.: `AHU_ACL_CU_017, Cx. 69\Doc. 6420`) e o campo `hPagFis`. O script `coleta/scripts/varrer_docreader.py` localiza a fronteira inicial (`pagfis`) de cada documento por salto exponencial (`3, 6, 12…` páginas) seguido de busca binária, com 8 threads e retomada tolerante a quedas de rede/VPN. A varredura completa rodou no Raspberry Pi `andeca` via IP brasileiro.

Arquivos gerados e versionados em `coleta/docreader/`:

| Arquivo | Linhas de dados | Observações de mapeamento |
|---|---:|---|
| `resgate_pagfis_003_BG_AV.tsv.gz` | 3.817 | Série `003` (Brasil Geral) |
| `resgate_pagfis_005_BA_AV.tsv.gz` | 16.967 | Série `005` (Bahia Avulsos) |
| `resgate_pagfis_005_BA_CA.tsv.gz` | 3.015 | Sub-série `005-001` (Bahia Castro e Almeida); lacunas agrupadas tratadas na frente de capilhas |
| `resgate_pagfis_005_BA_LF.tsv.gz` | 3.358 | Sub-série `005-002` (Bahia Luísa da Fonseca) |
| `resgate_pagfis_009_MA.tsv.gz` | 13.118 | Série `009` (Maranhão) |
| `resgate_pagfis_011_MG.tsv.gz` | 13.884 | Série `011` (Minas Gerais); `Doc.` reinicia por caixa → casado via `old_code` (`carregar_mapa_mg()`) |
| `resgate_pagfis_012_CSRP.tsv.gz` | 1.224 | Série `012` (Colônia do Sacramento / Rio da Prata) |
| `resgate_pagfis_013_PA.tsv.gz` | 11.197 | Série `013` (Pará) |
| `resgate_pagfis_015_PE.tsv.gz` | 20.029 | Série `015` (Pernambuco) |
| `resgate_pagfis_017_RJ_AV.tsv.gz` | 20.954 | Série `017` (Rio de Janeiro Avulsos) |
| `resgate_pagfis_017-1_RJ.tsv.gz` | 6.503 | Sub-série `017-001` (Rio de Janeiro Castro e Almeida); lacunas agrupadas em recuperação por capilhas |
| `resgate_pagfis_022_SE.tsv.gz` | 494 | Série `022` (Sergipe DocReader: 6 caixas contínuas `1..495`) |
| `resgate_pagfis_023_SP_AV.tsv.gz` | 1.383 | Série `023` (São Paulo Avulsos) |
| `resgate_pagfis_023-1_SP_MG.tsv.gz` | 5.112 | Sub-série `023-001` (São Paulo Mendes Gouveia) |
| `resgate_pagfis_030_SP_AV.tsv.gz` | 685 | Série `030` |
| `resgate_pagfis_050_CS_AV.tsv.gz` | 138 | Série `050` (Ceará/Sergipe/Capitanias segundo código do rolo) |
| `resgate_pagfis_084_OU_AV.tsv.gz` | 54 | Série `084` |
| `resgate_pagfis_089_CU.tsv.gz` | 2.019 | Série `089` (Secretaria do Conselho Ultramarino) |
| `resgate_pagfis_089_UL_AV.tsv.gz` | 3.078 | Etiqueta HTML traz `089`, mas as imagens pertencem à série `035` (Ultramar/Geral) |
| `sergipe_govbr_pagfis.tsv.gz` | 491 | Pares `(caixa, doc_local, sufixo, pagfis)` extraídos de `avulsos-sergipe-1619-1822.html` (`gov.br`) |
| **Total** | **127.520** | **127.029** dos 19 rolos + **491** do HTML de Sergipe |

### 2.2 Consolidação e regras especiais (`coleta/scripts/juntar_docreader.py`)

O script lê os TSVs de `coleta/docreader/`, normaliza cada registro para referência CRAV (`PT/AHU/CU/<série>/<caixa>/<doc>`) preservando as referências já existentes vindas do `gov.br` e gravando sempre a menor `pagfis` quando um documento aparece em mais de um trecho:

- **`089_UL_AV` → série `035`:** corrige o cabeçalho trocado do HTML da BN (`AHU_ACL_CU_089` nas imagens da série `035`), separando-o de `089_CU` (série `089`).
- **`011_MG` (`carregar_mapa_mg()`):** o DocReader numera os documentos reiniciando dentro de cada caixa, ao contrário do `reference_code` contínuo; o casamento usa `(cx, doc)` em `old_code` das coleções em `corpus/colecoes/*/documentos.parquet`.
- **`022_SE` (`carregar_quadros_se()`):** combina `resgate_pagfis_022_SE.tsv.gz` (sequência `1..495`) e `sergipe_govbr_pagfis.tsv.gz` usando `document_id_and_date`, `old_code` e `(caixa, doc_local, sufixo)`, evitando o deslocamento causado pelas caixas `3-A`/`3-B` (`PT/AHU/CU/022/0001/00004` → `022_SE#64`) e cobrindo **536 / 542** documentos de `PT/AHU/CU/022/` (**98,9%**; 503 referências únicas casadas).
- **Códices (`govbr_arvore.json`):** os 3.064 links de Códices no `gov.br` não foram incorporados nesta etapa porque os 2.125 registros correspondentes no catálogo ainda têm `reference_code = None`.

### 2.3 Integração ao app e reconstrução de `app_dados/`

- `build/cotas.py` e `ahu/links.py`: atualização da documentação e docstrings refletindo o índice ampliado; `cotas.ajustar()` preenche `documentos.parquet["resgate_docreader"]` e contabiliza `resgate_com_quadro = 141298` via `build/construir_app.py`.
- `app_dados/documentos.parquet` e `app_dados/manifest.json`: reconstruídos com os 28 artefatos SHA-256 verificados em disco.
- `coleta/FONTES.md` e `coleta/docreader/LEIA-ME.md`: documentam as fontes, restrição de IP brasileiro na BN e o fluxo de coleta/junção.
- `tests/test_link_busca.py` (`test_indice_do_resgate_tem_o_quadro_achado_pela_varredura`): valida diretamente em `corpus/resgate_docreader.tsv.gz` os quadros de `017_RJ_AV`, `021_SC`, `011_MG`, `017-1_RJ`, `023-1_SP_MG`, `089_UL_AV → 035`, `089_CU` e `022_SE`.

---

## 3. Endurecimento web

Integrado ao mesmo pacote para proteger o serviço público em `agiulfo`:

- `web/main.py`: desativa `/docs`, `/redoc` e `/openapi.json` por padrão (`docs_url=None, redoc_url=None, openapi_url=None`), reabilitáveis apenas via `AHU_DEV_DOCS=1|sim|true` (aplicado tanto no `inner_app` quanto no `app` pai quando `AHU_ROOT_PATH` está definido).
- `web/rotas_paginas.py`: define `MAX_DOCS_PDF = 100`; impede adicionar além de 100 documentos em `/selecao/adicionar/{doc_uid}` com mensagem orientando baixar o PDF e esvaziar a seleção, e retorna HTTP `400` em `/selecao/pdf` se o limite for excedido.
- `tests/test_web.py` (`test_docs_desligados_por_padrao_e_limite_pdf`): cobre `404` em `/docs`, `/redoc`, `/openapi.json` e `400` em `/selecao/pdf` acima de `MAX_DOCS_PDF`.

---

## 4. Publicação, build e verificação em produção (`14/11/2026`)

1. **Validação e CI:** `python scripts/gate.py` aprovado com **595 passed, 6 skipped, 1 warning**; workflow `Gate (Ruff + Pytest)` verde no PR `#35`; merge em `main` (`9f6d9ff`) e fechamento do PR `#34`.
2. **Imagem Docker e contêiner:** workflow `Imagem Docker` (run `43598372351`, `1m4s`) publicou `ghcr.io/saulorrp/catalogo-ahu:latest`. Em `~/catalogo-ahu` no `agiulfo`, `sudo -n docker compose pull && sudo -n docker compose up -d` recriou `catalogo-ahu-app-1` (`sha256:c470d364d73cd948ca821c554a6f8637c1dc75a2b3081cac81c6c2ab8f8c2100`), e `~/.local/bin/catalogo-git-sync.sh` sincronizou `~/claude/catalogo-ahu-cone-sul` (`ok: 9f6d9ff`).
3. **Conferência em produção (HTTPS via Tailscale Funnel):**
   - `app_dados/documentos.parquet` dentro de `catalogo-ahu-app-1`: `167.344` documentos, `141.298` com `resgate_docreader`.
   - `https://agiulfo.tailc278e6.ts.net/ahu/doc/ahu-sul:c79fd1d789` → HTTP `200`, `"Imagem no Projeto Resgate"`, link `bib=003_BG_AV&hf=resgate.bn.gov.br&pagfis=4644`.
   - `https://agiulfo.tailc278e6.ts.net/ahu/doc/ahu-brasil:c11802ede1` (`PT/AHU/CU/011/0001/00001`) → HTTP `200`, link `bib=011_MG&hf=resgate.bn.gov.br&pagfis=1`.
   - `https://agiulfo.tailc278e6.ts.net/ahu/doc/ahu-brasil:eabc8842f1` (`PT/AHU/CU/022/0002/00173`) → HTTP `200`, link `bib=022_SE&hf=resgate.bn.gov.br&pagfis=1023`.
   - `https://agiulfo.tailc278e6.ts.net/ahu/docs` → HTTP `404`.

---

## 5. Próxima frente em andamento (fora deste commit)

- **Recuperação de capilhas agrupadas (`AGRUPAMENTO_BN`):** os documentos não localizados na varredura por HTML pertencem majoritariamente a **capilhas multi-documento** nos fundos Castro e Almeida (`017-1_RJ`: ~5.381 docs faltantes em 2.000 pastas-alvo; `005_BA_CA`: ~5.155 docs em 1.558 pastas). Está em execução no `andeca` + `agiulfo` um pipeline separado (`resgate-capilhas.service` + `resgate-capilhas-ocr.service`, Tesseract 5.4 + escalonamento controlado ao Gemini Flash-Lite com teto de US$ 5,00) para ler as imagens de capa e recuperar esses `pagfis` sem alterar o repositório até a validação final dos pares recuperados.
