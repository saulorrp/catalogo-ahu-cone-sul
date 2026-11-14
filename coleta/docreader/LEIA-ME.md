# Varredura do DocReader do Projeto Resgate

Este é o mapa de cada documento do catálogo para a imagem dele no visualizador do Projeto Resgate (Biblioteca Nacional). Com ele, o botão "Imagem no Projeto Resgate" do app abre o documento certo, em vez do portal.

## O problema

As imagens dos microfilmes do Resgate ficam no DocReader da BN, e cada coleção (rolo) tem um código `bib`:

    https://resgate.bn.gov.br/docreader/DocReader.aspx?bib=017_RJ_AV&hf=resgate.bn.gov.br&pagfis=49044

`pagfis` é a página física dentro do rolo. Os catálogos em HTML do gov.br já trazem esse número para 13 capitanias (SC, RS, AL, CE, ES, GO, MT, PB, PI, RN, Rio Negro, SE). O app usava isso desde o PR #19 (cerca de 15,7 mil documentos). Os catálogos que só existem em PDF não trazem o número: Rio de Janeiro, São Paulo, Minas, Bahia, Pernambuco, Maranhão, Pará e outros.

## O que foi testado e descartado (12/11/2026)

| Ideia | Resultado |
|---|---|
| Links arquivados no Wayback Machine | Uns 700 URLs, quase nada do RJ. Serviu só para descobrir os códigos `bib` reais. |
| Tabela de microfilmes nos PDFs do gov.br | Não existe. O PDF do RJ só traz caixa → intervalo de documentos. |
| "Nível 2" do agy (abrir o rolo na página 1) | Códigos `bib` inventados (`017_RJ`, `023_SP`, `005_BA`...) e pouco útil. Ficou só numa cópia local antiga; não foi para o main. |
| Busca do DocReader (`Pesq="Cx. 69, D. 6420"`) | Às vezes acerta, mas recusa números de um dígito ("Menos de 2 letras") e muitas vezes cai no documento errado. |
| Árvore "Pastas" (Telerik) por postback | Existe (Caixa → Doc.), mas só carrega a caixa da página atual, e os postbacks encadeados falharam. |

## O método que funciona

Cada página do DocReader traz, no HTML, o rótulo do documento a que pertence (`AHU_ACL_CU_017, Cx. 69\Doc. 6420`) e o campo `hPagFis`. Basta um GET por página, sem login:

1. Acha o número de páginas do rolo por busca binária (`max_<bib>.txt`).
2. Divide o rolo em 8 trechos, um por thread.
3. Em cada trecho, a partir do início de um documento, salta 3, 6, 12... páginas até o rótulo mudar. Depois volta por busca binária até a primeira página do documento seguinte. Dá uns 5 pedidos por documento, mesmo em rolos de 190 mil páginas.
4. Grava `caixa, doc, pagfis` na hora, em `parcial_<bib>_w<n>.tsv`.

A varredura acha todos os documentos que estão no rolo, inclusive os que o nosso catálogo não tem.

## Acesso

- O servidor da BN só responde a IP brasileiro: precisa de VPN para o Brasil.
- Também barra o navegador embutido do app Claude, mesmo com IP do Brasil. O Chrome e o Python (com User-Agent de Chrome) passam.
- Não depende do Claude: o script roda sozinho no PC.

## Como rodar (Hermerico)

Pasta: `Desktop\resgate_varredura\`

1. Ligue a VPN (Brasil).
2. Dê dois cliques em `Varredura Resgate.bat`. Ele testa uma página de SC e diz "Teste OK" ou "TESTE FALHOU (HTTP 403)".
3. Deixe a janela aberta e o PC acordado. Para pausar, feche a janela.
4. Rode de novo para retomar. Coleções prontas (`resgate_pagfis_<bib>.tsv`) são puladas; as em andamento continuam do último documento gravado em cada trecho. Se a VPN cair, ele espera e tenta de novo em vez de pular.

Ritmo: uns 60 documentos por minuto com 8 threads (varia com a BN e a VPN). Não aumentar sem necessidade, para não arriscar bloqueio.

Fila (18 rolos): `017_RJ_AV`, `023_SP_AV`, `012_CSRP`, `003_BG_AV`, `011_MG`, `015_PE`, `005_BA_AV`, `005_BA_CA`, `005_BA_LF`, `009_MA`, `013_PA`, `050_CS_AV`, `084_OU_AV`, `030_SP_AV`, `089_UL_AV`, `017-1_RJ`, `023-1_SP_MG`, `089_CU`. Lista oficial de coleções: `docreader/docmulti.aspx?bib=resgate`.

## Do resultado ao app

No repositório (`saulorrp/catalogo-ahu-cone-sul`):

1. Copiar `resgate_pagfis_<bib>.tsv` para `coleta/docreader/` (em `.tsv.gz` com `mtime=0`).
2. `python coleta/scripts/juntar_docreader.py`: converte em referência CRAV (`PT/AHU/CU/017/0069/06420`), mapeia casos especiais (`089_UL_AV → 035`, `089_CU → 089`, sub-séries com 3 dígitos e `011_MG` via `old_code`) e acrescenta ao `corpus/resgate_docreader.tsv.gz`. O que veio do gov.br fica como está; documento repetido fica com a primeira página.
3. Refazer `app_dados` com `build/construir_app.py` (atualizando `documentos.parquet` e `manifest.json`).
4. O app lê a coluna `resgate_docreader` (`017_RJ_AV#49044`) e `ahu/links.py` monta o link.

Código: `coleta/scripts/varrer_docreader.py` (espelho do `varrer_resgate.py` otimizado do `andeca`) e `coleta/scripts/juntar_docreader.py`. Fontes: `coleta/FONTES.md`, seção 2.

## Situação final da varredura (14/11/2026 — 18/18 rolos concluídos)

Varredura concluída integralmente no `andeca` em `2026-11-14 00:41:53`: **126.535** linhas mapeadas nos 18 `.tsv` (`125.831` referências novas adicionadas ao `corpus/resgate_docreader.tsv.gz`, totalizando `141.554` entradas no índice junto às `15.723` do gov.br).

| Rolo (`bib`) | Série CRAV | Páginas no rolo | Docs no TSV | Casados no catálogo | Cobertura da série |
|---|---|---:|---:|---:|---:|
| `003_BG_AV` (Brasil Geral) | `003` | 24.353 | 3.817 | 3.677 / 3.785 | 97,1% |
| `005_BA_AV` (Bahia Avulsos) | `005` | 172.151 | 16.967 | 16.944 / 19.612 | 86,4% |
| `005_BA_CA` (Bahia Castro e Almeida) | `005-001` | 73.404 | 3.015 | 2.828 / 7.993 | 35,4%* |
| `005_BA_LF` (Bahia Luísa da Fonseca) | `005-002` | 19.904 | 3.358 | 3.141 / 3.206 | 98,0% |
| `009_MA` (Maranhão) | `009` | 109.612 | 13.118 | 13.078 / 13.126 | 99,6% |
| `011_MG` (Minas Gerais) | `011` | 94.917 | 13.884 | 13.938 / 14.079 | 99,0%** |
| `012_CSRP` (Sacramento/Rio da Prata) | `012` | 9.910 | 1.224 | 657 / 673 | 97,6% |
| `013_PA` (Pará) | `013` | 103.089 | 11.197 | 11.155 / 12.687 | 87,9% |
| `015_PE` (Pernambuco) | `015` | 191.421 | 20.029 | 19.981 / 20.016 | 99,8% |
| `017_RJ_AV` (Rio de Janeiro Avulsos) | `017` | 156.333 | 20.954 | 20.914 / 21.025 | 99,5% |
| `017-1_RJ` (RJ Castro e Almeida) | `017-001` | 42.277 | 6.503 | 6.421 / 11.858 | 54,1%* |
| `023_SP_AV` (São Paulo Avulsos) | `023` | 18.770 | 1.383 | 1.376 / 1.386 | 99,3% |
| `023-1_SP_MG` (SP Mendes Gouveia) | `023-001` | 35.205 | 5.112 | 5.049 / 5.112 | 98,8% |
| `030_SP_AV` (Serviço de Partes) | `030` | 2.134 | 685 | 684 / 684 | 100,0% |
| `050_CS_AV` (Contratos do Sal) | `050` | 1.297 | 138 | 137 / 137 | 100,0% |
| `084_OU_AV` (Visita do Ouro) | `084` | 312 | 54 | 54 / 54 | 100,0% |
| `089_CU` (Sec. Conselho Ultramarino) | `089` | 12.344 | 2.019 | 2.002 / 2.021 | 99,1% |
| `089_UL_AV` (Avulsos Ultramar) | `035` | 14.147 | 3.078 | 2.945 / 3.048 | 96,6% |

\* *Em `005_BA_CA` e `017-1_RJ`, a própria BN agrupa intervalos inteiros de documentos sob um único rótulo de `Doc.` (ex.: `p.10` a `p.24` de `005_BA_CA` trazem todas o rótulo `Doc. 2`, saltando direto para `Doc. 10` na `p.26`).*
\*\* *Em `011_MG`, a `resgate_com_quadro` inclui também a conversão por `old_code` (`cx. N, doc. M`) e 1 entrada já vinda do gov.br.*

No catálogo completo (`app_dados/documentos.parquet`), `resgate_com_quadro` subiu de **15.783** (`main`, 9,4%) → **42.407** (PR #34, 25,3%) → **140.762** de **167.344** documentos (**84,1%**).

Ficam fora desta leva: documentos sem cota CU (`9.770`, ex.: Série Reino `076`, Códices — o rolo `Códices` de 122.949 págs está fora do padrão `Cx./Doc.`), séries sem rolo próprio listado (`059` Brasil-Limites, `065` Montevidéu, `066` Buenos Aires, `071` Paraguai, `081`), `022` Sergipe (não tinha links HTML no índice original do gov.br; há rolo `022_SE` de 3.346 págs no DocReader caso se queira varrer depois), e as coleções fora do catálogo (*Angola*, *Catálogos e Guias* e *IHGB*).

## Achados no DocReader e Infraestrutura

### Mapeamento crítico `bib` → série CRAV (`juntar_docreader.py`)
Conferência direta nas páginas do DocReader e em `app_dados/documentos.parquet` revelou cinco pontos essenciais:
- **Sub-séries CRAV usam 3 dígitos** em `documentos.parquet`: **`005-001`**, **`005-002`**, **`017-001`** e **`023-001`** (não `-01`).
- **`089_UL_AV` (*Avulsos (ULT)*, 14.147 págs):** o HTML da BN traz erroneamente `AHU_ACL_CU_089`, mas as imagens pertencem à **`AHU_ACL_CU_035`** (Série **`035` — Ultramar**).
- **`089_CU` (*Secretaria do Conselho Ultramarino*, 12.344 págs):** corresponde à **`AHU_ACL_CU_089`** (Série **`089`**).
- **`005_BA_CA` (`005-001`) e `005_BA_LF` (`005-002`):** ambas trazem `AHU_ACL_CU_005` no HTML, exigindo mapeamento pelo `bib`; além disso, `005_BA_LF` traz 735 intervalos com `_` (`4_5`), tratados pelo primeiro número inteiro (`RE_DOC_INICIAL`).
- **`011_MG` (`011`, Minas Gerais):** os rótulos do DocReader reiniciam `Doc.` por caixa (`cx. 2, doc. 1`), igual ao `old_code`, enquanto `reference_code` usa numeração contínua (`00075`). `juntar_docreader.py` resolve `(caixa, doc) → reference_code` via `old_code` de `corpus/colecoes/*/documentos.parquet` (levando de 138 para **13.938** documentos casados).

### Preparação no `agiulfo` para subir ao ar (live)
Os resultados organizados já foram transferidos e preparados no worktree limpo `/home/saulor/claude/wt-resgate-docreader` (branch `hydrogen/resgate-varredura-completa`, baseado em `origin/main` + base do PR #34), preservando intacto `/home/saulor/claude/catalogo-ahu-cone-sul`:
- `coleta/docreader/resgate_pagfis_*.tsv.gz` (18 rolos)
- `coleta/scripts/juntar_docreader.py` e `varrer_docreader.py`
- `corpus/resgate_docreader.tsv.gz` (`141.554` referências)
- `app_dados/documentos.parquet` (`140.762` com quadro) e `app_dados/manifest.json`

Para publicar em produção (`https://agiulfo.tailc278e6.ts.net/ahu/`):
1. Revisar e fazer `git push -u origin hydrogen/resgate-varredura-completa` a partir de `/home/saulor/claude/wt-resgate-docreader`.
2. Abrir/fundir o PR para `main` no GitHub (o workflow `.github/workflows/imagem.yml` gera `ghcr.io/saulorrp/catalogo-ahu:latest` e o Watchtower em `~/catalogo-ahu/docker-compose.yml` atualiza o contêiner em produção).

### Infraestrutura no Raspberry Pi (`andeca`, `192.168.1.192`)
- **VPN:** Surfshark via WireGuard puro (`wg-quick@surfshark-br`, saída Brasil/SP), mantendo a LAN `192.168.1.0/24` acessível por SSH direto (`ssh andeca-local`) e `tailscaled` parado no `andeca` para não conflitar DNS (`/etc/resolv.conf`).
- **Scraper (`resgate-varredura.service`):** `varrer_resgate.py` com pool global de 8 threads entre coleções, HTTP/1.1 Keep-Alive, cooldown global em HTTP 403/429/503, escrita atômica e telemetria em `estado.json` (concluído em `2026-11-14 00:41:53`).
- **Painel Web (`resgate-painel.service`):** `http://192.168.1.192:8080` (`painel.py`) com progresso, ETA, saúde do Pi (temperatura, throttling, carga, RAM, disco), IP/país da VPN e log ao vivo.
- **Backup automático:** tarefa agendada `ResgateVarreduraBackupAndeca` no Hermerico rodando `sync_backup_andeca.ps1` a cada 15 minutos, espelhando `.tsv`, `.ok`, `max_*.txt` e `varredura.log` para `Desktop\resgate_varredura\`.
