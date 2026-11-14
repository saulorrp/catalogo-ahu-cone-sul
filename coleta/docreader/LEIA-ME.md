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

Fila: `017_RJ_AV`, `023_SP_AV`, `012_CSRP`, `003_BG_AV`, `011_MG`, `015_PE`, `005_BA_AV`, `005_BA_CA`, `005_BA_LF`, `009_MA`, `013_PA`, `050_CS_AV`, `084_OU_AV`, `030_SP_AV`, `089_UL_AV`. Lista oficial de coleções: `docreader/docmulti.aspx?bib=resgate` (o script a salva em `docmulti.html`).

## Do resultado ao app

No repositório (`saulorrp/catalogo-ahu-cone-sul`):

1. Copiar `resgate_pagfis_<bib>.tsv` para `coleta/docreader/` (em .gz).
2. `python coleta/scripts/juntar_docreader.py`: converte em referência CRAV (`PT/AHU/CU/017/0069/06420`) e acrescenta ao `corpus/resgate_docreader.tsv.gz`. O que veio do gov.br fica como está; documento repetido fica com a primeira página.
3. Refazer `app_dados` com `build/construir_app.py` (ou `scripts/reconstruir.sh`, que confere os outros arquivos byte a byte).
4. O app lê a coluna `resgate_docreader` (`017_RJ_AV#49044`) e `ahu/links.py` monta o link.

Código: `coleta/scripts/varrer_docreader.py` (igual ao `varrer_resgate.py` da pasta) e `coleta/scripts/juntar_docreader.py`. Fontes: `coleta/FONTES.md`, seção 2.

## Situação (13/11/2026)

| Coleção | Documentos | Estado |
|---|---|---|
| RJ avulsos (`017_RJ_AV`) | 20.954 | no PR #34 |
| SP avulsos (`023_SP_AV`) | 1.383 | no PR #34 |
| Sacramento (`012_CSRP`) | 662 | no PR #34 |
| Brasil Geral (`003_BG_AV`) | 3.694 | no PR #34 |
| Minas Gerais (`011_MG`) | ~14 mil | varrido, falta juntar |
| PE, BA (3 rolos), MA, PA e menores | ~105 mil | em andamento |

Com o PR #34, o app tem o quadro exato de 42.407 dos 167.344 documentos (antes, 15.724). Com a fila completa, deve passar de 80%.

Não entram: RJ Castro e Almeida (017-001) e SP Mendes Gouveia (023-001), que não apareceram na lista de rolos online (a conferir no `docmulti.html`), e os documentos sem cota CU (Reino, Códices).
