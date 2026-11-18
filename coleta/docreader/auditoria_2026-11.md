# Auditoria dos links do Projeto Resgate (17–18/11/2026)

Este é o registro final da auditoria por amostragem dos links do catálogo para o DocReader da Biblioteca Nacional (`resgate.bn.gov.br`). Ele também registra as correções que a auditoria motivou. Os horários são de Brasília.

## Resumo

- Foram conferidos **2.550 links**: 850 no piloto e 1.800 na auditoria completa, em amostras que não se repetem.
- **Erros achados em links:**
  - 146 na série 012 do rolo `012_CSRP`, achados no piloto e corrigidos no commit `3498a77`;
  - 4 nas capilhas da Fase 2 do rolo `005_BA_CA`, corrigidos aqui com a regra que os produzia.
- **Margem de erro da coleção.** Depois das correções, o catálogo tem **151.325 documentos com link**. A taxa estimada de links errados é de **0,03%** antes desta correção e de cerca de 0% depois. Com 95% de confiança, ela fica **abaixo de 0,23%**, cerca de 350 links. Uma conta mais conservadora, que soma os limites de cada grupo, dá **0,45%**. A conta está abaixo.
- **Defeitos do próprio catálogo, que não são do link:**
  - pares de linhas duplicadas: 48 receberam o link;
  - caixas de Sergipe que divergem da BN: 3 casos, não corrigidos.

## Como foi feita

Os links foram separados em cinco grupos pela origem, no histórico git de `corpus/resgate_docreader.tsv.gz`:

| Grupo | Origem | População |
|---|---|---:|
| A | gov.br HTML (`c4fd3a0`, PR #19) | 15.723 |
| B | varredura do DocReader, rolos comuns (`9f6d9ff`) | 108.781 |
| C | varredura, mapeamentos especiais: 011 pela cota antiga, 012, 022 (Sergipe) e 035 (`089_UL_AV`) | 17.892 |
| D | capilhas, Fase 1 (`31d10a4`) | 4.546 |
| E | capilhas, Fase 2 (`31d10a4`) | 4.922 |

**Grupos A, B e C.** A conferência foi automática, no Raspberry Pi `andeca`, com VPN no Brasil (`auditar_links.py`). Para cada link, ela conferiu três coisas:

1. se o rótulo da BN na página (`AHU_ACL_CU_<série>, Cx. N\Doc. N`) bate com a cota;
2. se a página anterior tem outro rótulo, ou seja, se o link cai na primeira página;
3. se o rótulo não reaparece nas 10 páginas seguintes ao fim do documento.

**Grupos D e E.** A página do link é uma capilha, isto é, a capa da pasta do documento. As páginas foram baixadas e a anotação vertical da margem esquerda ("Caixa N, Doc. N.º A–B") foi lida às cegas, sem que o leitor visse a cota esperada. A cota foi comparada depois.

- No piloto, leram o `gemini-3.1-flash-lite` e o `gemini-3.8-flash` (cerca de US$ 0,15), e o saulo conferiu 22 páginas à mão.
- Na auditoria completa, leu o agy (`gemini-3.8-flash-medium`). As divergências foram relidas pelo `gemini-4-argon-high`, e as que restaram o Claude conferiu na imagem.

| Rodada | Quando | Amostra | Pedidos à BN |
|---|---|---|---|
| Piloto (semente 20261117) | 17/11, 16:50–19:40 | A 100, B 300, C 150, D 150, E 150 | 10.188, 0 erros HTTP |
| Completa (semente 20261118, sem repetir o piloto) | 17/11, 19:30–21:55 | A 200, B 600, C 400 (100 por série especial), D 300, E 300 | 22.154, 0 erros HTTP |

## Resultados

| Grupo | Conferidos | Erros de link | Limite superior de 95% (Clopper-Pearson) |
|---|---:|---:|---:|
| A | 300 | 0 | 0,99% |
| B | 898 | 0 (*) | 0,33% |
| C | 550 | 0 | 0,54% |
| D | 450 | 0 | 0,66% |
| E | 450 | 4 | 2,02% antes da correção |

(*) As 2 falhas de B no piloto eram do rolo `012_CSRP` e levaram à correção de `3498a77`.

**O rolo `012_CSRP` (Colônia do Sacramento).** Ele mistura as séries 012, 059, 065, 066 e 071 com a mesma numeração de caixa e documento. Dos 657 links da série 012, 146 abriam documento de outra série. A correção, de 17/11, 18:30–18:50, separa as séries pelo rótulo da BN. Outros rolos com pares repetidos foram conferidos (264 links) e estavam certos.

**Os erros das capilhas (E).** Os quatro estão no rolo `005_BA_CA`, e todos abrem a capilha de um documento anterior:

| Cota | O link abria |
|---|---|
| `PT/AHU/CU/005-001/0117/23004` | a capilha de 23.002–23.003 |
| `PT/AHU/CU/005-001/0104/20308` | a capilha de 20.300–20.303 |
| `PT/AHU/CU/005-001/0070/13432` | a capilha de 13.430–13.431 |
| `PT/AHU/CU/005-001/0066/12708` | a primeira página do doc. 12.703 |

**Causa:** a Fase 2 guardava em `resolved_docs` todos os números que o Tesseract via na página, mesmo quando o Gemini tinha lido ali uma faixa menor. Por exemplo, o Tesseract leu 20.300 e 20.308, e o Gemini, 20.300–20.303.

**Falhas que são do catálogo, e não do link:**

- **A, 2 casos:** `010/0006/00434` e `018/0008/00450`. O catálogo tem pares de linhas com a mesma cota antiga e o mesmo verbete, mas cotas novas diferentes: `010/0006/00434` e `010/0007/00434`, e `018/0008/00450` e `016/0008/00450`. O link abre o documento certo, mas estava só numa das linhas, e a BN o rotula com a cota da outra (Cx. 7; série 016).
- **C, 6 casos de Sergipe:** 3 no piloto e 3 na completa. O link cai no documento de mesmo número, mas a BN o põe noutra caixa. No piloto, o erro era da regra de conferência. Nos 3 da completa (`022/0002/00230`, `00273` e `00439`), a BN diz caixa 3 ou 6. A caixa 2 do catálogo tem números fora da sua faixa (até 466), o que sugere caixa errada no catálogo, mas isso não foi confirmado.

## Correções (18/11/2026)

1. **`juntar_capilhas.py`, regra 6.** Quando o Gemini leu a página, só entram os números que ele confirma, isto é, os da lista explícita ou os da faixa lida. A opção `--refazer` retira os links que as capilhas trouxeram e os junta de novo.
   - **Resultado:** 76 links retirados (56 do `005_BA_CA` e 20 do `017-1_RJ`), 2 trocados para a capilha certa (`005-001/0111/21738`: p. 54037 → 54050; `005-001/0151/30278`: p. 73185 → 73195) e nenhum acrescentado. Os 9.390 links restantes das capilhas não mudaram.
   - Os 4 erros da tabela acima estão entre os retirados. Os outros 72 retirados não foram conferidos um a um. Como a regra só retira, o pior caso é perder links que estavam certos, e não ganhar links errados.
2. **`ligar_duplicatas.py`.** Nos pares duplicados das séries 010 e 016/018 com o mesmo número de documento, a linha sem link recebe o link da outra, porque as duas descrevem o mesmo documento. Foram 48 links novos: 15 na 010 e 33 na 016. Os pares da série 017, que têm números diferentes, ficaram como estão.
3. O índice foi levado ao app com `aplicar_indice_resgate.py`. Eram 151.353 documentos com link, e agora são **151.325**.
4. O teste `test_capilha_so_liga_os_numeros_que_o_gemini_confirma`, em `tests/test_link_busca.py`, cobre a regra, os 4 erros e as duplicatas.

## Margem de erro da coleção inteira

A estimativa é estratificada pelos cinco grupos, com as populações acima (151.864 entradas no índice), juntando piloto e auditoria completa.

| | Taxa estimada | Limite superior de 95%, estratificado (Jeffreys, Monte Carlo) | Limite conservador (soma dos Clopper-Pearson de cada grupo) |
|---|---:|---:|---:|
| Antes da correção das capilhas | 0,029% (~44 links) | 0,23% (~346 links) | 0,49% (~745 links) |
| Depois, se a regra 6 eliminou o tipo de erro | ~0% | 0,20% (~302 links) | 0,45% (~679 links) |

**Leitura:** nenhum erro foi achado em 2.100 links fora das capilhas da Fase 2. O limite superior vem do tamanho da amostra, e não de erros observados. A linha "depois" supõe que os erros de E eram todos do tipo corrigido, como foram os 4 achados. Uma nova amostra de E confirmaria isso.

## O que ficou aberto

- **Sergipe:** confirmar a caixa dos documentos da caixa 2 que estão fora da faixa (`022/0002/00230`, `00273`, `00439` e outros até o doc. 466).
- **Pares duplicados da série 017:** 27 pares têm números de documento diferentes e não foram mexidos.
- **Entradas órfãs:** o índice tem 1.144 entradas cuja cota não existe no catálogo. Elas não aparecem no app.
- **Cobertura:** buscar os trechos da varredura que terminam muito antes do fim do rolo, o que pode explicar a cobertura de 86,4% em 005_BA_AV e de 87,9% em 013_PA.

## Arquivos

- No `andeca`, em `~/resgate_varredura/`: `amostra_auditoria.json`, `auditoria_resultados.jsonl/.csv`, `estado_auditoria.json` e `revisao_capilhas.json`, com a fonte de cada veredito. O piloto está em `auditoria_piloto/`. O painel tem a aba 5, "Auditoria dos links".
- No repositório: `coleta/scripts/juntar_capilhas.py` (regra 6 e `--refazer`), `coleta/scripts/ligar_duplicatas.py`, `coleta/scripts/aplicar_indice_resgate.py` e este arquivo.
