# HANDOFF — Classificador AHU-Sul v3

**Leia este arquivo primeiro.** Ele existe para que outro agente de código — sem acesso à conversa
que produziu este repositório — consiga continuar o trabalho só com o que está aqui dentro.

Última atualização: **09/10/2026**, commit `0b0513e`.

> **Regra de manutenção:** este arquivo é atualizado **no mesmo commit** da mudança que ele
> descreve, nunca num commit à parte. Um HANDOFF desatualizado é pior que nenhum.

---

## 1. Estado atual, em um parágrafo

O app funciona de ponta a ponta no `localhost` e **não está publicado**: o repositório não tem
`remote` configurado, e o que está no ar em `catalogo-ahu-sul.streamlit.app` ainda é o app **v2**.
O repositório **está pronto para substituí-lo** — veja [`publicacao.md`](publicacao.md), que tem os
comandos exatos; o autor pode publicar a qualquer momento. O corpus modular está pronto (7.051
documentos, 1.359 pessoas, índice semântico `multilingual-e5-small` de 384 dimensões), a busca
híbrida funciona, a aba Pessoas e o painel diatópico funcionam, e o único arquivo que o app
entrega é o PDF da **lista para consulta no AHU**, mais o plano de coleta do EINS na aba Amostras.
O que **bloqueia o deploy** é o R7: em tempo de execução o app importa `torch`, e o pico de memória
é de ~948 MiB contra os 690 MiB que o Streamlit Community Cloud garante.

---

## 2. Registro do que foi feito

Datas em 2026. "Verificado por" nomeia o teste ou a medição, não a impressão.

| Data | Commit | O que mudou | Como foi verificado |
|---|---|---|---|
| 07/10 | `56d19d5` | F0 e WP 1.1: estrutura, licenças, base com `doc_uid` | `tests/test_importar_base.py`; 7.051 uids únicos |
| 08/10 | `55334ad` | WP 1.2: camadas tipologia e sociolinguística | `tests/test_importar_camadas.py`; 86 tipologias, SRSP 1–9 |
| 08/10 | `58d6d34` | WP 1.3: entidades e biografias v2 | `tests/test_importar_biografias_v2.py`; 253 nascimentos exibíveis |
| 08/10 | `f27249f` | WP 1.4: índice semântico reprodutível | sanidade 5/5 no `construir_indice` |
| 08/10 | `1ee2ba6` | WP 1.5: artefatos do app; **F1 completa** | A4 e A11 em `tests/test_construir_app.py` |
| 08/10 | `fed0056` | WP 2.1: rota de cotas, BM25 por campos, semântico, RRF | **A3** em `tests/test_busca.py` |
| 08/10 | `ef6dbe7` | WP 2.2: `Consulta` e filtros da §6.1 | **A6** (ida e volta) em `tests/test_consulta_filtros.py` |
| 08/10 | `a31b475` | WP 2.3: interface de busca e ficha | **A1** nas 7.051 fichas, `tests/test_ficha_app.py` |
| 08/10 | `6ff0bad` | WP 2.4: aba Pessoas, coletivos, painel diatópico | **A2** em `tests/test_pessoas.py` |
| 08/10 | `2045d0c` | WP 2.5: medição de memória; decisão do e5-small | `scripts/medir_memoria.py` |
| 09/10 | `f6929b9` | Reíndice com e5-small (feito pelo autor) | 7.051 docs, 8,3 min, sanidade 5/5 |
| 09/10 | `766047e` | WP 3.1: seleção que atravessa buscas | `tests/test_selecao.py` |
| 09/10 | `06955e3` | WP 3.2: exportação CSV e JSON | **retirado depois pela D7** |
| 09/10 | `269d0d7` | WP 3.4: link da busca | **A6** com cinco buscas, `tests/test_link_busca.py` |
| 09/10 | `28769da` | WP 3.3a: lista de consulta no AHU (PDF) | `tests/test_lista_ahu.py`; 7.050/7.051 cotas com CRAV |
| 09/10 | `eec56b7` | WP 3.3: dossiê em PDF | **retirado depois pela D7** |
| 09/10 | `7dd2075` | WP 3.5: amostras e plano EINS; **F3 completa** | **A7** e **A9** em `tests/test_amostras.py` |
| 09/10 | `df0afc2` | Correção da medida de memória | e5-small: 948 MiB, não 717 |
| 09/10 | `c23cefe` | Plano de remodelação v3.1 | — |
| 09/10 | `3c31ecd` | **R1**: remover lentes, exportações, identificadores | `tests/test_remocoes_r1.py` (16 testes) |
| 09/10 | `fe2f988` | **R2**: seleção rápida | 0 buscas por clique; fragmento em ~46 ms |
| 09/10 | `aa3f822`, `99aaf93` | **R3**: PDF da lista do AHU, refeito | `tests/test_lista_ahu.py` |
| 09/10 | `0b0513e` | Fleurão como interruptor (item 2 do R6) | `tests/test_estilo.py` (12 testes) |
| 09/10 | `e5cb70f` | `docs/HANDOFF.md` | — |
| 09/10 | `c135d5c` | **R9, parte 1**: Sobre sem "Como citar"; tabelas pequenas centradas; busca de nome tolerante; link da fonte | `tests/test_pessoas.py`: "domingo de britto" acha "Domingos de Brito Peixoto" |

| 09/10 | *(este)* | **R7**: consulta por ONNX int8, sem `torch` | `tests/test_consulta_onnx.py` (8 testes); pico 948 → **630 MiB** |

**Medidas que valem hoje** (máquina do autor, Windows 11, 8 núcleos, cache do HF quente):

- pico de RSS numa consulta semântica: **630 MiB** (era 948 com `torch`);
- abrir o app e buscar por cota: **229 MiB**, ~4 s — o modelo só carrega quando é preciso;
- gate completo: ~4 min, **436 testes**, 1 pulado.

---

## 3. O que vem a seguir

Ordem decidida pelo autor em 09/10. Cada etapa tem de deixar o `main` **publicável**.

### Em curso nesta sessão

- [x] **R9, parte 1**: aba Sobre sem "Como citar" e com o preprint marcado como provisório;
      tabelas pequenas com `width="content"` e centradas; busca de nome tolerante a grafia e a
      prefixo; coluna "Fonte" como `LinkColumn`.
- [x] **R9, parte 2**: tabela dos lotes com layout fixo (Incluir · Nº · Ano · Código de
      referência clicável · Pasta · Tipologia · Resumo), escolher documentos, reordenar com
      ↑/↓, mover entre lotes, PDF por lote, e o "Download as CSV" das tabelas escondido.
- [x] **Preparar a substituição no GitHub.** Auditoria feita sobre toda a história: **nenhum
      segredo** (nenhum `.env`, `*api_keys*`, `pplx-`, `sk-`, `hf_`, `ghp_`); maior objeto que já
      existiu: 13,8 MB; `app_dados/*` versionado. `main == v3`. README e
      [`publicacao.md`](publicacao.md) reescritos para o caminho do *force-push*.
- [x] **R7 — `torch` fora do caminho de consulta.** O app não importa `torch` nem
      `sentence_transformers` em execução (provado num processo próprio). Pico: **630 MiB**, de
      948. **O alvo de 600 MiB não foi alcançado** e o critério de "top-10 idêntico" não é
      cumprido — ver os riscos 1 e 2.

### Pausado, retomar depois (plano em [`plano_remodelacao_v3_1.md`](plano_remodelacao_v3_1.md))

- [ ] **R4** — rótulos e texto: traduzir no `build/construir_app.py` os valores em inglês do
      DeepSeek (`King of Portugal` → `Rei`, `Overseas Council` → `Conselho Ultramarino`) por uma
      tabela revisada `build/rotulos_pt.json`; tirar parênteses redundantes; ícone da aba sem
      emoji. **Aceite:** um teste falha se um rótulo em inglês chegar à interface.
- [ ] **R5** — "saiba mais" (`st.popover`) no lugar da linha de auditoria do SRSP, com a ressalva
      do preprint §2 citada **na íntegra** (o texto está no plano v3.1, R5).
- [ ] **R6, o que falta** — bordô como acento único em `.streamlit/config.toml` (temas claro e
      escuro separados; o Streamlit 1.55 tem `theme.light.*` e `theme.dark.*`), e a tipografia
      (monoespaçada na tela, serifada no documento). O item 2 (glifos) já está feito.
- [ ] **R8** — limpeza: `use_container_width` → `width=`, fast-forward do `main`, relatório final.
- [ ] **F4** (spec §11, plano F4): conjunto de ouro e calibração **no e5-small** (os limiares de
      `ahu/busca.py` foram medidos no e5-large), CI no GitHub Actions, Zenodo com DOI.
- [ ] **WP 0.3** — busca v2 de referência, pendente desde o início.
- [ ] **A10 em produção** — só se mede depois do primeiro deploy.

Documentos de referência: [`plano_remodelacao_v3_1.md`](plano_remodelacao_v3_1.md) (o que vale
agora), `../spec_catalogo_ahu_v3.md` (§0 decisões, §11 critérios de aceite),
`../plano_implementacao_v3.md` (as fases F0–F4), [`publicacao.md`](publicacao.md) (deploy).

---

## 4. Decisões em vigor

Uma linha cada. As D1–D6 estão na spec §0; as D7–D11 foram acrescentadas lá em 09/10.

| | Decisão |
|---|---|
| **D1** | `scribe_mediation_likely` só no dossiê em PDF — **revista pela D8**. |
| **D2** | Biografias exibidas e filtráveis: só **ALTA e MÉDIA**. |
| **D3** | Código sob **MIT**. |
| **D4** | O plano de coleta é importável pelo **EINS Lite**. |
| **D5** | Um pesquisador, sem verba: o conjunto de teste é julgado só pelo autor. |
| **D6** | O corpus é **único e modular**; pastas, coleções e tipologias vêm dos dados, nunca do código. |
| **D7** | A exportação é **uma coisa só**: o PDF da lista do AHU. Saíram CSV, JSON e o dossiê. **A8 aposentado.** |
| **D8** | `scribe_mediation_likely` **sai do documento**: fica no dado, e é explicado só no "saiba mais". |
| **D9** | **As lentes saíram.** `Consulta` subiu para a versão 2 da URL; um link antigo com `l=` ainda abre. |
| **D10** | **Nenhum identificador interno na interface.** Exceção: a *issue* de "Reportar erro". |
| **D11** | A aba **Amostras e o plano do EINS ficam** (a D4 os exige); o "sem JSON" da D7 era sobre a seleção. |

**Decididas nesta sessão:**

| | Decisão | Por quê |
|---|---|---|
| **D14** | **O modelo de consulta é o ONNX int8, não o fp32.** | O fp32 passaria no critério estrito de equivalência, mas custa 960 MiB — pior que os 948 MiB do `torch` que o R7 veio substituir, e acima dos 690 MiB garantidos. O int8 custa 630 MiB e é a única configuração que cabe. O preço é a ordem exata do top-10; os mesmos documentos aparecem (90 %) e o primeiro colocado é o mesmo em 26 de 30 consultas. Para triagem isso basta, e os limiares de `ahu/busca.py` ainda serão recalibrados na F4. |
| **D13** | **Reordenar linhas é com ↑ / ↓, não arrastando com o rato.** | Arrastar exigiria um componente de terceiros com pacote JavaScript próprio (`streamlit-sortables` e afins): mais uma dependência a construir no Streamlit Cloud, justo quando o R7 corta dependências para caber na memória; e esses componentes reordenam **listas de texto**, não linhas de tabela — não dariam a tabela que o autor pediu. Os ↑/↓ agem sobre a linha escolhida por número. Reavaliar se o Streamlit trouxer reordenação nativa de `st.data_editor`. |
| **D12** | **Não haverá repositório novo.** O `v3` vai por `push --force` sobre o `main` de `saulorrp/catalogo-ahu-cone-sul`, reaproveitando o app `catalogo-ahu-sul.streamlit.app`. | Menos passos para o autor e a URL conhecida se mantém. O app antigo é preservado numa branch `v2-arquivo` **antes** do *force-push*. |

---

## 5. Como trabalhar neste repositório

```
python scripts/gate.py            # ruff + pytest. Tem de imprimir PASS antes de cada commit.
streamlit run app.py              # o app, em http://localhost:8501
python scripts/medir_memoria.py --com-app   # RSS e tempo da carga fria
```

Construção do corpus, **nesta ordem** (só quando os dados mudam):

```
python build/importar_base.py
python build/importar_camadas.py
python build/importar_biografias_v2.py
python build/construir_indice.py --modelo intfloat/multilingual-e5-small --desativar-outras
python build/construir_app.py
```

### Armadilhas desta máquina (Windows + Yandex.Disk)

Não são detalhes: cada uma já custou tempo.

1. **Pause a sincronização do Yandex.Disk antes de construir.** O repositório vive dentro da pasta
   sincronizada. Durante o `construir_indice` e o `construir_app` o Yandex abre os arquivos para
   enviar, e o build falha com erro de acesso negado, ou grava um `.parquet` pela metade.
2. **Feche o Streamlit antes de reconstruir.** O app mantém `app_dados/*.parquet` e
   `embeddings.f16.npy` abertos; com ele no ar o `construir_app.py` não consegue sobrescrevê-los.
3. **Um comando de cada vez.** Encadear com `&&` ou `;` no PowerShell desta máquina trava ou
   dispara pedido de permissão; rode um, espere, rode o próximo.
4. **Sempre `python -m`** (`python -m pytest`, `python -m streamlit run app.py`). Os executáveis
   soltos nem sempre estão no `PATH`.
5. **`fpdf` 1.7.2 (PyFPDF) está instalado ao lado do `fpdf2`.** Hoje o `fpdf2` ganha o conflito de
   nomes e os PDFs saem certos, mas uma reinstalação em outra ordem quebraria os dois sem aviso
   claro. Corrija com `pip uninstall -y fpdf`.
6. **Acentos no console.** Use `python -X utf8 …` quando a saída tiver acentos, senão o `cp1252`
   do console derruba o script.

### Regras do projeto que o código faz valer

Estão em [`../AGENTS.md`](../AGENTS.md). As que mais pegam quem chega:

- o app **só lê `app_dados/`**, nunca `corpus/` (há teste que varre o `app.py`);
- **nada de lógica de negócio no `app.py`**: ela vai para `ahu/`, que não importa Streamlit;
- **CSS num lugar só**, `ahu/estilo.py` (há teste);
- pastas, coleções e tipologias **vêm dos dados**, nunca escritas no código.

---

## 6. Riscos conhecidos e perguntas em aberto

1. **Memória: 630 MiB, e o alvo era 600.** O R7 trouxe o pico de 948 para 630 MiB, que **cabe**
   nos 690 MiB garantidos, mas não alcança os 600 que o plano pedia. O que sobra é quase todo
   peso do modelo int8; o `onnxruntime` em si custa 21 MiB, contra 312 do `torch`. Para baixar
   mais, o caminho é um modelo menor, não mais engenharia no carregamento.
2. **O int8 não dá a mesma ordem.** Cosseno ≥ 0,9940 e 90 % dos mesmos documentos no top-10, com
   o mesmo primeiro colocado em 26 de 30 consultas — mas a ordem exata muda sempre. O plano pedia
   "top-10 idêntico em ≥ 90 %", e **só o fp32 entrega**, ao custo de 960 MiB (pior que o ponto de
   partida). `AHU_ONNX_ARQUIVO=onnx/model.onnx` volta ao fp32.
2. **Os limiares da busca foram calibrados no e5-large** e o índice agora é e5-small:
   `MARGENS_RIGOR`, `COSSENO_MINIMO`, `PESOS_RRF` e `LIMITE_DESTAQUE` em `ahu/busca.py` estão
   provisórios até a F4. Eles funcionam, mas não estão medidos neste modelo.
3. **Um documento sem código de referência CRAV** (`ahu-sul:6bb06bc3e0`): a única cota dele é
   antiga e por região, sem número de série. Está contado no teste, não escondido.
4. **165 tipologias marcadas "a revisar"** e a coluna `decisao_manual` do
   `revisao_biografias.csv` ainda vazia nas 1.359 linhas — trabalho do autor, não do código.
5. **O aviso de consulta distante (§5.5)** cala quando um único termo existe no corpus, mesmo por
   acidente (`military desertion` escapa). Decidir no WP 4.2.
6. **Em aberto:** o plano v3.1 diz "código e documentação em inglês", mas o repositório inteiro é
   em português. Mantive o português; virar para inglês seria um pacote próprio.
7. **O `AppTest` do Streamlit não respeita o escopo de `st.fragment`**: o tempo de parede dele não
   mede um clique no navegador. O que se testa é a propriedade robusta (nenhuma chamada ao motor
   de busca), não o relógio.
