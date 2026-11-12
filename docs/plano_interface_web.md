# Plano: nova interface web (servidor próprio)

12/11/2026. Pedido do autor: com o app hospedado no agiulfo, as limitações do Streamlit Cloud
(690 MiB, rerun a cada clique, CSS por seletores internos, sem URL por documento) não valem mais.
Este plano é o contrato para quem implementa (o agy). Leia antes `AGENTS.md` e `docs/HANDOFF.md`.

## Decisão de stack
- **FastAPI + Jinja2 + HTMX**, renderizado no servidor, num pacote novo `web/`. Sem Node, sem
  etapa de build de JS. O `htmx.min.js` (2.0.x) vai **copiado** em `web/static/` (nada de CDN:
  o site deve abrir mesmo se um CDN cair). CSS próprio em `web/static/catalogo.css`.
- **Toda a lógica continua em `ahu/`** (busca, consulta, filtros, ficha, links, conversor,
  lista_ahu, selecao, pessoas, amostras, pdf). `web/` só traduz HTTP <-> essas funções.
  Se faltar alguma função pura, crie-a em `ahu/` com teste; não duplique lógica.
- O **Streamlit (`app.py`) continua** como está: é a versão leve do Streamlit Cloud e o plano B.
  Não apague nem quebre nada dele.
- Servidor: `uvicorn web.main:app`. Prefixo configurável por `AHU_ROOT_PATH` (no agiulfo o site
  fica em `/ahu/` atrás do Tailscale Funnel). **O Funnel NÃO remove o prefixo**: a requisição
  chega como `/ahu/doc/...`. Logo, com `AHU_ROOT_PATH=/ahu` o app deve de fato **servir sob /ahu**
  (montar as rotas e o `/static` sob o prefixo, p. ex. `APIRouter(prefix=...)` ou um app pai que
  faz `mount("/ahu", app)`), e todos os links usam `url_for`. Testar com TestClient pedindo `/ahu/`.
- Dependências novas, fixadas (R2): `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`.
  Para testes: `httpx` em requirements-dev.

## Regras que não mudam (AGENTS.md)
- `[sic]` vermelho escuro via `ahu.ficha.marcar_sic`; **nunca** corrigir o texto da fonte.
- `scribe_mediation_likely` não aparece em lugar nenhum (há teste; estenda-o a `web/`).
- IDs internos (`doc_uid`, `pessoa_uid`) **não aparecem como rótulo** na página. Podem estar em
  URL e atributos (`/doc/{doc_uid}`), como chave técnica, mas o texto visível é cota, ano, resumo.
- Biografias só ALTA/MÉDIA (já garantido em `app_dados/`).
- Links DigitArq e Resgate via `ahu.links.do_documento`; aba Conversor preservada.
- O app só lê `app_dados/`.
- Atribuição de inferências: dado do DeepSeek ou do Gemini aparece com o nome do modelo.

## Páginas (rotas)
| Rota | Conteúdo |
|---|---|
| `/` | Busca: caixa grande, filtros em barra lateral recolhível (gaveta no celular), resultados |
| `/busca?q=…&modo=…&rigor=…&ano_de=…&ano_ate=…&remetente=…&destinatario=…&pagina=…` | mesma página; **a URL é o estado** (usar `Consulta.para_url`/`de_url`) |
| `/resultados?…` | fragmento HTMX só com a lista (atualiza sem recarregar ao mudar filtro) |
| `/doc/{doc_uid}` | **Ficha permanente e citável**: cota nas três grafias, ano, resumo com [sic] e termos marcados, remetente/destinatário, pessoas com link, DigitArq, Resgate, "como citar" (copiar), "reportar erro" (issue), botão "pôr na seleção" |
| `/pessoas`, `/pessoa/{pessoa_uid}` | lista com filtro e página da pessoa (documentos, biografia com confiança e fonte) |
| `/conversor` | conversor de cotas (uma ou em lote), com envio à lista do AHU |
| `/selecao` | seleção (cookie ou URL `?s=`, reaproveitar `Selecao.para_url`), PDF da lista de consulta no AHU, CSV |
| `/amostras` | amostra diacrônica e estratificada (paridade com a aba atual) |
| `/painel` | gráficos (plotly gerado no servidor como JSON + plotly.min.js local, ou SVG simples) |
| `/sobre`, `/referencias` | textos atuais da aba Sobre e Referências (mover o texto para `web/templates/`, mesma redação) |
| `/api/busca`, `/api/doc/{doc_uid}` | JSON (consultor item 8: API simples). Mesmos parâmetros da busca |
| `/saude` | 200 + versão dos dados (para o monitor do servidor) |

Busca enriquecida (DeepSeek) deixa de ser uma "seção" separada: os filtros enriquecidos ficam num
grupo "Inferências (DeepSeek)" fechado por padrão, com o "saiba mais" atual.

## Estado depois do redesenho (PR #28, 12/11/2026)

O plano abaixo foi feito pelo Gemini (PR #20) e revisto no PR #28. Onde o código diverge do plano,
vale o código:

- **Navegação**: abas no topo, como as de um navegador, no lugar da coluna lateral. A aba Seleção
  mostra quantos documentos há nela. Referências virou uma subaba de Sobre.
- **Seleção**: gaveta própria ao lado dos resultados da busca (`_gaveta_selecao.html`), botão
  "Pôr na seleção" / "Na seleção" em cada resultado e na ficha. Guardada no cookie `selecao`
  (o mesmo espelho curto de `Selecao.para_url`); passado o teto, o documento novo não entra e a
  interface avisa. Sem CSV (D7).
- **Período**: o histograma grande saiu de cima da lista. Ficou uma miniatura por década dentro
  do filtro de período, que filtra pela década clicada. A linha do tempo do catálogo inteiro,
  legível e com os números, está na aba Sobre.
- **Busca em dois tempos** (`web/cache.py`): a fusão responde primeiro e o reranker reordena num
  segundo pedido; respostas e notas de pares ficam em memória. Ver `docs/plano_busca_servidor.md`.
- **Visual**: acento único bordô `#800000` nos dois temas, tokens em `web/static/catalogo.css`.
- Pendentes do plano: paginação numerada (há "Carregar mais"), ordenação por cota, `j`/`k`, posição
  clicada no registro de buscas, capturas em `docs/capturas/web_*.png`.

## Melhorias pedidas (parecer do consultor, 12/11/2026)
1. **Filtro por período** (ano de / até, com "incluir sem data") sempre visível, com histograma
   de anos dos resultados acima da lista (clicar numa barra filtra a década).
2. **Filtro por remetente e destinatário** com autocompletar (`/api/sugerir?campo=remetente&q=`),
   usando as colunas que existirem (`sender_name`, `recipient_name`; e, quando a camada
   `pessoas_gemini_v1` chegar a `app_dados/`, cargo/tipo do Gemini). Código tolerante à ausência
   da coluna: o filtro some se o dado não existe.
3. **Facetas com contagem** (coleção, capitania/pasta, tipologia, década) calculadas sobre o
   resultado atual, clicáveis.
4. **Registro anônimo de buscas** (consultor item 4): só texto, data, nº de resultados e posição
   clicada, num SQLite em `AHU_LOG_DIR` (padrão: desligado). Sem IP, sem cookie de rastreio.
5. Ordenação: relevância, ano crescente/decrescente, cota.
6. Paginação real (25 por página, links numerados) e "carregar mais" via HTMX.

## Visual
- Identidade atual (docs/relatorio_identidade_visual_eins.md): bordô `#800000`, papel `#f5f2eb`,
  Noto Serif no resumo, sem emoji, fleurões ❧/❦ nas gavetas. Tema escuro por
  `prefers-color-scheme` (paleta do `.streamlit/config.toml`).
- Leitura confortável: coluna de resultados ~70ch, cota em fonte tabular, ano em destaque.
- Responsivo a 360 px. Acessível: rótulos em todos os campos, foco visível, navegação por teclado
  (`/` foca a busca; `j`/`k` entre resultados é opcional).
- Página de documento com `<title>` = cota + ano e metadados `citation_*` (Zotero/Google Scholar
  capturam a ficha).

## Desempenho
- Carregar `Dados` uma vez no início (lifespan do FastAPI); encoder semântico carregado preguiçoso
  como hoje (`encoder_preguicoso` em app.py: extrair para `ahu/` se preciso). Respeitar
  `AHU_SEM_SEMANTICA`. Reranker (PR da frente de hospedagem, pode ainda não estar no main):
  `ahu/reranker.py`, `ligado()` lê `AHU_RERANKER`, `criar_reranker()` devolve
  `reranker(consulta, textos) -> notas`; criar **um por processo** (cache) e passar a
  `consulta.executar(..., reranker=rr)`. Importar com `try/except ImportError` e só passar o
  argumento se o módulo existir, para funcionar antes e depois do merge.
- Meta: página de busca < 1 s no agiulfo para consulta léxica.

## Container
- `Dockerfile`: o mesmo, com `CMD` escolhido por `AHU_FRONT` (`web` padrão -> uvicorn na 7860;
  `streamlit` -> o comando atual). Script `deploy/iniciar.sh`.
- Não mexer no Funnel nem no docker-compose do agiulfo além do necessário (outra frente cuida).

## Testes (obrigatório; `python scripts/gate.py` deve dar PASS)
- `tests/test_web.py` com `fastapi.testclient` sobre o mini catálogo de `tests/fixtures`
  (ver `tests/conftest.py`): cada rota responde 200; busca por texto e por cota; filtros por ano e
  remetente reduzem o total; `/doc/{uid}` mostra [sic] e o link DigitArq; nenhum `doc_uid` como
  texto visível; nenhuma menção a `scribe_mediation_likely`; `AHU_ROOT_PATH=/ahu` gera links com
  prefixo; `/api/busca` devolve JSON com cota e ano.
- Capturas de tela (Playwright, claro e escuro, 1280 px e 390 px) da busca e da ficha em
  `docs/capturas/web_*.png`.

## Fora do escopo
Mudar a busca em si (pesos, reranker, julgamentos): outras frentes. Mapa Wikidata (WP 5.2).
