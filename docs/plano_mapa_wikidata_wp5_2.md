# WP 5.2 — Plano Técnico: Mapa das Cidades com Coordenadas da Wikidata (P625)

> **Documento de passagem (handoff):** Guia técnico para agentes de código implementarem a extensão
> WP 5.2 (spec §9 e §12: "Mapa das cidades de nascimento, com coordenadas da Wikidata").

---

## 1. Contexto e Objetivo

No **Painel Diatópico** (`app.py` / `ahu/pessoas.py`), o usuário pode analisar a distribuição geográfica
dos remetentes coloniais (por documento ou por pessoa). Hoje, existem métricas agregadas, gráfico de
barras por país e distribuição temporal por década. Na base do painel, há um aviso:
> *"Em breve: o mapa das cidades de nascimento, com coordenadas da Wikidata (P625, WP 5.2)."*

O objetivo do WP 5.2 é substituir esse aviso por um **mapa geográfico interativo de pontos/bolhas**,
posicionando as cidades de nascimento dos autores dos documentos do catálogo.

---

## 2. Restrições Arquiteturais Inegociáveis

1. **Custo Zero e Zero Chamadas de Rede em Runtime (Regra R1):**
   - O app em produção no Streamlit Community Cloud **NÃO** deve consultar a API da Wikidata durante
     a execução do usuário.
   - Toda a geocodificação deve ocorrer **offline**, na máquina do autor, durante a etapa de build.
2. **O App só lê `app_dados/` (Regra R3):**
   - As coordenadas resolvidas (`latitude` e `longitude`) devem ser armazenadas no arquivo
     `app_dados/pessoas.parquet` (ou tabela compacta auxiliar em `app_dados/`).
3. **Sem Novas Dependências Pesadas de Mapa:**
   - **Não** instalar Folium, GeoPandas ou Pydeck (que aumentariam o consumo de memória RAM além
     do teto de 690 MiB).
   - Utilizar o **Plotly Express** (`px.scatter_geo`), que já é dependência instalada e homologada
     no projeto (`ahu/graficos.py`).
4. **Filtro Biográfico Estrito (Decisão D2):**
   - O mapa só plota cidades de pessoas com confiança biográfica **ALTA ou MÉDIA** (ou com
     `decisao_manual = aceitar`), respeitando o mesmo filtro de `ahu/pessoas.py`.

---

## 3. Volume e Dados Existentes

- Pessoas exibíveis no corpus: **253 pessoas**.
- Total de cidades de nascimento distintas estimadas: **entre 40 e 80 cidades** (ex.: Lisboa,
  Salvador, Rio de Janeiro, Porto, Coimbra, Évora, Braga, São Paulo, Santos, Viana do Castelo, etc.).
- Como o número de cidades é muito pequeno, a resolução offline é imediata (< 10 segundos).

---

## 4. Cuidados Metodológicos e Históricos

1. **Evitar Falsos Centróides:**
   - Muitos remetentes têm apenas o país identificado (*"Portugal"* ou *"Brasil"*), sem vila/cidade
     natal conhecida.
   - **Regra:** Nunca atribuir coordenadas do centro geográfico do país a essas pessoas. Quem não tem
     cidade comprovada não entra como ponto no mapa (continua contabilizado nas barras por país).
2. **Toponímia Histórica vs. Moderna:**
   - Vilas coloniais (*ex.: Nossa Senhora do Desterro → Florianópolis; Vila Nova de Portimão → Portimão;
     Angra da Ilha Terceira → Angra do Heroísmo*).
   - O identificador de entidade da Wikidata (QID) unifica a toponímia histórica e moderna, fornecendo
     a coordenada exata através da propriedade **P625** (*coordinate location*).
3. **Escala e Enquadramento da Câmera:**
   - O fluxo documental do Cone Sul conecta a América Portuguesa à bacia do Atlântico (Portugal,
     Açores, Madeira, Angola, bacia do Prata).
   - A projeção padrão do mapa deve ser centrada no **Atlântico**, permitindo visualização simultânea
     do Brasil e da Península Ibérica.

---

## 5. Roteiro Passo a Passo de Implementação

### Passo 1: Script de Geocodificação Offline (`build/geocodificar_cidades.py`)
- Ler `corpus/camadas/biografias_v2/` ou `biografias_validadas.json`.
- Para cada cidade única vinculada a uma pessoa com confiança ALTA/MÉDIA:
  - Se a pessoa tem `url_fonte` da Wikipédia ou `qid` da Wikidata, consultar via `wbgetentities`:
    `https://www.wikidata.org/w/api.php?action=wbgetentities&ids={QID}&props=claims&format=json`
  - Extrair declaração `P625` -> `latitude` e `longitude`.
  - Salvar cache em `corpus/camadas/biografias_v2/coordenadas_cidades.json`.

### Passo 2: Propagação nos Artefatos de `app_dados/`
- Em `build/construir_app.py`:
  - Mesclar `latitude` e `longitude` no DataFrame de `pessoas.parquet`.
  - Reconstruir os artefatos com `python build/construir_app.py`.

### Passo 3: Criação da Função do Mapa em `ahu/graficos.py`
- Adicionar função `mapa_cidades(painel: PainelDiatopico, tema: str) -> go.Figure`:
  - Agregar contagem por cidade (respeitando o modo `por = "documento"` ou `"pessoa"`).
  - Configurar `px.scatter_geo` com marcadores proporcionais ao volume documental, coloridos pelo
    acento bordô (`#800000` / `#a83030`) ou paleta de países validada no R6.
  - Hover contendo: Nome da Cidade, País, Nº de Documentos / Pessoas.

### Passo 4: Interface em `app.py`
- Na função `aba_painel(dados)`:
  - Substituir `st.info("Em breve: o mapa...")` pela renderização do gráfico:
    ```python
    fig_mapa = mod_graficos.mapa_cidades(painel, tema())
    st.plotly_chart(fig_mapa, width="stretch")
    ```

### Passo 5: Testes Automatizados
- Adicionar testes em `tests/test_pessoas.py` e `tests/test_graficos.py` validando que:
  - O mapa renderiza sem erros com e sem dados;
  - Nenhuma coordenada nula/inválida causa exceção.
