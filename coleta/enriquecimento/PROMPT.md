# Tarefa: extrair remetente, destinatário, pessoas, lugares e assunto de resumos do catálogo do AHU

Você recebe resumos (verbetes) do catálogo dos documentos avulsos do Conselho Ultramarino, do
Arquivo Histórico Ultramarino (Lisboa), sobre o Brasil colonial. Cada verbete vem numa linha, com
um número de ordem, assim: `[7] REQUERIMENTO de João da Silva, morador na vila de ..., ao rei
[D. João V], solicitando ...`.

Para cada verbete, extraia só o que **está escrito no resumo**:

- `rem`: quem escreve ou pede (o remetente). Num REQUERIMENTO, quem requer; numa CARTA ou num
  OFÍCIO, quem escreve; numa CONSULTA, o Conselho Ultramarino (ou o órgão que consulta).
- `dest`: a quem o documento é dirigido (o destinatário).
- `mencionados`: outras pessoas citadas pelo nome, que não são o remetente nem o destinatário.
- `lugares`: cidades, vilas, capitanias, rios, fortalezas e outros lugares citados.
- `assunto`: de 1 a 3 palavras-chave do assunto principal, em minúsculas, em português atual
  (ex.: `["sesmaria"]`, `["deserção", "soldados"]`, `["contrato", "dízimos"]`).

Para `rem` e `dest`, dê um objeto com:
- `nome`: o nome como aparece, sem o cargo (ex.: `Gomes Freire de Andrada`; `D. João V`;
  `Conselho Ultramarino`). Se só houver o cargo (ex.: "o governador do Rio de Janeiro"), deixe
  `nome` vazio e preencha `cargo`.
- `cargo`: cargo, título ou condição, como aparece (ex.: `governador e capitão-general da
  capitania de Minas Gerais`; `rei`; `viúva`; `morador na vila de Santos`). Vazio se não houver.
- `tipo`: `pessoa`, `instituicao` (Conselho Ultramarino, câmara, Mesa da Consciência…) ou
  `coletivo` (moradores, oficiais da câmara, religiosos de um convento…).

Regras:
1. Não invente. Se o resumo não diz, use `null` para `rem` ou `dest` e `[]` para as listas.
   O `assunto` é o único campo de interpretação: resuma o tema, sem julgar.
2. Mantenha a grafia do resumo: não modernize nomes nem complete abreviaturas. Tire só os
   colchetes (`[D. João V]` vira `D. João V`).
3. Não repita nos `mencionados` quem já é `rem` ou `dest`.
4. Responda **somente** com um array JSON, um objeto por verbete, na mesma ordem, com o número de
   ordem em `n`. Sem comentários, sem texto antes ou depois, sem cercas de código.

Exemplo de resposta para dois verbetes:

[{"n":1,"rem":{"nome":"Conselho Ultramarino","cargo":"","tipo":"instituicao"},"dest":{"nome":"D. Pedro II","cargo":"rei","tipo":"pessoa"},"mencionados":["Gomes Freire de Andrade"],"lugares":["Cabo do Norte","Caiena"],"assunto":["fortificação","fronteira"]},
{"n":2,"rem":{"nome":"","cargo":"oficiais da câmara de Salvador","tipo":"coletivo"},"dest":{"nome":"D. João V","cargo":"rei","tipo":"pessoa"},"mencionados":[],"lugares":["Salvador"],"assunto":["impostos"]}]

Verbetes:
