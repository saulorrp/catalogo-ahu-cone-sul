# Publicação da v3 núcleo (WP 2.5)

Este arquivo tem quatro coisas: a **medição** que o A10 cobra, a **decisão sobre o modelo de
embeddings**, o **comando de reindexação** e os **passos de publicação** — git e site do Streamlit.
É o registro do WP 2.5, para ser lido de cima para baixo.

> **Estado:** medição feita e decisão registrada. O merge de `v3` em `main` está feito
> **localmente**. O `push` e os passos no site do Streamlit **dependem do autor**: este
> repositório não tem `remote` configurado e a máquina não tem credencial do GitHub.

---

## 1. O que o A10 cobra

> **A10** — Interação típica (mudar um filtro) responde em < 1 s; uma consulta híbrida em < 2 s;
> carga fria em < 90 s, no Community Cloud.

E o teto de recursos do Streamlit Community Cloud gratuito, segundo a documentação (os números
estão datados de **fevereiro de 2024** e a própria página avisa que podem mudar sem aviso):

| Recurso | Mínimo | Máximo |
|---|---|---|
| CPU | 0,078 núcleo | 2 núcleos |
| Memória | 690 MiB | 2,7 GiB |
| Armazenamento | — | 50 GiB |

Duas coisas importam nessa tabela. A primeira: **o número que vale para planejar é o mínimo**,
690 MiB. O máximo de 2,7 GiB é o teto do que a plataforma *pode* dar, não o que ela garante, e os
relatos de "this app has gone over its resource limits" no fórum do Streamlit aparecem bem abaixo
de 2,7 GiB — há threads de 2026 supondo 1 GiB. A segunda: **0,078 núcleo é um oitavo de núcleo**,
de modo que um tempo medido numa máquina de verdade é um piso otimista, não uma previsão.

## 2. A medição

`python scripts/medir_memoria.py` mede, num processo novo e em etapas, o RSS (memória residente) e
o tempo de cada parte da carga fria. O script fica no repositório para a medida poder ser
repetida, e aceita `--modelo` para comparar modelos sem reindexar nada.

Medido em **08/10/2026**, na máquina do autor (Windows 11, Python 3.12, cache do Hugging Face
quente), nos 7.051 documentos:

| Etapa | e5-large | e5-base | e5-small |
|---|---:|---:|---:|
| 1. python + streamlit + pandas | 112 MiB | 112 MiB | 113 MiB |
| 2. artefatos de `app_dados/` | 203 MiB | 205 MiB | 205 MiB |
| 3. primeira busca léxica (sem modelo) | 205 MiB | 207 MiB | 207 MiB |
| 4. `torch` + `sentence_transformers` importados | 517 MiB | 519 MiB | 520 MiB |
| 5. modelo carregado | 658 MiB | 872 MiB | 625 MiB |
| 6. **primeira consulta codificada (pico)** | **1.771 MiB** | **1.204 MiB** | **717 MiB** |

O que esses números dizem:

- **o e5-large não cabe.** O pico de 1,77 GiB é 2,6× o mínimo garantido de 690 MiB, e fica acima
  do patamar em que os apps do Community Cloud começam a ser derrubados. A resposta à pergunta do
  WP 2.5 é **não**;
- **o e5-base também não cabe**: 1,20 GiB, ainda 1,7× o mínimo;
- **o e5-small cabe, apertado**: 717 MiB de pico — na vizinhança do mínimo garantido, e com folga
  grande até o máximo documentado;
- **a etapa 5 engana e a etapa 6 é a que vale.** Na etapa 5 o e5-small (625 MiB) aparece *abaixo*
  do e5-large (658 MiB), o que é impossível pelo tamanho dos pesos: é o carregamento preguiçoso do
  `sentence-transformers`, que só materializa os tensores na primeira passagem. É por isso que a
  linha que decide é a 6, o pico de uma consulta de verdade;
- **o custo maior não é o modelo, é o `torch`**: só importar `torch` e `sentence_transformers`
  custa ~310 MiB, mais que o peso do e5-small (~105 MiB na etapa 5). Trocar de modelo resolve o
  estouro, mas não deixa o app leve;
- **abrir o app e buscar por cota custa 205 MiB e ~7 s**, porque o modelo semântico só carrega na
  primeira consulta que precisa dele (decisão do WP 2.3). É a mesma razão pela qual a carga fria
  medida não é a carga fria *percebida*: quem abre o app e busca uma cota não paga o modelo.

Sobre o tempo: a carga fria completa (até a primeira consulta semântica) ficou em 92,6 s com o
e5-large e 108 s com os outros dois — mas nestes dois o número inclui o **download** do modelo, que
na nuvem acontece uma vez e fica em cache. O que a medição permite afirmar com segurança é a
divisão: ~5 s de Python e Streamlit, ~1 s dos artefatos, ~60 s de importação do `torch` nesta
máquina. **O A10 (< 90 s) não está verificado em produção** e só poderá ser verificado depois do
primeiro deploy — está anotado como pendência abaixo.

## 3. Decisão: `multilingual-e5-small`

**Decidido em 08/10/2026: o app vai para o ar com `intfloat/multilingual-e5-small`.**

Razão: é o único dos três que cabe na memória do Community Cloud gratuito, e a escolha entre
"semântica pior" e "app que não sobe" não é uma escolha. O e5-small tem 384 dimensões contra 1.024
do large, o que também reduz `embeddings.f16.npy` de 14,1 MB para ~5,3 MB e encurta a indexação.

O que **não** foi decidido aqui, e fica para o **WP 4.2**, como o plano já previa:

- a calibração fina: as margens de rigor (`MARGENS_RIGOR`), o piso `COSSENO_MINIMO`, os pesos do
  RRF (`PESOS_RRF`) e o `LIMITE_DESTAQUE` do aviso de consulta distante foram medidos **no
  e5-large** e estão todos em constantes de `ahu/busca.py`, de propósito. Eles continuam
  funcionando, mas os números certos para o e5-small saem do conjunto de ouro, no WP 4.2;
- a medida do custo em qualidade: o nDCG@10 do e5-small contra o do e5-large é o WP 4.1 (A5);
- a pendência do aviso de consulta distante com `military desertion`, que segue no WP 4.2.

Também fica registrado o que **não** resolveria o problema: deixar o e5-large e carregar sob
demanda já é o que o app faz desde o WP 2.3 — o pico de 1,77 GiB é o pico *de uma consulta
semântica*, e basta uma para derrubar o app.

### Mudança de dependência que vem com a publicação

`requirements.txt` passou a pedir o `torch` só-CPU:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0+cpu
```

Sem isso, o `torch` do PyPI em Linux arrasta as bibliotecas CUDA da NVIDIA — alguns GB que o app
nunca usa, porque não há GPU no Community Cloud. É uma das causas conhecidas de deploy que não
termina. O sufixo `+cpu` só existe no índice do PyTorch, e por isso o índice extra e o pin andam
juntos.

## 4. Reindexação (na máquina do autor)

Um comando para reindexar, um para reconstruir os artefatos, um para conferir:

```
python build/construir_indice.py --modelo intfloat/multilingual-e5-small --desativar-outras
python build/construir_app.py
python scripts/gate.py
```

Notas:

- **`--desativar-outras` é novo e é necessário.** A camada nova entra como
  `embeddings_e5small_v1`; sem a opção, o manifesto fica com *duas* camadas de embeddings ativas e
  o `construir_app.py` para com erro, porque quem decide qual camada vale é o manifesto, não o
  código (spec §3.1). A opção faz a troca e imprime o que desativou;
- **tempo:** o e5-large levou 97,6 min nos 7.051 documentos. O e5-small é cerca de oito vezes mais
  leve, então a ordem de grandeza é de **10 a 20 min**; o script imprime o tempo real e tem
  retomada, de modo que interromper não perde o que já foi codificado;
- a camada antiga (`embeddings_e5_v1`) **não é apagada**: ela fica no disco, inativa, e voltar
  atrás é marcar `"ativa": true` nela e `false` na nova, no `corpus/manifest.json`, e rodar o
  `construir_app.py` de novo;
- depois disso, `app_dados/manifest.json` passa a declarar o modelo novo, e a aba Sobre mostra
  isso sozinha — ela lê o manifesto, não o código.

## 5. Publicação

### 5.1 O que já está feito

- `main` criado a partir de `v3`, no mesmo commit (não houve divergência, então o "merge" é um
  *fast-forward* puro);
- `app_dados/` (24 MB) e `corpus/` (8 MB) estão versionados, e são o que o app lê: não há nada
  para subir à parte, e nenhum segredo a configurar. O repositório inteiro versiona ~32 MB;
- `requirements.txt` com o `torch` só-CPU.

### 5.2 Uma decisão sua antes dos comandos

Este repositório do app v3 é um repositório git **novo**, com história própria (um punhado de
commits, de `56d19d5` para cá). Ele **não** é uma ramificação de `saulorrp/catalogo-ahu-cone-sul`, que é onde vive o app antigo. Então
há dois caminhos, e eles não dão o mesmo trabalho:

| | **A — repositório novo** (recomendado) | **B — dentro do repositório antigo** |
|---|---|---|
| O que você faz | cria `saulorrp/catalogo-ahu-sul-v3` vazio no GitHub e dá push | dá push desta história sobre o `main` de `catalogo-ahu-cone-sul` |
| Risco | nenhum: o app antigo continua intacto | a história do repositório antigo é substituída (`--force`, com histórias não relacionadas) |
| O app antigo | continua no ar até você apagá-lo | é substituído no mesmo push |
| URL | precisa de subdomínio novo (e dá para renomear depois) | o app atual passa a servir o código novo |

Como o app público ainda não tem usuários, **o caminho A é o mais simples e o único sem risco**. Os
comandos abaixo são do caminho A; o caminho B está no fim, para o caso de você preferir.

### 5.3 Caminho A — repositório novo e app novo

**No GitHub, no navegador:**

1. <https://github.com/new>. Nome: `catalogo-ahu-sul-v3`. Visibilidade: **público** — o Community
   Cloud também implanta de repositório privado, mas o gratuito só permite **um** app privado, e
   aqui não há nada a esconder (o código é MIT e os dados são abertos). **Não** marque README,
   `.gitignore` nem licença: o repositório local já tem os três, e um commit inicial do GitHub só
   criaria conflito de histórias.
2. Clique em *Create repository* e deixe a página aberta: ela mostra a URL que o passo seguinte usa.

**No terminal, nesta pasta (`app v3`):**

```
git remote add origin https://github.com/saulorrp/catalogo-ahu-sul-v3.git
git push -u origin main
git push origin v3
```

O `push` vai pedir usuário e senha: o GitHub não aceita mais a senha da conta, então use um
**token** (Settings → Developer settings → Personal access tokens → Fine-grained, com permissão
*Contents: Read and write* neste repositório) como senha. Se o Git Credential Manager do Windows
estiver instalado, ele abre uma janela do navegador e resolve isso sozinho.

**No site do Streamlit (<https://share.streamlit.io>):**

1. Entre com a conta do GitHub e autorize o acesso ao repositório novo, se ele pedir.
2. *Create app* e escolha a opção de implantar um app público a partir do GitHub.
3. Preencha:
   - **Repository:** `saulorrp/catalogo-ahu-sul-v3`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** um subdomínio livre, por exemplo `catalogo-ahu-sul-v3` (6 a 63 caracteres).
     O subdomínio `catalogo-ahu-sul` está ocupado pelo app antigo e só libera quando ele for
     apagado.
4. Em **Advanced settings**, escolha **Python 3.11** (o padrão é 3.12). O `runtime.txt` do
   repositório **não** é lido pelo Community Cloud — a versão do Python vem desta caixa. O campo
   *Secrets* fica vazio: o app não usa nenhuma chave.
5. *Deploy*. A primeira construção instala o `torch` e demora vários minutos; acompanhe pelo painel
   de log e, se aparecer "error installing requirements", o log diz qual pacote.
6. Com o app no ar, confira na tela: a aba Busca (A3, uma cota e uma consulta conceitual), uma
   ficha (A1), a aba Pessoas com "Lavradio" (A2) e o tempo de resposta ao mudar um filtro (A10).

**Para trocar a URL depois** (se você quiser `catalogo-ahu-sul`): apague o app antigo primeiro —
em <https://share.streamlit.io>, menu ⋮ ao lado dele → *Delete*, digitando a frase de confirmação.
A documentação diz que o subdomínio fica disponível para reuso imediatamente. Em seguida, no app
novo, *Settings* → troque o subdomínio. A caixa de *Settings* muda **só** a
URL, o compartilhamento e os *secrets*; **repositório, branch e arquivo principal não se mudam por
ali** — para trocá-los é preciso apagar o app e criar outro, o que não custa nada e não perde dado,
porque o app não guarda estado.

### 5.4 Caminho B — substituir o repositório antigo

Só se você preferir manter a URL atual sem apagar nada no Streamlit. **Isto descarta a história do
`main` de `catalogo-ahu-cone-sul`**; faça antes uma cópia local dele, se quiser guardá-la.

```
git remote add origin https://github.com/saulorrp/catalogo-ahu-cone-sul.git
git push --force origin main
```

Depois, no painel do Streamlit, o app `catalogo-ahu-sul` já aponta para `main`, mas o arquivo
principal dele é o do app antigo. Se o caminho do arquivo mudou, o app precisa ser apagado e criado
de novo, do jeito do passo 5.3 — e nesse caso o caminho A era mais simples desde o começo.

### 5.5 O que fica pendente depois do deploy

- **A10 em produção:** medir, na nuvem, a resposta a uma troca de filtro (< 1 s), uma consulta
  híbrida (< 2 s) e a carga fria (< 90 s). A medição local diz que cabe na memória; o tempo com um
  oitavo de núcleo só se conhece lá;
- **A1, A2 e A3 em produção:** conferidos na tela, como no passo 6;
- se a memória estourar mesmo com o e5-small, o passo seguinte não é outro modelo: é tirar o
  `torch` do caminho (os ~310 MiB da importação), servindo o encoder de consulta por ONNX Runtime.
  Isso é maior que o WP 2.5 e deve entrar no WP 4.2.
