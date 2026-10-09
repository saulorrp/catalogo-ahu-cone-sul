# Publicação — substituir o app que está no ar

Os passos exatos para pôr a **v3** no lugar da v2, reaproveitando o repositório e o app que já
existem. A medição de memória e a decisão do modelo estão na §7.

> **Decisão do autor (D12), 09/10/2026: não haverá repositório novo.** O `v3` vai por
> `push --force` sobre o `main` de `saulorrp/catalogo-ahu-cone-sul`, e o app
> `catalogo-ahu-sul.streamlit.app` se reimplanta sozinho a partir do `main`. O app antigo é
> **preservado numa branch** antes disso, e nada se perde.

---

## 1. O que já está pronto no repositório

- `main` e `v3` apontam para o mesmo commit;
- **auditoria de segredos feita sobre toda a história do git** (não só o `HEAD`): nenhuma chave,
  nenhum `.env`, nenhum `*api_keys*.txt`, nenhum token. O maior objeto que já existiu na história
  tem 13,8 MB, muito abaixo do limite de 100 MB do GitHub;
- `app_dados/` está versionado (7 arquivos, 23,7 MB): é o que o app lê, e não há nada para subir
  à parte;
- `requirements.txt` instala o `torch` só-CPU.

## 2. No GitHub, pelo navegador: guardar o app antigo

**Faça isto primeiro.** Depois do `push --force` o `main` antigo some, e sem este passo ele se
perde.

1. Abra <https://github.com/saulorrp/catalogo-ahu-cone-sul>.
2. No seletor de branch (diz **main**), clique, escreva **`v2-arquivo`** na caixa e clique em
   **"Create branch: v2-arquivo from main"**.
3. Confira que a branch nova aparece na lista e tem os arquivos do app antigo.

## 3. No seu computador: enviar a v3

Um comando de cada vez, na pasta `app v3`:

```
git remote add origin https://github.com/saulorrp/catalogo-ahu-cone-sul.git
```

```
git push --force origin main
```

**Na primeira vez abre uma janela de login do GitHub** (o Git Credential Manager do Windows). Entre
com a sua conta; ele guarda a credencial e não pergunta mais. Se em vez da janela ele pedir usuário
e senha no terminal, a senha **não** é a da conta: é um *token* (GitHub → Settings → Developer
settings → Personal access tokens → Fine-grained, com permissão *Contents: Read and write* neste
repositório).

O `--force` é necessário **só desta vez**: as duas histórias não se relacionam, e o Git recusaria o
envio sem ele.

Confira em <https://github.com/saulorrp/catalogo-ahu-cone-sul> que o `main` agora mostra o
`README.md` da v3, e que `v2-arquivo` continua lá.

## 4. No Streamlit Cloud

O app `catalogo-ahu-sul.streamlit.app` segue o `main`, então **ele se reimplanta sozinho** poucos
minutos depois do push. Acompanhe em <https://share.streamlit.io>.

### Ver em que Python ele está

No painel, abra o app → menu **⋮** → **Settings**. A versão do Python aparece ali. Ela foi fixada
quando o app foi criado e **não se muda depois**.

### Se a construção falhar por causa da versão do Python

Nos registros aparecerá algo como *"error installing requirements"* mencionando `torch`, `numpy` ou
uma roda indisponível. Nesse caso, recrie o app — não se perde nada, porque ele não guarda estado:

1. no painel, no app atual: **⋮ → Delete**, digitando a frase de confirmação;
2. **Create app** → implantar app público a partir do GitHub;
3. **Repository:** `saulorrp/catalogo-ahu-cone-sul` · **Branch:** `main` ·
   **Main file path:** `app.py`;
4. **App URL:** `catalogo-ahu-sul` — o subdomínio fica livre assim que o app antigo é apagado;
5. em **Advanced settings**, escolha **Python 3.11**. O `runtime.txt` do repositório **não** é lido
   pelo Community Cloud: a versão vem desta caixa;
6. *Deploy*. A primeira construção demora vários minutos.

## 5. Conferir depois do deploy

Cinco coisas, na ordem:

1. **cota** — busque `AHU_ACL_CU_017, Cx. 5, D. 487` no modo *Exata*: um documento, encontrado
   pela cota;
2. **pessoa** — aba Pessoas, procure `domingo de britto`: tem de achar *Domingos de Brito
   Peixoto*;
3. **PDF** — marque três documentos na Busca e gere o PDF da seleção: A4 paisagem, acentos certos,
   links "abrir" funcionando;
4. **semântica** — busque `deserção de soldados` no modo *Conceitual*. **É a primeira consulta
   semântica que carrega o modelo**: ela demora, as seguintes não;
5. **A10** — depois disso, mude um filtro e veja se a página responde em menos de um segundo.

## 6. Atualizações seguintes

Depois da primeira vez, publicar é commitar e enviar:

```
git push origin main
```

**Sem `--force`.** O Streamlit reimplanta sozinho.

Se o app tiver sido criado com o `torch` no ambiente e você acabar de publicar a versão sem ele (o
R7), o ambiente antigo pode ficar em cache. Nesse caso: painel do Streamlit → o app → **⋮ → Reboot
app**. Isso recria o ambiente a partir do `requirements.txt` novo, sem apagar o app nem mudar a URL.

---

## 7. Memória: por que o modelo é o `e5-small`

### O teto da plataforma

Segundo a documentação do Streamlit Community Cloud (números datados de **fevereiro de 2024**, e a
própria página avisa que podem mudar):

| Recurso | Mínimo | Máximo |
|---|---|---|
| CPU | 0,078 núcleo | 2 núcleos |
| Memória | 690 MiB | 2,7 GiB |
| Armazenamento | — | 50 GiB |

O número que vale para planejar é o **mínimo**: o máximo é o teto do que a plataforma *pode* dar, e
os relatos de "this app has gone over its resource limits" aparecem bem abaixo dele.

### A medida

`python scripts/medir_memoria.py --com-app` mede, num processo novo e em etapas, o RSS e o tempo da
carga fria. **Meça com o cache do Hugging Face quente**: numa máquina que acabou de baixar o modelo
o pico sai baixo demais, porque os pesos ainda não estão todos residentes.

Medido em 09/10/2026, na máquina do autor, nos 7.051 documentos:

| Etapa | e5-small | e5-base | e5-large |
|---|---:|---:|---:|
| 1. python + streamlit + pandas | 113 MiB | — | — |
| 2. artefatos de `app_dados/` | 203 MiB | — | — |
| 2b. módulos do app (plotly, fpdf2) | 226 MiB | — | — |
| 3. primeira busca léxica (sem modelo) | 229 MiB | — | — |
| 4. `torch` + `sentence_transformers` | 541 MiB | — | — |
| 5. modelo carregado | 862 MiB | 880 MiB | 881 MiB |
| 6. **primeira consulta codificada (pico)** | **948 MiB** | **1.211 MiB** | **2.013 MiB** |

O custo maior não era o modelo — era o `torch`, que pesa ~312 MiB antes de qualquer peso.

### Depois do R7: 630 MiB

Desde o R7 a consulta é codificada por `onnxruntime` + `tokenizers`, e o `torch` não é importado
em execução. As três configurações medidas, com o cache quente:

| caminho | pico de RSS | equivalência com o `sentence-transformers` |
|---|---:|---|
| `torch` + `sentence-transformers` (antes) | 948 MiB | é a referência |
| ONNX fp32 | 960 MiB | cosseno ≥ 0,99; top-10 idêntico |
| **ONNX int8 (o que vai ao ar)** | **630 MiB** | cosseno ≥ 0,9940; 90 % dos mesmos documentos no top-10; mesmo primeiro colocado em 26 de 30 |

**630 MiB cabe nos 690 MiB garantidos** — é a primeira configuração que cabe. O fp32 piora o que
veio consertar, e por isso não é o padrão; `AHU_ONNX_ARQUIVO=onnx/model.onnx` volta a ele.

O modelo **não vem no repositório** (120 MB, contra os 100 MB que o GitHub aceita): é baixado do
Hugging Face na primeira consulta conceitual e fica em cache. No Streamlit Cloud isso acontece uma
vez por ambiente, e é por isso que **a primeira busca conceitual depois de um deploy demora**.

Abrir o app e buscar por cota custa 229 MiB e ~4 s, porque o modelo semântico só carrega na
primeira consulta que precisa dele.

### Reindexar com outro modelo

Se um dia for preciso trocar o modelo, é um comando — e o `--desativar-outras` é obrigatório,
senão o manifesto fica com duas camadas de embeddings ativas e o `construir_app.py` para:

```
python build/construir_indice.py --modelo intfloat/multilingual-e5-small --desativar-outras
python build/construir_app.py
python scripts/gate.py
```
