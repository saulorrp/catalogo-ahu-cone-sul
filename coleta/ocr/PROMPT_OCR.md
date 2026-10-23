# Tarefa: transcrever paginas escaneadas dos catalogos do Projeto Resgate (AHU)

Modelo: Gemini Flash (trabalho em lote, mecanico).

## O que fazer

Para cada imagem em `paginas/<volume>/NNNN.webp`, crie `ocr/<volume>/NNNN.txt`
com a transcricao da pagina. Pule as paginas cujo `.txt` ja existe.
Trabalhe volume por volume, em ordem de pagina.

## Regras de transcricao

1. Transcreva **literalmente**. Nao corrija ortografia, nao modernize, nao
   complete abreviaturas, nao traduza. Mantenha colchetes `[ ]`, reticencias
   `[…]`, acentos e pontuacao exatamente como impressos.
2. A pagina tem colunas (em geral 3). Leia a coluna da esquerda de cima a
   baixo, depois a do meio, depois a da direita.
3. Cada verbete fica separado do seguinte por **uma linha em branco**.
   Dentro do verbete, uma linha para cada parte, nesta ordem quando existir:
   - linha da data (ex.: `[ant. 1710, Março, 12]` ou `1710, Junho, 6, Bahia`)
   - resumo (o texto todo numa linha so, juntando as quebras de linha;
     palavra hifenizada no fim da linha deve ser juntada: `pro-` + `vedor` = `provedor`)
   - `Anexo: ...` / `Anexos: ...`
   - `Obs.: ...`
   - cota antiga (ex.: `AHU-Bahia, cx. 6, doc. 7`)
   - cota nova (ex.: `AHU_ACL_CU_005, Cx. 6, D. 502.`)
4. Se o verbete tiver numero proprio impresso antes da data, mantenha o numero
   no inicio da linha da data, como impresso.
5. **Ignore** cabecalho e rodape da pagina (titulo do catalogo, numero da pagina).
6. Se um verbete comeca numa pagina e termina na seguinte, transcreva so o
   pedaco que esta nesta pagina; nao repita nem complete.
7. Paginas sem verbetes (capa, apresentacao, indices) : transcreva o texto
   normalmente, sem formatacao especial. A primeira linha do arquivo deve ser
   `# SEM VERBETES` nesse caso.
8. Se algum trecho estiver ilegivel, escreva `[ilegivel]` no lugar; nunca invente.

## Exemplo de saida (pagina 50 de Bahia v1)

```
[ant. 1709, Novembro, 23]
REQUERIMENTO do ex-ouvidor-geral e provedor da Bahia Miguel Manso Preto ao rei [D. João V] solicitando que ordene ao desembargador Gregório Pereira que no primeiro paquete que vier do Reino remeta ao Conselho a residência do referido.
AHU-Bahia, cx. 6, doc. 7
AHU_ACL_CU_005, Cx. 6, D. 502.

[ant. 1710, Janeiro, 18, Lisboa]
REQUERIMENTO do mestre da nau Jesus Maria, José Antônio dos Reis ao rei [D. João V] solicitando ordem para viajar a Bahia.
Anexo: 2 docs.
AHU-Bahia, cx. 6, doc. 9
AHU_ACL_CU_005, Cx. 6, D. 503.
```

## Ao terminar cada volume

Escreva `ocr/<volume>/_FEITO.txt` com o numero de paginas transcritas.
