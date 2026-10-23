# Enriquecimento leve com o Gemini

Extrai de cada resumo do catálogo, para os 167 mil documentos (ahu-sul e ahu-brasil):

| Campo | O que é |
|---|---|
| `rem` | Remetente: `nome`, `cargo` (cargo, título ou condição, como no resumo) e `tipo` (pessoa, instituição, coletivo) |
| `dest` | Destinatário, nos mesmos três campos |
| `mencionados` | Outras pessoas citadas pelo nome |
| `lugares` | Lugares citados |
| `assunto` | 1 a 3 palavras-chave do assunto (o único campo interpretativo) |

São variáveis **feitas pelo Gemini**: cada lote guarda o modelo e a rota que o produziram, e é assim
que o app as cita. Na ahu-sul elas convivem com as do DeepSeek, de propósito: as duas ficam
separadas e identificadas. O pedido exato está em [`PROMPT.md`](PROMPT.md).

## Como roda

1. `python coleta/enriquecimento/preparar.py` (no repositório) gera `entrada/lotes.jsonl.gz`: 4.183
   lotes de 40 verbetes. Os 15 primeiros são uma amostra de 600 documentos da ahu-sul, que já tem
   remetente e destinatário anotados.
2. No computador que roda (Hermerico), dentro desta pasta: `python enriquecer.py`. Usa duas rotas
   sem custo, a assinatura AI Pro pelo `agy` e a chave da conta AI Plus no nível gratuito
   (`GEMINI_API_KEY_ESTUDANTE`). A chave paga é recusada. Quando a cota de uma rota acaba, ela
   espera e continua. Pode ser parado e retomado: lote pronto não roda de novo.
3. `python coleta/enriquecimento/avaliar.py saida/` compara remetente e destinatário com os
   anotados da ahu-sul, por modelo.
4. A importação como camada do corpus (`pessoas_gemini_v1`) vem depois da avaliação.

Saída: `saida/<lote>.json`, com `modelo`, `rota`, `data` e os campos acima por `doc_uid`. Verbetes
que falharam mesmo em lotes de 5 vão para `saida/_falhas.jsonl`.
