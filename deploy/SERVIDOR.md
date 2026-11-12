# O catálogo no servidor próprio

O GitHub publica a imagem do app (`ghcr.io/saulorrp/catalogo-ahu`, pública) a cada mudança no
`main`. O servidor roda essa imagem e se atualiza sozinho (Watchtower, de hora em hora). O
endereço público vem do **Tailscale Funnel**, sem abrir portas no roteador e sem expor o IP de casa.

No agiulfo (Mac mini, Ubuntu 24.04, CasaOS) o app está em
**https://agiulfo.tailc278e6.ts.net/ahu/**, ao lado do `/triagem/`.

Esta é a **versão completa**: e5-small em fp32 e o reranker (`AHU_RERANKER=1`, ver
`docs/plano_busca_servidor.md`). A versão leve, sem nada disso, continua no Streamlit Cloud.

Precisa de: Docker, ~3 GB de RAM livres para o app, ~2 GB de disco.

## Uma vez só

1. **Pasta e compose.** `mkdir -p ~/catalogo-ahu` e copie para lá o `docker-compose.yml` desta
   pasta.
2. **Modelos.** Na primeira busca o app baixa o e5 e o reranker do Hugging Face para o volume
   `catalogo-ahu_modelo` (~1 GB). Se a rede da casa cortar o download no meio, baixe com `curl`
   (que retoma) e copie para o volume, mantendo a forma `<dono>__<repositório>/<arquivo>`:

       Xenova__multilingual-e5-small/{model.onnx,tokenizer.json}
       cross-encoder__mmarco-mMiniLMv2-L12-H384-v1/{model_quint8_avx2.onnx,tokenizer.json}

3. **Funnel.** O app escuta só em `127.0.0.1:8502` e serve em `/ahu`
   (`STREAMLIT_SERVER_BASE_URL_PATH`). Publique esse caminho:

       sudo tailscale funnel --bg --set-path /ahu http://127.0.0.1:8502/ahu

   O Funnel guarda a configuração e volta sozinho depois de reiniciar a máquina.

## Subir

    cd ~/catalogo-ahu && sudo docker compose up -d

Para conferir: `curl -s https://agiulfo.tailc278e6.ts.net/ahu/saude` responde
`{"status":"ok",…}` com o número de documentos. (Com `AHU_FRONT=streamlit`, o endereço é
`/ahu/_stcore/health`, que responde `ok`.)

Ao subir, o app carrega os modelos e faz uma busca de aquecimento em segundo plano, para a
primeira pessoa não pagar a carga (`AHU_SEM_AQUECIMENTO=1` desliga). `AHU_RERANKER_TOPO`
(padrão 30) e `AHU_RERANKER_TOKENS` (padrão 192) ajustam o reranker sem mexer no código; as
medidas estão em `docs/plano_busca_servidor.md`.

## Tirar do ar

    sudo tailscale funnel --set-path /ahu off       # só o endereço público
    cd ~/catalogo-ahu && sudo docker compose down    # o app
