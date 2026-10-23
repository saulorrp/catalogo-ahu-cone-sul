# O catálogo no servidor próprio

O GitHub publica a imagem do app (`ghcr.io/saulorrp/catalogo-ahu`) a cada mudança no `main`.
O servidor roda essa imagem e se atualiza sozinho (Watchtower, de hora em hora). O endereço
público vem do Cloudflare Tunnel, sem abrir portas no roteador e sem expor o IP de casa.

Precisa de: Docker, ~3 GB de RAM livres para o app, ~2 GB de disco.

## Uma vez só

1. **Imagem pública.** Depois do primeiro envio, em github.com/saulorrp?tab=packages →
   `catalogo-ahu` → Package settings → Change visibility → **Public**. (Assim o servidor baixa
   sem login.)
2. **Túnel da Cloudflare** (grátis; precisa de um domínio na Cloudflare):
   Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared → copie o **token**.
   Em *Public Hostname*, aponte o endereço escolhido (ex.: `ahu.seudominio.org`) para
   `http://app:7860`.
3. **No servidor**, numa pasta (ex.: `~/catalogo-ahu`), ponha o `docker-compose.yml` desta pasta
   e um arquivo `.env` com a linha `CF_TUNNEL_TOKEN=...` (o token do passo 2).

## Subir

- **CasaOS:** App Store → Custom Install → Import → cole o `docker-compose.yml`; em
  `cloudflared`, troque `${CF_TUNNEL_TOKEN}` pelo token.
- **Terminal:** `docker compose up -d` na pasta.

Na rede de casa o app abre em `http://IP-DO-SERVIDOR:8501`; de fora, no endereço do túnel.

## Sem domínio próprio

Troque o comando do `cloudflared` por `tunnel --no-autoupdate --url http://app:7860`. O endereço
`https://<palavras>.trycloudflare.com` aparece no log (`docker compose logs cloudflared`) e muda
a cada reinício: serve para testar, não para divulgar.
