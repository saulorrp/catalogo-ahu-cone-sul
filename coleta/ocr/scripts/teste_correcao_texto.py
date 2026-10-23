"""Teste: corrigir o Tesseract SO pelo texto (sem a imagem), na amostra tesseract_amostra/.

Saida em teste_correcao_texto_so/ (nunca no pipeline). Compara com a transcricao direta do
Gemini e com o Opus onde houver. Uso: python teste_correcao_texto.py
"""
import re, statistics
from concurrent.futures import ThreadPoolExecutor
import corrigir_tesseract as ct
import ocr_comum as c
import qualidade_tesseract as q

SAIDA = c.KIT / "teste_correcao_texto_so"
PEDIDO = """Abaixo esta o texto que um OCR (Tesseract) leu de uma pagina de um catalogo impresso do Projeto Resgate \
(Arquivo Historico Ultramarino), em portugues. Voce NAO tem a imagem. Corrija SO erros previsiveis de OCR:
letras trocadas (c/e, 1/I, rn/m, li/h), "ex" por "cx", "20" por "ao", º/ª, ")" ou "|" por "]", "(" por "[",
pontuacao trocada, palavras partidas ou coladas. NUNCA mude algarismos de cotas (Cx., D., doc., Doc.), de
numeros de verbete nem de datas; nao modernize a grafia; nao acrescente nem remova verbetes; nao invente.
Exemplos de correcao: "Consclho" -> "Conselho"; "AHU-Bahia, ex. 6" -> "AHU-Bahia, cx. 6"; "20 rei" -> "ao rei";
"[ant. 1710, Marco, 12)" -> "[ant. 1710, Marco, 12]".
Organize o resultado segundo estas regras:

{regras}

{fim}

TEXTO DO OCR:
{cru}"""


def corrige(alvo):
    v, p = alvo
    destino = SAIDA / v / f"{p:04d}.txt"
    if destino.exists():
        return 0.0
    cru = (c.KIT / "tesseract_amostra" / v / f"{p:04d}.txt").read_text(encoding="utf-8", errors="replace")
    corpo = {"contents": [{"parts": [{"text": PEDIDO.format(regras=c.regras(), fim=c.FIM, cru=cru)}]}],
             "generationConfig": {"temperature": 0, "thinkingConfig": {"thinkingLevel": "low"}}}
    d = c.pede(f"{c.API}/models/{c.MODELO_API}:generateContent", corpo)
    cand = (d.get("candidates") or [{}])[0]
    texto = "".join(x.get("text", "") for x in cand.get("content", {}).get("parts", []) if not x.get("thought"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(c.limpa(texto) if texto.strip() else "", encoding="utf-8")
    return c.custo(d.get("usageMetadata", {}), "padrao")


def nomes(t):
    """Capitalised words (proper names, places), not the all-caps document types."""
    return re.findall(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b", t)


def erros_nomes(ref, hip):
    import difflib
    a, b = nomes(ref), nomes(hip)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(i2 - i1 for tag, i1, i2, _, _ in sm.get_opcodes() if tag != "equal"), len(a)


def main():
    alvos = [(f.parent.name, int(f.stem)) for f in sorted((c.KIT / "tesseract_amostra").glob("*/[0-9]*.txt"))
             if q.referencia(f.parent.name, int(f.stem))[1]]
    with ThreadPoolExecutor(4) as pool:
        custos = list(pool.map(corrige, alvos))
    linhas = ["| grupo | pag. | WER cru | WER corr. texto | D. cru | D. corr. texto | anos iguais corr. | nomes errados cru / corr. |",
              "|---|---|---|---|---|---|---|---|"]
    grupos = {}
    for v, p in alvos:
        g = next(k for k, vs in q.GRUPOS.items() if v in vs)
        grupos.setdefault(g, []).append((v, p))
    for g, ps in grupos.items():
        w0, w1, d0, d1, dt, an, nt, n0, n1 = [], [], 0, 0, 0, 0, 0, 0, 0
        for v, p in ps:
            ref = q.referencia(v, p)[1]
            cru = (c.KIT / "tesseract_amostra" / v / f"{p:04d}.txt").read_text(encoding="utf-8", errors="replace")
            cor = (SAIDA / v / f"{p:04d}.txt").read_text(encoding="utf-8")
            w0.append(q.wer(ref, cru)[0]); w1.append(q.wer(ref, cor)[0])
            cr = q.cotas(ref); dt += len(cr); d0 += len(cr & q.cotas(cru)); d1 += len(cr & q.cotas(cor))
            an += q.datas(ref) == q.datas(cor)
            e0, tot = erros_nomes(ref, cru); e1, _ = erros_nomes(ref, cor); nt += tot; n0 += e0; n1 += e1
        linhas.append(f"| {g} | {len(ps)} | {statistics.median(w0):.1%} | {statistics.median(w1):.1%} | {d0}/{dt} | "
                      f"{d1}/{dt} | {an}/{len(ps)} | {n0} / {n1} de {nt} |")
    opus = []
    for f in sorted((c.KIT / "ocr_opus").glob("*/[0-9]*.txt")):
        cor = SAIDA / f.parent.name / f.name
        if cor.exists():
            o = f.read_text(encoding="utf-8")
            opus.append(q.wer(o, cor.read_text(encoding="utf-8"))[0])
    print("\n".join(linhas))
    if opus:
        print(f"\ncontra o Opus ({len(opus)} pag.): WER mediano da correcao so-texto {statistics.median(opus):.1%}")
    print(f"custo: US$ {sum(custos):.3f} em {sum(1 for x in custos if x)} paginas "
          f"(~US$ {sum(custos) / max(1, sum(1 for x in custos if x)):.4f}/pag; a correcao com imagem custa ~US$ 0.0105)")


if __name__ == "__main__":
    main()
