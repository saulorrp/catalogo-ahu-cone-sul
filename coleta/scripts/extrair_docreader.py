"""Extrai, das páginas de verbetes do Projeto Resgate (gov.br), o quadro do DocReader de cada documento.

As tabelas HTML de 13 capitanias no gov.br (Alagoas, Ceará, Espírito Santo, Goiás, Mato Grosso,
Paraíba, Piauí, Rio Grande do Norte, Rio Grande do Sul, Rio Negro, Santa Catarina...) ligam o número
de cada verbete ao visualizador de imagens da Biblioteca Nacional:

    https://resgate.bn.br/docreader/DocReader.aspx?bib=021_SC&hf=resgate.bn.br&pagfis=37

`bib` é o rolo digitalizado da série (`<série>_<UF>`) e `pagfis` o quadro em que o documento começa.
O `parse_verbetes.py` guarda esse href em `url_imagem`; aqui ele vira um índice pela referência
CRAV (`PT/AHU/CU/021/0001/00004`), que o `build/cotas.py` encaixa em cada documento.

Uso: python coleta/scripts/extrair_docreader.py /mnt/project-files/resgate/verbetes corpus/resgate_docreader.tsv.gz
"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ahu.conversor import para_referencia, reconhecer  # noqa: E402

_HREF = re.compile(r"DocReader\.aspx\?(?:.*&)?bib=([\w]+)(?:&.*)?&pagfis=(\d+)", re.I)


def extrair(pasta: Path):
    vistos, contagem = {}, Counter()
    for arquivo in sorted(pasta.glob("*.jsonl")):
        if arquivo.name.startswith(("tematico-", "contratos-")):
            continue  # seleções temáticas repetem verbetes das capitanias, sem link próprio
        for linha in open(arquivo, encoding="utf-8"):
            r = json.loads(linha)
            m = _HREF.search(r.get("url_imagem") or "")
            if not m:
                continue
            contagem["com_link"] += 1
            cota = reconhecer(r.get("new_code") or "")
            ref = para_referencia(cota) if cota else None
            if not ref:
                contagem["sem_cota"] += 1
                continue
            if ref in vistos:
                contagem["referencia_repetida"] += 1  # vale o primeiro verbete (o quadro mais baixo)
                continue
            vistos[ref] = (m.group(1), int(m.group(2)))
    contagem["no_indice"] = len(vistos)
    return vistos, contagem


if __name__ == "__main__":
    vistos, contagem = extrair(Path(sys.argv[1]))
    with gzip.open(sys.argv[2], "wt", encoding="utf-8") as f:
        f.write("referencia\tbib\tpagfis\n")
        for ref in sorted(vistos):
            bib, pagfis = vistos[ref]
            f.write(f"{ref}\t{bib}\t{pagfis}\n")
    print(dict(contagem))
    print(Counter(b for b, _ in vistos.values()).most_common())
