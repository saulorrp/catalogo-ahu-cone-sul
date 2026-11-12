"""auditar_banco.py — auditoria de integridade do banco que o app lê (app_dados/documentos.parquet).

Só lê; não muda nada. Cada verificação gera uma contagem e um CSV com os documentos afetados,
para que cada problema possa ser conferido um a um.

Uso: python scripts/auditar_banco.py [--dados app_dados] [--corpus corpus] [--saida auditoria]
Saída: <saida>/resumo.json e <saida>/<verificacao>.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ahu import conversor  # noqa: E402

COLS = ["doc_uid", "colecao", "folder", "new_code", "reference_code", "old_code",
        "document_id_and_date", "ano_isolado", "tipologia"]
ANO_MIN, ANO_MAX = 1500, 1850
LARGURA_ABSURDA = 200


def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _sugerir_fim(inicio, fim) -> str:
    """O fim provável de um intervalo com dígito a mais: tira um dígito do fim e fica com o
    menor valor acima do início (até 60 números). Sem candidato único, devolve ''."""
    inicio, s = int(inicio), str(int(fim))
    cand = {int(s[:k] + s[k + 1:]) for k in range(len(s))}
    cand = sorted(c for c in cand if inicio < c <= inicio + 60)
    return str(cand[0]) if len(cand) == 1 else ""
_MOJIBAKE = re.compile(r"Ã[\x80-\xbf©£§¡¢ª«º³µ]|Â[ºª°]|â€|�")
_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ANO = re.compile(r"\b(1[4-9]\d\d)\b")
_COTA_RESTO = re.compile(r"D(?:oc)?\.?\s*[\d.\s\-–—]+\.?\s*$")


class Auditoria:
    def __init__(self, saida: Path):
        self.saida = saida
        self.resumo: dict[str, dict] = {}

    def registrar(self, chave, titulo, gravidade, grupo, df, explicacao, colunas=None):
        n = int(len(df))
        self.resumo[chave] = {"titulo": titulo, "gravidade": gravidade, "grupo": grupo,
                              "documentos": n, "explicacao": explicacao}
        if n:
            cols = [c for c in (colunas or COLS) if c in df.columns]
            df[cols].to_csv(self.saida / f"{chave}.csv", index=False)
            self.resumo[chave]["exemplos"] = (
                df[cols].head(5).astype(str).to_dict(orient="records"))
        print(f"{gravidade:>6}  {n:>7}  {chave}")


def _cota(texto):
    return conversor.reconhecer(texto) if isinstance(texto, str) else None


def auditar(dados: Path, corpus: Path, saida: Path) -> dict:
    saida.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(dados / "documentos.parquet")
    a = Auditoria(saida)
    total = len(d)

    # ---- identidade
    rep = d[d.doc_uid.duplicated(keep=False)]
    a.registrar("uid_repetido", "doc_uid repetido", "alta", "Identidade", rep,
                "Dois documentos com o mesmo identificador interno.")
    pref = d[d.doc_uid.str.split(":").str[0] != d.colecao]
    a.registrar("uid_colecao", "doc_uid de outra coleção", "alta", "Identidade", pref,
                "O prefixo do doc_uid não bate com a coluna colecao.")

    # ---- cotas
    cotas = [_cota(t) for t in d.new_code]
    d["_serie"] = [c.codigo_serie if c else None for c in cotas]
    d["_cx"] = [c.caixa if c else None for c in cotas]
    d["_doc"] = [c.documento if c else None for c in cotas]
    d["_fim"] = [c.documento_fim if c else None for c in cotas]
    reino_cod = d.folder.isin(["Reino", "Códices"])

    sem = d[d.new_code.isna() | (d.new_code.astype(str).str.strip() == "")]
    a.registrar("cota_vazia", "Sem cota (new_code vazio)", "alta", "Cotas", sem,
                "Documento sem cota atual: não pode ser pedido no arquivo nem achado por cota.")
    ilegivel = d[d._serie.isna() & ~reino_cod & d.new_code.notna()]
    a.registrar("cota_ilegivel", "Cota que o conversor não reconhece", "alta", "Cotas", ilegivel,
                "Fora de Reino e Códices, toda cota deveria ter série, caixa e documento.")
    sem_ref = d[d.reference_code.isna() & ~reino_cod]
    a.registrar("sem_referencia", "Sem código de referência", "media", "Cotas", sem_ref,
                "Sem PT/AHU/CU/…, o documento some da busca por referência e do link DigitArq.")
    esperado = [conversor.para_referencia(c) if c else None for c in cotas]
    d["_ref_esperada"] = esperado
    diverge = d[d._ref_esperada.notna() & d.reference_code.notna()
                & (d._ref_esperada != d.reference_code)]
    a.registrar("referencia_diverge", "Referência diferente da cota atual", "alta", "Cotas",
                diverge, "A referência guardada não é a que o conversor tira do new_code.",
                COLS + ["_ref_esperada"])
    lixo = d[d.new_code.notna() & ~reino_cod & d._serie.notna()
             & ~d.new_code.astype(str).str.strip().str.contains(_COTA_RESTO)]
    a.registrar("cota_com_sobra", "Cota com texto sobrando depois do número", "baixa", "Cotas",
                lixo, "Texto extra depois do número do documento (cabeçalho, nota, pontuação).")
    seis = d[d._doc.fillna(0) > 99999]
    a.registrar("doc_seis_digitos", "Número de documento com 6 dígitos", "media", "Cotas", seis,
                "Provável erro de catálogo (dígito a mais); a série vai até 5 dígitos.")

    # intervalos
    interv = d[d._fim.notna()].copy()
    interv["_largura"] = interv._fim - interv._doc + 1
    a.registrar("intervalo", "Cota em intervalo (D. a-b)", "info", "Intervalos", interv,
                "Um verbete cobre vários números. A referência guarda só o primeiro; os demais "
                "números só são achados se a busca souber expandir o intervalo.",
                COLS + ["_largura"])
    absurdo = interv[interv._largura > LARGURA_ABSURDA].copy()
    absurdo["_sugestao"] = [_sugerir_fim(x, f) for x, f in zip(absurdo._doc, absurdo._fim)]
    a.registrar("intervalo_absurdo", f"Intervalo com mais de {LARGURA_ABSURDA} números",
                "alta", "Intervalos", absurdo, "Dígito a mais ou a menos num dos extremos "
                "(D. 17330-173332). Se a busca expandir o intervalo, o verbete casa com milhares "
                "de números que não são dele. A coluna _sugestao traz o fim provável.",
                COLS + ["_largura", "_sugestao"])
    interv = interv[interv._largura <= LARGURA_ABSURDA]
    largo = interv[interv._largura > 30]
    a.registrar("intervalo_largo", "Intervalo com mais de 30 números", "media", "Intervalos",
                largo, "Intervalo suspeito de dígito trocado.", COLS + ["_largura"])
    invertido = d[d.new_code.astype(str).str.contains(r"D\.\s*\d+\s*[-–]\s*\d+")
                  & d._fim.isna() & d._doc.notna()]
    a.registrar("intervalo_invertido", "Intervalo com fim menor que o início", "media",
                "Intervalos", invertido, "O conversor descarta o fim e o intervalo se perde.")

    # chaves repetidas e sobreposição
    com = d[d._serie.notna()]
    chave = com._serie + "|" + com._cx.astype(int).astype(str) + "|" + com._doc.astype(int).astype(str)
    com = com.assign(_chave=chave)
    com["_texto"] = com.description.fillna("").map(_normalizar)
    rep = com[com._chave.duplicated(keep=False)].sort_values("_chave")
    exata = rep.duplicated(["_chave", "_texto"], keep=False) & (rep._texto != "")
    duas = rep.groupby(["_chave", "_texto"]).colecao.transform("nunique") > 1
    a.registrar("duplicata_entre_colecoes", "Mesmo verbete no ahu-sul e no ahu-brasil", "alta",
                "Duplicatas", rep[exata & duas], "Mesma cota e mesmo resumo nas duas coleções: o "
                "documento aparece duas vezes na busca. A cópia do ahu-brasil deveria ter ficado "
                "de fora.", COLS + ["_chave"])
    a.registrar("duplicata_exata", "Mesmo verbete repetido na mesma coleção", "media",
                "Duplicatas", rep[exata & ~duas], "Mesma cota e mesmo resumo duas vezes "
                "(verbete repetido na extração ou no catálogo).", COLS + ["_chave"])
    a.registrar("numero_compartilhado", "Mesma cota com resumos diferentes", "baixa",
                "Duplicatas", rep[~exata], "Verbetes diferentes com o mesmo número: anexos "
                "descritos à parte (comum no Castro e Almeida) ou número digitado errado.",
                COLS + ["_chave", "description"])
    dup_texto = d[d.description.notna() & (d.description.str.len() > 40)]
    dup_texto = dup_texto[dup_texto.duplicated(["folder", "description"], keep=False)]
    dup_texto = dup_texto.sort_values(["folder", "description"])
    a.registrar("resumo_repetido", "Resumo idêntico na mesma série", "baixa", "Duplicatas",
                dup_texto, "Mesmo texto em verbetes diferentes: duplicata real ou documento "
                "copiado em vias. Conferir a cota.", COLS + ["description"])

    # número dentro do intervalo de outro verbete
    faixas = defaultdict(list)  # só os intervalos plausíveis
    for i, s, x, f in zip(interv.index, interv._serie, interv._doc, interv._fim):
        faixas[s].append((int(x), int(f), i))
    dentro = []
    for i, s, x in zip(com.index, com._serie, com._doc):
        for ini, fim, j in faixas.get(s, ()):
            if j != i and ini <= x <= fim:
                dentro.append(i)
                break
    a.registrar("dentro_de_intervalo", "Número contido no intervalo de outro verbete", "media",
                "Duplicatas", d.loc[dentro], "O mesmo documento descrito sozinho e dentro de "
                "um intervalo de outro verbete.")

    # sequência: dentro da série, a caixa não pode descer quando o número sobe
    fora = []
    for s, g in com.groupby("_serie"):
        g = g.sort_values("_doc")
        cx = g._cx.to_numpy()
        doc = g._doc.to_numpy()
        for k in range(1, len(g) - 1):
            # caixa fora dos vizinhos, que concordam entre si
            if cx[k - 1] <= cx[k + 1] and not (cx[k - 1] <= cx[k] <= cx[k + 1]):
                fora.append(g.index[k])
            # salto de número muito grande (dígito a mais)
            elif doc[k] > 3 * doc[k - 1] + 1000 and doc[k + 1] < doc[k] / 2:
                fora.append(g.index[k])
    a.registrar("fora_de_sequencia", "Caixa ou número fora da sequência da série", "media",
                "Cotas", d.loc[fora], "Ordenada a série pelo número, a caixa deste verbete não "
                "fica entre a dos vizinhos: dígito trocado na caixa ou no documento.")

    # pasta x série
    nome_serie = {s.codigo: s.nome.split(",")[0] for s in conversor.SERIES}
    d["_pasta_serie"] = d._serie.map(nome_serie)
    pasta = d[d._pasta_serie.notna() & (d._pasta_serie != d.folder)]
    a.registrar("pasta_diverge", "Pasta diferente da série da cota", "media", "Campos", pasta,
                "A pasta (folder) não é a da série na cifra; o filtro por pasta erra.",
                COLS + ["_pasta_serie"])

    # cota antiga de numeração igual
    antigas = [_cota(t) for t in d.old_code]
    ruim = []
    for i, c_ant, s, x in zip(d.index, antigas, d._serie, d._doc):
        if c_ant and c_ant.serie is not None and s == c_ant.codigo_serie and x != c_ant.documento:
            ruim.append(i)
    a.registrar("antiga_diverge", "Cota antiga Calmeida/MGouveia com número diferente", "media",
                "Cotas", d.loc[ruim], "Nessas duas coleções a cota antiga e a atual têm o mesmo "
                "número; diferença indica erro em uma delas.")

    # DigitArq
    sem_dig = d[d.reference_code.notna() & d.referencia_digitarq.isna()]
    a.registrar("sem_digitarq", "Sem registro no DigitArq", "baixa", "Ligações", sem_dig,
                "O link vai para a caixa, não para o documento.")

    # ---- campos
    desc = d.description.fillna("").str.strip()
    a.registrar("sem_resumo", "Sem resumo", "alta", "Campos", d[desc == ""],
                "Verbete sem texto: invisível para a busca por assunto.")
    a.registrar("resumo_curto", "Resumo com menos de 25 caracteres", "media", "Campos",
                d[(desc.str.len() > 0) & (desc.str.len() < 25)],
                "Texto curto demais: provável corte na extração.", COLS + ["description"])
    a.registrar("sem_ano", "Sem ano", "media", "Campos", d[d.ano_isolado.isna()],
                "Fica de fora de qualquer filtro por período.")
    fora_ano = d[d.ano_isolado.notna() & ((d.ano_isolado < ANO_MIN) | (d.ano_isolado > ANO_MAX))]
    a.registrar("ano_improvavel", f"Ano fora de {ANO_MIN}–{ANO_MAX}", "media", "Campos",
                fora_ano, "Ano fora do intervalo do acervo: provável erro de leitura.")
    anos_txt = d.document_id_and_date.fillna("").map(lambda t: {int(x) for x in _ANO.findall(t)})
    fora_txt = pd.Series([int(a_) not in s for a_, s in zip(d.ano_isolado.fillna(0), anos_txt)],
                         index=d.index)
    ano_mismatch = d[d.ano_isolado.notna() & anos_txt.map(bool) & fora_txt]
    a.registrar("ano_diverge_data", "Ano não aparece na data do verbete", "media", "Campos",
                ano_mismatch, "O ano isolado não é nenhum dos anos escritos na data.")
    par = d.document_id_and_date.fillna("").str.extract(r"^\s*(\d+)\s*[.\-–]\s*(\d+)\s*\.?\s*$")
    par = par.astype(float)
    ano_f = d.ano_isolado.astype("float")
    falso = d[par[1].notna() & (ano_f == par[1])
              & ((par[1] == d._fim) | ((par[1] > par[0]) & (par[1] - par[0] <= 60)))]
    a.registrar("ano_falso", "Ano tirado do número do documento", "alta", "Campos", falso,
                "A data do verbete é só o intervalo de números (\"1399. 1401\") e o fim do "
                "intervalo virou ano. O documento aparece num período que não é o dele.")
    texto = d.description.fillna("") + " " + d.document_id_and_date.fillna("")
    a.registrar("mojibake", "Texto com codificação quebrada", "media", "Campos",
                d[texto.str.contains(_MOJIBAKE)], "Sequências como Ã© ou � no texto.",
                COLS + ["description"])
    a.registrar("caractere_controle", "Caractere de controle no texto", "baixa", "Campos",
                d[texto.str.contains(_CONTROLE)], "Caracteres invisíveis que atrapalham a busca.")
    nfc = d[d.description.fillna("").map(lambda t: unicodedata.normalize("NFC", t) != t)]
    a.registrar("nao_nfc", "Texto fora da forma NFC", "baixa", "Campos", nfc,
                "Acentos decompostos: a mesma palavra fica com duas grafias no índice.")
    espacos = d[d.description.fillna("").str.contains(r"^\s|\s$|\s{2,}")]
    a.registrar("espacos", "Espaços sobrando no resumo", "baixa", "Campos", espacos,
                "Espaço no começo, no fim ou dobrado.")
    # "Sergipe" (série 081) e "Sergipe d’el-Rei" (022) são séries diferentes: não entram aqui.
    grafias = {"Montividéu": "Montevidéu", "Brasil": "Brasil Geral"}
    a.registrar("pasta_grafia", "Pasta com grafia variante", "media", "Campos",
                d[d.folder.isin(grafias)], "A mesma região aparece em duas pastas no filtro: "
                + ", ".join(f"{k} / {v}" for k, v in grafias.items()) + ".")

    # ---- tipologia
    a.registrar("sem_tipologia", "Sem tipologia", "baixa", "Tipologia", d[d.tipologia.isna()],
                "Fica de fora do filtro por tipo documental.")
    freq = d.tipologia_chave.value_counts()
    suspeitas = {"PARA", "PRIMEIRA", "SEGUNDA", "TERCEIRA", "QUARTA", "QUINTA", "FORTALEZA",
                 "DE", "DO", "DA", "EM", "SOBRE", "SENHOR", "REI", "DOM"}
    lixo_tip = d[d.tipologia_chave.isin(suspeitas)
                 | d.tipologia_chave.fillna("").str.match(r"^[A-Z]{1,2}$|.*\d")]
    a.registrar("tipologia_lixo", "Tipologia que é palavra solta da frase", "media", "Tipologia",
                lixo_tip, "A extração pegou a palavra errada (ex.: PARA, PRIMEIRA).",
                COLS + ["description"])
    plural = {k for k in freq.index if k.endswith("S") and k[:-1] in freq.index}
    a.registrar("tipologia_plural", "Tipologia no plural ao lado do singular", "baixa",
                "Tipologia", d[d.tipologia_chave.isin(plural)],
                f"{len(plural)} tipologias têm forma plural e singular (OFÍCIOS/OFÍCIO); o filtro "
                "mostra as duas como tipos diferentes.")
    rara = d[d.tipologia_chave.map(freq).fillna(0) == 1]
    a.registrar("tipologia_unica", "Tipologia que aparece uma vez só", "baixa", "Tipologia",
                rara, "Candidatas a erro de grafia ou a juntar com outra.")

    # ---- coleções e ligações
    cb = corpus / "colecoes" / "ahu-brasil" / "documentos.parquet"
    extra = {}
    if cb.exists():
        b = pd.read_parquet(cb, columns=["doc_uid", "qc", "caminho_resgate", "url_actd"])
        extra["colunas_do_corpus_fora_do_app"] = [
            c for c in ["qc", "caminho_resgate", "url_actd", "pagina_actd", "fonte_texto"]
            if c not in d.columns]
        extra["qc_marcados"] = int(b.qc.notna().sum())
        extra["qc_por_marca"] = (b.qc.dropna().str.split("; ").explode().value_counts()
                                 .astype(int).to_dict())
        extra["com_caminho_resgate"] = int(b.caminho_resgate.notna().sum())
        extra["com_url_actd"] = int(b.url_actd.notna().sum())
    sul = d[d.colecao == "ahu-sul"]
    extra["ahu_sul"] = int(len(sul))
    extra["ahu_brasil"] = int((d.colecao == "ahu-brasil").sum())
    extra["ahu_brasil_sem_remetente"] = int(d[d.colecao == "ahu-brasil"].sender_name.isna().sum())

    resumo = {"total": total, "verificacoes": a.resumo, "extra": extra,
              "gerado_de": str(dados / "documentos.parquet")}
    (saida / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1,
                                                  default=str), encoding="utf-8")
    return resumo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dados", type=Path, default=RAIZ / "app_dados")
    p.add_argument("--corpus", type=Path, default=RAIZ / "corpus")
    p.add_argument("--saida", type=Path, default=RAIZ / "auditoria")
    args = p.parse_args(argv)
    auditar(args.dados, args.corpus, args.saida)


if __name__ == "__main__":
    main()
