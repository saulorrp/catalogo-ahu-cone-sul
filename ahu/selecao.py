"""selecao.py — WP 3.1: a seleção de documentos que atravessa buscas (spec §8.1).

Sem Streamlit. A `Selecao` é uma lista ordenada de `doc_uid` — nada mais. O `app.py` a guarda em
`st.session_state`, e é só isso: **nada é persistido no servidor** (R4). Quem quiser guardar a
seleção para amanhã baixa um JSON e carrega de volta, e é por isso que `para_json` e `de_json`
estão aqui, testáveis, em vez de no `app.py`.

Três coisas que a §8.1 pede e que esta classe garante:

1. **a seleção atravessa buscas.** Ela não guarda posição, nota nem página: só o `doc_uid`, que é
   o identificador público e estável (AGENTS.md 6). Mudar a busca não mexe nela;
2. **teto de 500 documentos.** `adicionar` nunca passa do teto e diz quantos ficaram de fora, para
   a interface poder avisar em vez de truncar em silêncio;
3. **ordem significa algo.** A ordem padrão é a de chegada — a ordem em que o pesquisador marcou —
   e `ordenar` reordena por ano, cota ou tipologia quando ele pede. Nunca é a ordem da busca.

O JSON tem esquema versionado (`selecao-ahu-sul/1`). Um arquivo de versão desconhecida é recusado
com mensagem clara, e `doc_uid` que não existe mais no corpus é descartado e **relatado**: um
arquivo de seis meses atrás não pode virar uma seleção silenciosamente menor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

LIMITE = 500            # §8.1
ESQUEMA = "selecao-ahu-sul/1"
ORDENS = ("chegada", "ano_asc", "ano_desc", "cota", "tipologia")
ROTULOS_ORDEM = {
    "chegada": "Ordem em que foram marcados",
    "ano_asc": "Ano (mais antigo primeiro)",
    "ano_desc": "Ano (mais recente primeiro)",
    "cota": "Código de referência",
    "tipologia": "Tipologia",
}


class ErroDeSelecao(ValueError):
    """Arquivo de seleção inválido: esquema desconhecido ou conteúdo fora do formato."""


@dataclass
class Selecao:
    """Os `doc_uid` marcados, na ordem em que foram marcados."""

    uids: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.uids)

    def __contains__(self, doc_uid: str) -> bool:
        return doc_uid in self.uids

    @property
    def cheia(self) -> bool:
        return len(self.uids) >= LIMITE

    def limpar(self) -> None:
        self.uids = []

    def adicionar(self, doc_uids) -> int:
        """Acrescenta o que ainda não está, sem passar do teto. Devolve quantos **não** couberam."""
        if isinstance(doc_uids, str):
            doc_uids = [doc_uids]
        novos = [u for u in dict.fromkeys(doc_uids) if u and u not in self.uids]
        espaco = LIMITE - len(self.uids)
        self.uids.extend(novos[:espaco])
        return max(0, len(novos) - espaco)

    def remover(self, doc_uids) -> None:
        if isinstance(doc_uids, str):
            doc_uids = [doc_uids]
        fora = set(doc_uids)
        self.uids = [u for u in self.uids if u not in fora]

    def alternar(self, doc_uid: str, marcado: bool) -> int:
        """O que a caixa "Selecionar" de cada resultado faz. Devolve quantos não couberam."""
        if marcado:
            return self.adicionar(doc_uid)
        self.remover(doc_uid)
        return 0

    # -----------------------------------------------------------------
    def ordenar(self, dados, ordem: str = "chegada") -> list[str]:
        """Os `doc_uid` na ordem pedida. Não muda a seleção: a ordem de chegada é preservada."""
        if ordem not in ORDENS:
            raise ValueError(f"Ordem desconhecida: {ordem!r}. Use uma de {', '.join(ORDENS)}.")
        if ordem == "chegada":
            return list(self.uids)
        docs = dados.documentos
        def chave(uid: str):
            linha = docs.iloc[dados.indice_de(uid)]
            if ordem in ("ano_asc", "ano_desc"):
                ano = linha.get("ano_isolado")
                ausente = ano is None or pd.isna(ano)
                valor = 0 if ausente else int(ano)
                # Documento sem data vai para o fim nas duas ordens.
                return (ausente, -valor if ordem == "ano_desc" else valor, uid)
            campo = "reference_code" if ordem == "cota" else "tipologia"
            valor = linha.get(campo)
            return ("" if valor is None or pd.isna(valor) else str(valor), uid)
        return sorted(self.uids, key=chave)

    def tabela(self, dados, ordem: str = "chegada") -> pd.DataFrame:
        """A seleção em forma de tabela, para o pesquisador conferir. **Sem `doc_uid`** (D10)."""
        docs = dados.documentos
        linhas = []
        for posicao, uid in enumerate(self.ordenar(dados, ordem), start=1):
            linha = docs.iloc[dados.indice_de(uid)]
            def valor(campo):
                v = linha.get(campo)
                return "" if v is None or pd.isna(v) else v
            ano = valor("ano_isolado")
            linhas.append({
                "Nº": posicao,
                "Ano": "" if ano == "" else int(ano),
                "Tipologia": valor("tipologia"),
                "Código de referência": valor("reference_code"),
                "Código atual": valor("new_code"),
                "Pasta": valor("folder"),
            })
        return pd.DataFrame(linhas, columns=["Nº", "Ano", "Tipologia", "Código de referência",
                                             "Código atual", "Pasta"])

    # -----------------------------------------------------------------
    # Baixar e carregar (R4: nada fica no servidor)
    # -----------------------------------------------------------------
    def para_json(self, proveniencia: dict | None = None) -> bytes:
        """O arquivo que o botão "Baixar seleção" entrega."""
        conteudo = {
            "schema": ESQUEMA,
            "gerado_em": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "documentos": len(self.uids),
            "doc_uids": list(self.uids),
        }
        if proveniencia:
            conteudo["proveniencia"] = proveniencia
        return json.dumps(conteudo, ensure_ascii=False, indent=2).encode("utf-8")

    @classmethod
    def de_json(cls, bruto: bytes | str, dados=None) -> tuple["Selecao", dict]:
        """Lê o arquivo de volta. Devolve a seleção e um relatório do que foi descartado.

        O relatório é o que a interface mostra: `desconhecidos` são `doc_uid` que não existem
        mais no corpus ativo (o corpus mudou, ou o arquivo é de outra coleção) e `excedentes` é
        quanto passou do teto de 500.
        """
        try:
            conteudo = json.loads(bruto.decode("utf-8") if isinstance(bruto, bytes) else bruto)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ErroDeSelecao(f"Este arquivo não é um JSON válido ({e}).") from e
        if not isinstance(conteudo, dict):
            raise ErroDeSelecao("O arquivo de seleção deveria ser um objeto JSON.")
        esquema = conteudo.get("schema")
        if esquema != ESQUEMA:
            raise ErroDeSelecao(
                f"Esquema de seleção desconhecido: {esquema!r}. Este app lê {ESQUEMA!r}."
            )
        uids = conteudo.get("doc_uids")
        if not isinstance(uids, list) or any(not isinstance(u, str) for u in uids):
            raise ErroDeSelecao("`doc_uids` deveria ser uma lista de identificadores.")

        desconhecidos = []
        if dados is not None:
            conhecidos = []
            for u in uids:
                (conhecidos if u in dados.posicao else desconhecidos).append(u)
            uids = conhecidos
        selecao = cls()
        excedentes = selecao.adicionar(uids)
        return selecao, {"carregados": len(selecao), "desconhecidos": desconhecidos,
                         "excedentes": excedentes, "gerado_em": conteudo.get("gerado_em")}
