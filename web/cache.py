"""cache.py — a memória das buscas e a busca em dois tempos da interface web.

Duas coisas tiram a espera da tela, sem mexer no que a busca devolve:

1. **Memória das respostas.** "Carregar mais", o botão de voltar e um filtro desmarcado repetem
   uma consulta que o servidor acabou de fazer. A resposta fica guardada, pela consulta sem o
   número da página, e a segunda vez não custa nada.
2. **Dois tempos.** Com o reranker ligado, a fusão (léxico + sentido) responde em menos de um
   segundo e a tela já mostra a lista; o cross-encoder relê o topo num segundo pedido, e a lista
   se reordena quando ele termina (`ahu.busca.refinar`). O total, os filtros e as contagens são os
   mesmos nos dois tempos: só a ordem do topo muda.

Nada aqui é persistido: a memória vive no processo e some com ele (R4).
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import pandas as pd

from ahu import busca
from ahu.consulta import Consulta, executar

MAXIMO_DE_RESPOSTAS = 64
MAXIMO_DE_RESULTADOS = 600_000      # soma dos resultados guardados (o corpus inteiro tem ~167 mil)

_tranca = threading.Lock()
_respostas: OrderedDict = OrderedDict()
_por_dados: dict = {}


def limpar() -> None:
    with _tranca:
        _respostas.clear()
        _por_dados.clear()


def _chave(dados, consulta: Consulta) -> str:
    d = consulta.para_dicionario()
    d.pop("pagina", None)
    d.pop("filtros_ativos", None)
    return f"{id(dados)}|" + json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)


def _ler(chave):
    with _tranca:
        valor = _respostas.get(chave)
        if valor is not None:
            _respostas.move_to_end(chave)
        return valor


def _guardar(chave, valor) -> None:
    with _tranca:
        _respostas[chave] = valor
        _respostas.move_to_end(chave)
        while len(_respostas) > 1 and (
                len(_respostas) > MAXIMO_DE_RESPOSTAS
                or sum(len(v[0].resultados) for v in _respostas.values()) > MAXIMO_DE_RESULTADOS):
            _respostas.popitem(last=False)


def _facetas(dados, resposta) -> dict:
    """Quantos resultados caem em cada coleção, pasta, tipologia e década."""
    if not resposta.resultados:
        return {}
    df = dados.documentos.iloc[[r.linha for r in resposta.resultados]]
    facetas = {
        "colecao": df["colecao"].value_counts().to_dict(),
        "folder": df["folder"].value_counts().to_dict(),
        "tipologia": df["tipologia"].value_counts().to_dict(),
    }
    anos = pd.to_numeric(df["ano_isolado"], errors="coerce").dropna()
    decadas = (anos // 10 * 10).astype(int).value_counts().to_dict() if not anos.empty else {}
    facetas["decada"] = {int(k): int(v) for k, v in decadas.items()}
    return facetas


def usa_reranker(consulta: Consulta, reranker) -> bool:
    """O reranker só reordena por relevância: com outra ordenação, ou no modo Exata, não entra."""
    return (reranker is not None and bool(consulta.texto.strip())
            and consulta.modo != "exata" and consulta.ordenacao == "relevancia")


def buscar(dados, consulta: Consulta, encoder=None, reranker=None, esperar: bool = True):
    """Devolve `(resposta, facetas, pendente)`.

    `pendente` é verdadeiro quando a resposta é a da fusão e o reranker ainda vai relê-la: a
    interface mostra a lista e pede a ordem fina em seguida. Com `esperar=True` a resposta já
    vem relida.
    """
    chave = _chave(dados, consulta)
    com_reranker = usa_reranker(consulta, reranker)
    if com_reranker:
        fina = _ler(("fino", chave))
        if fina is not None:
            return fina[0], fina[1], False

    base = _ler(("base", chave))
    if base is None:
        resposta = executar(dados, consulta, encoder=encoder)
        base = (resposta, _facetas(dados, resposta))
        _guardar(("base", chave), base)
    resposta, facetas = base

    refinavel = com_reranker and any(set(r.posicoes) - {"cota"} for r in resposta.resultados[:2])
    if not refinavel:
        return resposta, facetas, False
    if not esperar:
        return resposta, facetas, True
    fina = (busca.refinar(dados, consulta.texto, resposta, reranker), facetas)
    _guardar(("fino", chave), fina)
    return fina[0], facetas, False


def por_dados(dados, nome: str, calcular):
    """Guarda o que só depende dos dados carregados (opções dos filtros, linha do tempo)."""
    chave = (id(dados), nome)
    with _tranca:
        if chave in _por_dados:
            return _por_dados[chave]
    valor = calcular()
    with _tranca:
        _por_dados[chave] = valor
    return valor
