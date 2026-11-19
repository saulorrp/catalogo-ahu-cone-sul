"""A Lista do eins zwei: a memória dos processos que o pesquisador leva do catálogo à triagem.

Um arquivo JSON por processo, com o nome da cota curta da Triagem (`089-0010-00955.json`,
`017.001-0057-13257.json`), numa pasta fora do app (`EINS_LISTA_DIR`). Cada registro guarda a
ficha do catálogo do dia em que o processo entrou, o estado (na lista, fotografado, triado), onde
estão as pastas organizadas e a cópia datada do `_eins_triagem.json` de cada uma.

A Lista é um índice, nunca a dona dos dados: a pasta organizada é a verdade, e a Lista pode ser
refeita a partir dela (o eins zwei reimporta as pastas). Tirar um processo da Lista não apaga
nada: o arquivo vai para `lixeira/` com a data.

Sem web aqui; as rotas estão em `web/rotas_lista.py`.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

FORMATO = "eins-lista/1"
ESTADOS = ("na lista", "fotografado", "triado")
LIXEIRA = "lixeira"

# A cota curta da Triagem: série (com subsérie depois de ponto), caixa e documento.
_RE_CURTA = re.compile(r"^\d{3}(?:\.\d{3})?-\d{4}-\d{5}$")
_RE_REFERENCIA = re.compile(r"PT/AHU/CU/(\d{3})(?:-(\d{1,3}))?/(\d{1,4})/(\d{1,5})(?!\d)")
_RE_ATUAL = re.compile(r"CU_(\d{3})(?:-(\d{1,3}))?,\s*Cx\.\s*(\d{1,4}),\s*D\.\s*(\d{1,5})(?!\d)")
_RE_CURTA_SOLTA = re.compile(r"^\s*(\d{3})(?:[.-](\d{1,3}))?[-_ ](\d{1,4})[-_ ](\d{1,5})(?![\d.])")


def agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cota_curta(texto: str | None) -> str | None:
    """A cota curta da Triagem (`017.001-0057-13257`) de uma cota em qualquer das formas, ou None."""
    texto = str(texto or "")
    m = _RE_REFERENCIA.search(texto) or _RE_ATUAL.search(texto) or _RE_CURTA_SOLTA.match(texto)
    if not m:
        return None
    serie = m.group(1) + (f".{int(m.group(2)):03d}" if m.group(2) else "")
    return f"{serie}-{int(m.group(3)):04d}-{int(m.group(4)):05d}"


def cota_valida(cota: str) -> bool:
    """Só uma cota curta vira nome de arquivo: nada de barra, ponto-ponto ou outro texto."""
    return bool(_RE_CURTA.match(cota or ""))


def _resumo(registro: dict) -> dict:
    return {
        "cota": registro.get("cota"),
        "cota_referencia": registro.get("cota_referencia"),
        "doc_uid": registro.get("doc_uid"),
        "titulo": registro.get("titulo"),
        "ano": registro.get("ano"),
        "estado": registro.get("estado"),
        "adicionado_em": registro.get("adicionado_em"),
        "atualizado_em": registro.get("atualizado_em"),
        "pastas": registro.get("pastas") or [],
    }


class Lista:
    """A pasta da Lista. Toda escrita é atômica (arquivo temporário + troca) e serializada."""

    _trava = threading.Lock()

    def __init__(self, pasta: str | Path):
        self.pasta = Path(pasta)
        self._indice: tuple[int, dict[str, str]] | None = None

    def _arquivo(self, cota: str) -> Path:
        if not cota_valida(cota):
            raise ValueError(f"cota fora do formato curto: {cota!r}")
        return self.pasta / f"{cota}.json"

    def _arquivos(self):
        if not self.pasta.is_dir():
            return []
        return sorted(p for p in self.pasta.glob("*.json") if cota_valida(p.stem))

    @staticmethod
    def _ler_arquivo(caminho: Path) -> dict | None:
        try:
            obj = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    def _gravar(self, caminho: Path, registro: dict) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=caminho.parent, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(registro, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, caminho)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._indice = None

    def listar(self) -> list[dict]:
        itens = [self._ler_arquivo(p) for p in self._arquivos()]
        return [_resumo(r) for r in itens if r]

    def ler(self, cota: str) -> dict | None:
        caminho = self._arquivo(cota)
        return self._ler_arquivo(caminho) if caminho.exists() else None

    def estados_por_doc(self) -> dict[str, str]:
        """doc_uid -> estado, para os botões do catálogo. Refeito só quando a pasta muda."""
        try:
            marca = self.pasta.stat().st_mtime_ns
        except OSError:
            return {}
        if self._indice and self._indice[0] == marca:
            return self._indice[1]
        mapa = {}
        for p in self._arquivos():
            r = self._ler_arquivo(p)
            if r and r.get("doc_uid"):
                mapa[str(r["doc_uid"])] = r.get("estado") or ESTADOS[0]
        self._indice = (marca, mapa)
        return mapa

    def atualizar(self, cota: str, mudanca: dict, ficha_nova: dict | None = None) -> dict:
        """Junta uma mudança ao registro (criando-o se preciso) e devolve o registro gravado.

        A mudança pode ter: `estado`, `ficha`, `pasta` ({computador, nome, caminho, gerado_em}:
        uma entrada por computador + nome), `triagem` (a cópia do `_eins_triagem.json`),
        `triagem_sha256` e `evento` ({tipo, em, ...}: vai para o histórico). `ficha_nova` é a ficha
        do catálogo para um registro que ainda não tem uma.
        """
        if not isinstance(mudanca, dict):
            raise ValueError("a mudança precisa ser um objeto JSON")
        caminho = self._arquivo(cota)
        with self._trava:
            quando = agora()
            registro = self._ler_arquivo(caminho) if caminho.exists() else None
            novo = registro is None
            if novo:
                registro = {"formato": FORMATO, "cota": cota, "cota_referencia": None, "doc_uid": None,
                            "titulo": None, "ano": None, "estado": ESTADOS[0], "adicionado_em": quando,
                            "atualizado_em": quando, "ficha": None, "pastas": [], "triagem": None,
                            "triagem_sha256": None, "historico": [{"tipo": "adicionado", "em": quando}]}

            ficha = mudanca.get("ficha")
            if not isinstance(ficha, dict):
                ficha = ficha_nova if registro.get("ficha") is None else None
            if isinstance(ficha, dict):
                registro["ficha"] = ficha
                registro["ficha_em"] = quando
                cotas = ficha.get("cotas") or {}
                registro["cota_referencia"] = cotas.get("Código de referência") or registro.get("cota_referencia")
                registro["doc_uid"] = ficha.get("doc_uid") or registro.get("doc_uid")
                registro["titulo"] = ficha.get("titulo") or registro.get("titulo")
                registro["ano"] = ficha.get("ano") if ficha.get("ano") is not None else registro.get("ano")

            estado = mudanca.get("estado")
            if estado is not None:
                if estado not in ESTADOS:
                    raise ValueError(f"estado desconhecido: {estado!r}")
                if estado != registro.get("estado"):
                    registro["historico"].append({"tipo": "estado", "em": quando, "de": registro.get("estado"),
                                                  "para": estado})
                registro["estado"] = estado

            pasta = mudanca.get("pasta")
            if isinstance(pasta, dict) and pasta.get("nome"):
                chave = (pasta.get("computador") or None, pasta["nome"])
                pastas = [p for p in registro.get("pastas") or []
                          if (p.get("computador") or None, p.get("nome")) != chave]
                pastas.append({**pasta, "registrada_em": quando})
                registro["pastas"] = pastas

            if "triagem" in mudanca and isinstance(mudanca["triagem"], dict):
                registro["triagem"] = mudanca["triagem"]
                registro["triagem_sha256"] = mudanca.get("triagem_sha256")
                registro["triagem_em"] = quando

            evento = mudanca.get("evento")
            if isinstance(evento, dict) and evento.get("tipo"):
                registro["historico"].append({**evento, "recebido_em": quando, "em": evento.get("em") or quando})

            registro["atualizado_em"] = quando
            self._gravar(caminho, registro)
            return registro

    def remover(self, cota: str) -> bool:
        """Tira o processo da Lista; o arquivo vai para `lixeira/`, com a data no nome."""
        caminho = self._arquivo(cota)
        with self._trava:
            if not caminho.exists():
                return False
            lixeira = self.pasta / LIXEIRA
            lixeira.mkdir(parents=True, exist_ok=True)
            carimbo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            os.replace(caminho, lixeira / f"{cota} - {carimbo}.json")
            self._indice = None
            return True
