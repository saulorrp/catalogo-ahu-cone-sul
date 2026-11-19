"""As notas do eins zwei (aba γ notas): o texto que o pesquisador digita na sala de consulta.

Um arquivo Markdown por processo, com o nome da cota curta, em `notas/` dentro da pasta da Lista
(`EINS_LISTA_DIR/notas/089-0010-00955.md`), mais um `.json` ao lado com a versão, a data e o
aparelho da última gravação. O texto é a verdade: o eins não o reescreve nem o interpreta aqui.

Nada se perde numa gravação: antes de trocar o texto, a versão anterior vai para
`notas/versoes/<cota>/<data>.md` quando ela veio de outro aparelho ou tem mais de dez minutos.
Assim, dois aparelhos escrevendo a mesma nota nunca apagam um ao outro sem deixar cópia.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ahu.lista_eins import cota_valida

FORMATO = "eins-notas/1"
MAX_TEXTO = 2 * 1024 * 1024
INTERVALO_VERSAO = timedelta(minutes=10)


def _agora() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def _gravar_atomico(caminho: Path, texto: str) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=caminho.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(texto)
        os.replace(tmp, caminho)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Notas:
    """A pasta das notas. Toda escrita é atômica e serializada."""

    _trava = threading.Lock()

    def __init__(self, pasta: str | Path):
        self.pasta = Path(pasta)

    def _md(self, cota: str) -> Path:
        if not cota_valida(cota):
            raise ValueError(f"cota fora do formato curto: {cota!r}")
        return self.pasta / f"{cota}.md"

    def _meta(self, cota: str) -> dict:
        try:
            obj = json.loads((self.pasta / f"{cota}.json").read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError):
            return {}

    def ler(self, cota: str) -> dict | None:
        md = self._md(cota)
        if not md.exists():
            return None
        meta = self._meta(cota)
        return {"formato": FORMATO, "cota": cota, "texto": md.read_text(encoding="utf-8"),
                "versao": int(meta.get("versao") or 0), "atualizado_em": meta.get("atualizado_em"),
                "aparelho": meta.get("aparelho")}

    def listar(self) -> list[dict]:
        if not self.pasta.is_dir():
            return []
        itens = []
        for md in sorted(self.pasta.glob("*.md")):
            if not cota_valida(md.stem):
                continue
            meta = self._meta(md.stem)
            itens.append({"cota": md.stem, "versao": int(meta.get("versao") or 0),
                          "atualizado_em": meta.get("atualizado_em"), "caracteres": md.stat().st_size})
        return itens

    def gravar(self, cota: str, texto: str, aparelho: str = "") -> dict:
        if not isinstance(texto, str):
            raise ValueError("o texto precisa ser uma string")
        if len(texto.encode("utf-8")) > MAX_TEXTO:
            raise ValueError("nota grande demais")
        aparelho = str(aparelho or "")[:80]
        md = self._md(cota)
        with self._trava:
            agora = _agora()
            meta = self._meta(cota)
            versao = int(meta.get("versao") or 0)
            guardou = False
            if md.exists():
                atual = md.read_text(encoding="utf-8")
                if atual == texto:
                    return self.ler(cota)
                guardou = self._guardar_versao(cota, atual, meta, aparelho, agora)
            versao += 1
            _gravar_atomico(md, texto)
            novo = {"formato": FORMATO, "cota": cota, "versao": versao, "atualizado_em": _iso(agora),
                    "aparelho": aparelho or None,
                    "criado_em": meta.get("criado_em") or _iso(agora),
                    "versao_guardada_em": _iso(agora) if guardou else meta.get("versao_guardada_em")}
            meta_texto = json.dumps(novo, ensure_ascii=False, indent=2) + "\n"
            _gravar_atomico(self.pasta / f"{cota}.json", meta_texto)
            return self.ler(cota)

    def _guardar_versao(self, cota: str, atual: str, meta: dict, aparelho: str, agora: datetime) -> bool:
        """Copia o texto atual para `versoes/` se ele veio de outro aparelho ou a última cópia é velha."""
        if not atual.strip():
            return False
        outro_aparelho = (meta.get("aparelho") or "") != aparelho
        ultima = meta.get("versao_guardada_em") or meta.get("criado_em")
        try:
            velha = ultima is None or agora - datetime.fromisoformat(ultima) >= INTERVALO_VERSAO
        except ValueError:
            velha = True
        if not (outro_aparelho or velha):
            return False
        carimbo = agora.strftime("%Y%m%dT%H%M%SZ")
        nome = f"{carimbo} v{int(meta.get('versao') or 0)}.md"
        _gravar_atomico(self.pasta / "versoes" / cota / nome, atual)
        return True
