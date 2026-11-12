import pathlib
import re

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ahu.consulta import Consulta

BASE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _marcas(texto: str) -> str:
    """`ahu.ficha.marcar_termos` marca com **…** (já sobre texto escapado): vira <mark>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", texto or "")


templates.env.filters["marcas"] = _marcas

def parse_consulta(request: Request) -> Consulta:
    params = {}
    for k, v in request.query_params.multi_items():
        # O formulário manda todos os campos, inclusive os vazios (ano_de="", etc.); vazio é
        # "sem filtro", e a Consulta recusa um número vazio, o que antes zerava a consulta toda.
        if not v.strip():
            continue
        if k in params:
            params[k] = f"{params[k]}|{v}"
        else:
            params[k] = v
    if "v" not in params:
        params["v"] = "2"
    try:
        return Consulta.de_url(params)
    except Exception:
        return Consulta()
