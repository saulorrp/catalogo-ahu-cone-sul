import pathlib

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ahu.consulta import Consulta

BASE_DIR = pathlib.Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

def parse_consulta(request: Request) -> Consulta:
    params = {}
    for k, v in request.query_params.multi_items():
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
