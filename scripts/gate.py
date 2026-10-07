"""gate.py — portão de qualidade: `ruff check` + `pytest`. Imprime PASS se tudo passar.

Uso: python scripts/gate.py   (rodar antes de cada commit)
"""
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def executar(nome: str, comando: list[str]) -> bool:
    print(f"== {nome}: {' '.join(comando)}")
    return subprocess.run(comando, cwd=RAIZ).returncode == 0


def main() -> int:
    ok = executar("ruff", [sys.executable, "-m", "ruff", "check", "."])
    ok = executar("pytest", [sys.executable, "-m", "pytest"]) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
