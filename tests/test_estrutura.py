"""Testes do WP 0.1: estrutura do repositório e regras de convenção."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def test_estrutura_do_repositorio():
    pastas = ("ahu", "build", "avaliacao", "tests", "corpus", "app_dados", "scripts", "schemas", "legado")
    for pasta in pastas:
        assert (RAIZ / pasta).is_dir(), pasta


def test_arquivos_de_convencao():
    for arq in ("AGENTS.md", "CLAUDE.md", "GEMINI.md", "LICENSE", "LICENSE-DATA.md", "README.md",
                "requirements.txt", "runtime.txt", "pyproject.toml"):
        assert (RAIZ / arq).is_file(), arq
    assert "AGENTS.md" in (RAIZ / "CLAUDE.md").read_text(encoding="utf-8")
    assert "AGENTS.md" in (RAIZ / "GEMINI.md").read_text(encoding="utf-8")


def test_python_311_declarado():
    assert (RAIZ / "runtime.txt").read_text(encoding="utf-8").strip() == "python-3.11"


def test_dependencias_fixadas_sem_openai():
    texto = (RAIZ / "requirements.txt").read_text(encoding="utf-8")
    # Comentário e opção do pip (`--extra-index-url`, do WP 2.5) não são dependência.
    linhas = [ln.strip() for ln in texto.splitlines()
              if ln.strip() and not ln.lstrip().startswith(("#", "-"))]
    assert linhas
    assert all("==" in ln for ln in linhas)
    assert not any(ln.lower().startswith("openai") for ln in linhas)


def test_torch_e_so_cpu():
    """WP 2.5: no Streamlit Community Cloud o torch do PyPI arrasta as bibliotecas CUDA."""
    texto = (RAIZ / "requirements.txt").read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in texto
    assert "torch==2.8.0+cpu" in texto


def test_readme_usa_nomes_do_preprint():
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    assert "SRSP" in texto and "1 a 9" in texto
    assert "IRSP" not in texto
