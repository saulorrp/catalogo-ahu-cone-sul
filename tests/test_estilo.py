"""Testes do estilo: o fleurão é o interruptor, e vem do CSS (plano v3.1, R6 §2).

O erro que estes testes existem para não deixar voltar: o ❧ escrito **no rótulo** do expansor.
Dessa forma ele ficava ao lado do galão do Streamlit — dois marcadores —, não mudava ao abrir, e
passava pelo interpretador de Markdown do rótulo.
"""
import ast
import re
from pathlib import Path

import pytest

from ahu import estilo

RAIZ = Path(__file__).resolve().parent.parent
APP = RAIZ / "app.py"

# Os glifos de manuscrito da identidade do EINS. No rótulo, nenhum; no CSS, só os dois primeiros.
GLIFOS = ("❧", "❦", "⸿")


@pytest.fixture(scope="module")
def fonte_do_app() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def textos_do_app(fonte_do_app) -> list[str]:
    arvore = ast.parse(fonte_do_app)
    return [n.value for n in ast.walk(arvore)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


# ---------------------------------------------------------------------------
# 1. nenhum glifo escrito em rótulo
# ---------------------------------------------------------------------------

def test_nenhum_texto_da_interface_traz_um_glifo(textos_do_app):
    """O fleurão vem do CSS. No `app.py` ele não aparece em cadeia nenhuma."""
    for texto in textos_do_app:
        for glifo in GLIFOS:
            assert glifo not in texto, f"{glifo!r} em {texto[:60]!r}"


def test_nenhum_modulo_de_dados_traz_um_glifo():
    """Nem nos módulos que montam rótulos (`ficha`, `pessoas`, `selecao`, `amostras`)."""
    for modulo in (RAIZ / "ahu").glob("*.py"):
        if modulo.name in {"estilo.py", "pdf.py"}:
            continue        # o CSS e a documentação da fonte podem citá-los
        texto = modulo.read_text(encoding="utf-8")
        for glifo in GLIFOS:
            assert glifo not in texto, f"{glifo!r} em {modulo.name}"


# ---------------------------------------------------------------------------
# 2. um lugar só com CSS
# ---------------------------------------------------------------------------

def test_o_css_mora_num_lugar_so(fonte_do_app):
    assert "<style>" not in fonte_do_app
    assert "mod_estilo.CSS" in fonte_do_app
    for modulo in (RAIZ / "ahu").glob("*.py"):
        if modulo.name == "estilo.py":
            continue
        assert "<style>" not in modulo.read_text(encoding="utf-8"), modulo.name


def test_o_css_e_injetado_uma_vez_so(fonte_do_app):
    assert fonte_do_app.count("mod_estilo.CSS") == 1


# ---------------------------------------------------------------------------
# 3. o CSS faz o que o R6 §2 pede
# ---------------------------------------------------------------------------

def test_o_fleurao_entra_por_content_e_muda_com_o_estado():
    css = estilo.CSS
    assert f'content: "{estilo.FECHADA}"' in css
    assert f'content: "{estilo.ABERTA}"' in css
    # O estado vem do <details>, que é quem o Streamlit abre e fecha.
    assert 'details[open] > summary::before' in css
    assert estilo.FECHADA == "❧" and estilo.ABERTA == "❦"


def test_o_galao_do_streamlit_e_escondido_com_o_seu_involucro():
    css = estilo.CSS
    assert 'span:has(> [data-testid="stIconMaterial"])' in css
    assert "span:has(> svg)" in css
    assert "display: none !important" in css


def test_a_pilha_de_fontes_do_fleurao_nao_e_de_emoji():
    """O glifo é texto: pilha explícita, e nada de "Segoe UI Emoji"."""
    assert "Segoe UI Symbol" in estilo.FONTE_GLIFOS
    assert "DejaVu Sans" in estilo.FONTE_GLIFOS
    assert "Noto Sans Symbols" in estilo.FONTE_GLIFOS
    assert "Emoji" not in estilo.FONTE_GLIFOS
    assert "font-variant-emoji: text" in estilo.CSS
    # A cor é a do texto, nunca o acento.
    assert "color: inherit" in estilo.CSS


def test_o_foco_de_teclado_fica_visivel_e_pontilhado():
    css = estilo.CSS
    assert "summary:focus-visible" in css
    assert "outline: 1px dotted currentColor" in css
    assert "box-shadow: none" in css          # o anel do Streamlit sai


# ---------------------------------------------------------------------------
# 4. a gaveta: sem moldura, cabeçalho chapado, conteúdo recuado
# ---------------------------------------------------------------------------

def test_a_gaveta_e_marcada_pela_chave_do_container():
    assert estilo.chave_de_gaveta("x") == "gaveta_x"
    assert f'[class*="st-key-{estilo.PREFIXO_GAVETA}"]' in estilo.CSS


def test_a_gaveta_nao_tem_moldura_e_recua_o_conteudo():
    gaveta = [bloco for bloco in estilo.CSS.split("}")
              if f"st-key-{estilo.PREFIXO_GAVETA}" in bloco]
    assert gaveta, "não há regra de gaveta"
    junto = "}".join(gaveta)
    assert "border: none" in junto
    assert "border-top: none" in junto
    assert "padding: 0.5rem 0 0.5rem 1rem" in junto      # 16 px de recuo
    assert "font-weight: 600" in junto
    assert "background: transparent !important" in junto


def test_todas_as_gavetas_do_app_passam_pelo_ajudante(fonte_do_app):
    """Quem abre uma gaveta usa `gaveta(...)`; quem usa `st.expander` direto é cartão."""
    assert "def gaveta(" in fonte_do_app
    assert "mod_estilo.chave_de_gaveta" in fonte_do_app
    # Nenhum `st.expander` recebe `key=`: no Streamlit 1.55 isso só funcionaria com `on_change`,
    # que reexecutaria a página a cada clique (o que o R2 tirou).
    for achado in re.finditer(r"st\.expander\((.*?)\)", fonte_do_app, re.S):
        assert "key=" not in achado.group(1), achado.group(0)[:80]


def test_as_gavetas_comecam_fechadas(fonte_do_app):
    """§4 da identidade: "estado inicial: sempre fechada"."""
    aberturas = re.findall(r"gaveta\((.*?)\)\s*:", fonte_do_app, re.S)
    assert aberturas
    # Duas exceções conscientes, e só estas: a seleção abre quando já há documentos marcados,
    # e o primeiro lote abre para a aba de amostras não começar como uma lista de títulos.
    excecoes = ("bool(quantos)", "expandida=i == 0")
    for argumentos in aberturas:
        if "expandida=" not in argumentos:
            continue
        assert any(e in argumentos for e in excecoes), argumentos[:80]
