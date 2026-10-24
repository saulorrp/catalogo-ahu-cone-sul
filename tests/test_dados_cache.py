"""O cache do Streamlit carrega os dados de novo quando o código das classes em cache muda."""
from ahu import dados as mod_dados


def test_versao_do_codigo_muda_com_o_lexico(tmp_path, monkeypatch):
    for nome in mod_dados._MODULOS_EM_CACHE:
        (tmp_path / nome).write_text("x = 1\n", "utf-8")
    monkeypatch.setattr(mod_dados, "__file__", str(tmp_path / "dados.py"))
    antes = mod_dados._versao_do_codigo()
    (tmp_path / "lexico.py").write_text("x = 2\n", "utf-8")
    assert mod_dados._versao_do_codigo() != antes


def test_carregar_no_streamlit_usa_a_versao_na_chave(monkeypatch):
    chamadas = []
    monkeypatch.setattr(mod_dados, "_cacheado_streamlit", lambda *args: chamadas.append(args))
    monkeypatch.setattr(mod_dados, "_versao_do_codigo", lambda: "v2")
    mod_dados.carregar("pasta")
    assert chamadas == [("pasta", "v2")]
