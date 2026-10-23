"""Acertos de cota e de texto da construção (build/cotas.py) e o link do DigitArq que sai deles."""
import pandas as pd

from ahu import links
from ahu.ficha import marcar_sic
from build import cotas

INDICE = cotas.IndiceDigitArq([
    "PT/AHU/CU/017-001/0016/03265.03267",
    "PT/AHU/CU/017-001/0016/03266",
    "PT/AHU/CU/017-001/0016/03426.03433",
    "PT/AHU/CU/022/0001/00067",
    "PT/AHU/CU/005/0054/04615",
])


def test_registro_do_digitarq():
    # o mesmo número de início vence o intervalo que o contém
    assert INDICE.achar("PT/AHU/CU/017-001/0016/03266") == "PT/AHU/CU/017-001/0016/03266"
    # documento dentro de um intervalo que começa antes
    assert INDICE.achar("PT/AHU/CU/017-001/0016/03429") == "PT/AHU/CU/017-001/0016/03426.03433"
    # caixa trocada entre o catálogo e o DigitArq
    assert INDICE.achar("PT/AHU/CU/022/0002/00067") == "PT/AHU/CU/022/0001/00067"
    assert INDICE.achar("PT/AHU/CU/017-001/0016/03500") is None
    assert INDICE.achar(None) is None


def test_ajustar_refaz_cotas_e_textos():
    base = pd.DataFrame({
        "new_code": ["AHU_CU_0005-01, Cx. 107, D. 20.836-20.840",
                     "AHU_CU_Reino, Cx. 133, pasta 10. INSTRUMENTO DE PESQUISA DOS DOCUMENTOS",
                     "AHU_CU_017-01, Cx. 16, D. 3429."],
        "reference_code": ["PT/AHU/CU/000-501/0107/20836", None, "PT/AHU/CU/017-001/0016/03429"],
        "description": ["Ordem da SãoSão Trindade.", "nau São São Sacramento", "sem erro"],
    })
    saida, contagem = cotas.ajustar(base, INDICE)
    assert saida["reference_code"].tolist()[0] == "PT/AHU/CU/005-001/0107/20836"
    assert saida["new_code"].tolist()[1] == "AHU_CU_Reino, Cx. 133, pasta 10."
    # o resumo não se corrige: vem como a fonte o traz
    assert saida["description"].tolist() == base["description"].tolist()
    assert saida["referencia_digitarq"].tolist()[2] == "PT/AHU/CU/017-001/0016/03426.03433"
    assert contagem["referencia_refeita"] == 1
    assert contagem["new_code_sem_cabecalho"] == 1


def test_link_do_digitarq_por_registro_ou_caixa():
    com = {"reference_code": "PT/AHU/CU/017-001/0016/03429",
           "referencia_digitarq": "PT/AHU/CU/017-001/0016/03426.03433"}
    link = links.do_documento(com)
    assert "03426.03433" in link["digitarq"] and link["digitarq_rotulo"] == "Ficha no DigitArq"
    sem = {"reference_code": "PT/AHU/CU/081/0001/00010", "referencia_digitarq": None}
    link = links.do_documento(sem)
    assert link["digitarq"].endswith("query=PT%2FAHU%2FCU%2F081%2F0001&isAdvancedSearch=false")
    assert link["digitarq_rotulo"] == "Caixa no DigitArq"
    # base sem a coluna: busca pela referência, como antes
    assert "00010" in links.do_documento({"reference_code": "PT/AHU/CU/081/0001/00010"})["digitarq"]


def test_sic_marca_o_erro_da_fonte_sem_corrigir():
    marcado = marcar_sic("Ordem da SãoSão Trindade, Roque da Costa Barrreto, deTávora, D. João III")
    assert marcado.count("[sic]") == 3
    assert "SãoSão" in marcado and "Barrreto" in marcado and "III [sic]" not in marcado
    assert marcar_sic("carro de Lisboa, 1717") == "carro de Lisboa, 1717"
