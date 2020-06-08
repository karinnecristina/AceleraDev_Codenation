import pandas as pd


def test_0():
    answer = pd.read_csv("answer.csv")
    assert answer.shape == (4576, 2) 
    assert set(["NU_INSCRICAO", "NU_NOTA_MT"]) == set(answer.columns)
