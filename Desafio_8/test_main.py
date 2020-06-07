import pandas as pd


def test_0():
    answer = pd.read_csv("answer.csv")
    assert answer.shape == (4570, 2) 
    assert set(["NU_INSCRICAO", "IN_TREINEIRO"]) == set(answer.columns)
