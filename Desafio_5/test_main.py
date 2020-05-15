from main import *


def test_q1():
    result = q1()

    assert type(result) == float

def test_q2():
    result = q2()

    assert type(result) == int

def test_q3():
    result = q3()

    assert type(result) == tuple
    assert len(result) == 2

def test_q4():
    result = q4()

    assert type(result) == list
    assert len(result) == 5
