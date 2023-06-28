import numpy as np
import pandas as pd
import parser


def extract_element(t):
    if type(t) != list:
        return [t]
    xs = []
    for e in t:
        xs += extract_element(e)
    return xs


def stringTest(code):
    ast = parser.st2list(parser.suite(code))
    leaves = extract_element(ast)
    return "random" in leaves


def test(result, ans):
    size, one_ratio = ans
    assert result.shape == (size,)
    assert abs(len(np.where(result == 1)[0]) - size * one_ratio) / size <= 0.05
    assert abs(len(np.where(result == 0)[0]) - size * (1 - one_ratio)) / size <= 0.05
    return 1
