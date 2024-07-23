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
    return "choice" in leaves


def test(result, ans):
    tuples = np.unique(ans, axis=0)
    for tuple in tuples:
        ratio = np.sum(np.all(result == tuple, axis=-1)) / result.shape[0]
        ans_ratio = np.sum(np.all(ans == tuple, axis=-1)) / ans.shape[0]
        assert abs(ratio - ans_ratio) <= 0.05
    return 1
