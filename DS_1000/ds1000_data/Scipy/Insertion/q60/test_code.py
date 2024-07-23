import numpy as np
import pandas as pd
import io
from scipy import integrate, interpolate
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
    return "while" not in leaves and "for" not in leaves


def test(result, ans):
    assert type(result) == np.ndarray
    assert np.allclose(result, ans)
    return 1
