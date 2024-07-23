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
    return "select" in leaves and "np" in leaves


def test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1
