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
    return "randint" in leaves


def test(result, ans=None):
    r_old, r_new = result
    assert id(r_old) != id(r_new)
    np.testing.assert_array_equal(r_old, r_new)
    return 1
