import pandas as pd
import numpy as np

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
    return "for" not in leaves and "while" not in leaves


def test(result, ans=None):
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0
