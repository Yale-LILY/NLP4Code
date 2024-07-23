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
    return "while" not in leaves and "for" not in leaves


def test(result, ans):
    if ans.shape[0]:
        np.testing.assert_array_equal(result, ans)
    else:
        ans = ans.reshape(0)
        result = result.reshape(0)
        np.testing.assert_array_equal(result, ans)
    return 1
