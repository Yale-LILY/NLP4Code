import tensorflow as tf

import parser


def extract_element(t):
    if type(t) != list:
        return [t]
    xs = []
    for e in t:
        xs += extract_element(e)
    return xs


def stringTest(code):
    code = "def f():\n" + code
    ast = parser.st2list(parser.suite(code))
    leaves = extract_element(ast)
    return "tf" in leaves


def test(result, ans=None):
    try:
        assert result == ans
        return 1
    except:
        return 0
