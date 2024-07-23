import argparse
import os
import parser
import pickle
import numpy as np
import torch


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
    return "hierarchy" in leaves


def test(result, ans=None):
    
    
    ret = 0
    try:
        np.testing.assert_equal(result, ans)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_equal(result, 1 - np.array(ans))
        ret = 1
    except:
        pass
    return ret

























