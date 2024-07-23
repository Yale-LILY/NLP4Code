import argparse
import os
import parser
import pickle
import numpy as np
import pandas as pd


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
    return "SelectFromModel" in leaves


def test(result, ans=None):
    
    
    try:
        np.testing.assert_equal(np.array(ans), np.array(result))
        return 1
    except:
        return 0


























