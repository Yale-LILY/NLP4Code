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
    return "get_dummies" in leaves and "OneHotEncoder" not in leaves


def test(result, ans=None):
    
    
    try:
        if type(result) == np.ndarray:
            np.testing.assert_equal(ans[:, :3], result[:, :3])
        elif type(result) == pd.DataFrame:
            np.testing.assert_equal(ans[:, :3], result.to_numpy()[:, :3])
        return 1
    except:
        return 0


























