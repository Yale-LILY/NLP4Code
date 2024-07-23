import argparse
import os
import parser
import pickle
import numpy as np
import torch
import pandas as pd


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
    return "LabelEncoder" in leaves


def test(result, ans=None):
    
    
    try:
        pd.testing.assert_frame_equal(result, ans, check_dtype=False)
        return 1
    except:
        return 0

























