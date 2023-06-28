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
    return "TfidfVectorizer" in leaves


def test(result, ans=None):
    
    
    try:
        assert result == "accept"
        return 1
    except:
        return 0


























