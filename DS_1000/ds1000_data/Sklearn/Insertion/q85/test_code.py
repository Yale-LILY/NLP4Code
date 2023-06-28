import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    try:
        np.testing.assert_equal(result[0], ans[0])
        np.testing.assert_equal(result[1], ans[1])
        return 1
    except:
        return 0


























