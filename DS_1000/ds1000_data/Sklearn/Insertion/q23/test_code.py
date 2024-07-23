import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    ret = 0
    try:
        np.testing.assert_allclose(result, ans[0], rtol=1e-3)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_allclose(result, ans[1], rtol=1e-3)
        ret = 1
    except:
        pass
    return ret


























