import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    ret = 0
    try:
        np.testing.assert_equal(result, ans)
        ret = 1
    except:
        pass
    try:
        np.testing.assert_equal(result, 1 - ans)
        ret = 1
    except:
        pass
    return ret


























