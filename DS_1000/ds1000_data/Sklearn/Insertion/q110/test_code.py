import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    try:
        np.testing.assert_equal(sorted(result), sorted(ans))
        return 1
    except:
        return 0


























