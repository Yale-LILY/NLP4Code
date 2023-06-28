import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    try:
        pd.testing.assert_frame_equal(result[0], ans[0])
        pd.testing.assert_frame_equal(result[1], ans[1])
        pd.testing.assert_series_equal(result[2], ans[2])
        pd.testing.assert_series_equal(result[3], ans[3])
        return 1
    except:
        return 0


























