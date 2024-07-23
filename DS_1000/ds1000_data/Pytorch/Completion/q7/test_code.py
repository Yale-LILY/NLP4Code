import argparse
import os
import pickle
import numpy as np
import pandas as pd


def test(result, ans=None):
    
    
    try:
        assert type(result) == pd.DataFrame
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


























