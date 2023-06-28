import argparse
import os
import pickle
import numpy as np
import pandas as pd
import scipy


def test(result, ans=None):
    
    
    try:
        if type(result) == scipy.sparse.csr.csr_matrix:
            result = result.toarray()
        np.testing.assert_allclose(result, ans)
        return 1
    except:
        return 0


























