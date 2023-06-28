from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
