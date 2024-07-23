from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd


def test(result, ans):
    if np.isnan(ans[0]):
        assert np.isnan(result[0]) and np.isnan(result[1])
    else:
        assert result[0] == ans[0] and result[1] == ans[1]
    return 1
