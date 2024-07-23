from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd


def test(result, ans):
    assert result[0] == ans[0] and result[1] == ans[1]
    return 1
