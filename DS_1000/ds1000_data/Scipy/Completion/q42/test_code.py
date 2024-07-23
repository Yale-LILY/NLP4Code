import numpy as np
import pandas as pd
import io
from scipy import integrate
import scipy.stats as ss


def test(result, ans):
    assert abs(result[0] - ans[0]) <= 1e-5
    assert np.allclose(result[1], ans[1])
    assert abs(result[2] - ans[2]) <= 1e-5

    return 1
