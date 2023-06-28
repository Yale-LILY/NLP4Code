import numpy as np
import pandas as pd


def test(result, ans):
    if type(result) == list:
        result = pd.DatetimeIndex(result)
    result = np.array(result).astype(float)
    ans = np.array(ans).astype(float)
    assert np.allclose(result, ans)
    return 1
