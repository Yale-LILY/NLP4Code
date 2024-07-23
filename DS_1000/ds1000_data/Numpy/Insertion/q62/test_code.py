import numpy as np
import pandas as pd


def test(result, ans):
    if ans.shape[0]:
        np.testing.assert_array_equal(result, ans)
    else:
        result = result.reshape(0)
        ans = ans.reshape(0)
        np.testing.assert_array_equal(result, ans)
    return 1
