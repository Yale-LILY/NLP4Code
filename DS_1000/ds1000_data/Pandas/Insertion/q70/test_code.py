import pandas as pd
import numpy as np


def test(result, ans=None):
    try:
        assert type(result) == type(ans)
        np.testing.assert_array_equal(result, ans)
        return 1
    except:
        return 0
