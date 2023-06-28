import numpy as np
import pandas as pd


def test(result, ans):
    np.testing.assert_array_equal(sorted(result), sorted(ans))
    return 1
