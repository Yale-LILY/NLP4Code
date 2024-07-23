import numpy as np
import pandas as pd


def test(result, ans):
    assert type(result) == np.ndarray
    np.testing.assert_array_equal(result, ans)

    return 1
