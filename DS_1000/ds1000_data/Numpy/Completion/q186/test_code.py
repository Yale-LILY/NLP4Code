import numpy as np
import pandas as pd


def test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1
