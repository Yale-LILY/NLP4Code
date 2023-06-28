import numpy as np
import pandas as pd


def test(result, ans):
    np.testing.assert_allclose(result, ans, atol=1e-2)
    return 1
