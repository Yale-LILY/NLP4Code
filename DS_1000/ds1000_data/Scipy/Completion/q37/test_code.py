import numpy as np
import pandas as pd
import io
from scipy import integrate
import scipy.optimize


def test(result, ans):
    np.testing.assert_allclose(result[0], ans[0], atol=1e-3)
    np.testing.assert_allclose(result[1], ans[1], atol=1e-2)

    return 1
