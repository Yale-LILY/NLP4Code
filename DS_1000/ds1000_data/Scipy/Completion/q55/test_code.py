import numpy as np
import pandas as pd
import io
from scipy import integrate
import scipy.spatial


def test(result, ans):
    np.testing.assert_array_equal(result, ans)
    return 1
