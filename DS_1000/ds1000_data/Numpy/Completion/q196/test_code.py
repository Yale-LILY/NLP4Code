import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, minmax_scale


def test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1
