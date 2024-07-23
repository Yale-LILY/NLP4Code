import numpy as np
import pandas as pd
import io
from scipy import stats


def test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1
