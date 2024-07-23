import numpy as np
import pandas as pd
import io
from scipy import integrate, ndimage


def test(result, ans):
    np.testing.assert_allclose(sorted(result), ans)
    return 1
