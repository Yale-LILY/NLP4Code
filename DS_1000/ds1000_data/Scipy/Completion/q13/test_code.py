import numpy as np
import pandas as pd
import io
from scipy import integrate, interpolate


def test(result, ans):
    np.testing.assert_allclose(result, ans, atol=1e-3)
    return 1
