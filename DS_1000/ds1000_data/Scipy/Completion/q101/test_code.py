import numpy as np
import pandas as pd
import io
from scipy import integrate


def test(result, ans):
    assert np.allclose(result, ans, atol=1e-2)
    return 1
