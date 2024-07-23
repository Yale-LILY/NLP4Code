import numpy as np
import pandas as pd
from scipy import interpolate


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
