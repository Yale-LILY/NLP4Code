import numpy as np
import pandas as pd
import io
from scipy import integrate, optimize


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
