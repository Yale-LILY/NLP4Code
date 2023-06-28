import numpy as np
import pandas as pd
import io
from scipy import integrate
from scipy.optimize import minimize


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
