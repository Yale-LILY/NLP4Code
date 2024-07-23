import numpy as np
import pandas as pd
import io
from scipy import integrate
import scipy.optimize


def test(result, ans):
    a, y = ans

    def residual(x, a, y):
        s = ((y - a.dot(x**2)) ** 2).sum()
        return s

    assert residual(result, a, y) < 1e-5
    return 1
