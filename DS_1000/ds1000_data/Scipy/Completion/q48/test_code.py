import numpy as np
import pandas as pd
import io
from scipy import integrate
import scipy.stats as ss


def test(result, ans):
    assert abs(result - ans) <= 1e-5
    return 1
