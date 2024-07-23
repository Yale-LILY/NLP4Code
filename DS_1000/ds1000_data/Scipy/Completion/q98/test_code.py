import numpy as np
import scipy as sp
from scipy import integrate, stats


def test(result, ans):
    assert result == ans
    return 1
