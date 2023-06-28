import numpy as np
import scipy as sp
from scipy import integrate, stats


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
