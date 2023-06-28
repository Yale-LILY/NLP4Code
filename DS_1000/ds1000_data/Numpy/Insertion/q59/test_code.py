import numpy as np
import scipy.stats


def test(result, ans):
    assert abs(ans - result) <= 1e-5
    return 1
