import numpy as np


def test(result, ans):
    assert np.allclose(result, ans)
    return 1
