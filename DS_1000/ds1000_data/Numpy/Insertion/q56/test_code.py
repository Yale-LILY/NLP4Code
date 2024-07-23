import numpy as np


def test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1
