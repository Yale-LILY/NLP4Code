import numpy as np


def test(result, ans):
    for arr1, arr2 in zip(ans, result):
        np.testing.assert_array_equal(arr1, arr2)
    return 1
