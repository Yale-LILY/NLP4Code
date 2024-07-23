import numpy as np
import pandas as pd


def test(result, array_of_arrays):
    assert id(result) != id(array_of_arrays)
    for arr1, arr2 in zip(result, array_of_arrays):
        assert id(arr1) != id(arr2)
        np.testing.assert_array_equal(arr1, arr2)

    return 1
