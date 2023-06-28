import numpy as np
import pandas as pd


def test(result, a):
    result = np.array(result)
    if result.shape[1] == a.shape[1]:
        assert np.linalg.matrix_rank(a) == np.linalg.matrix_rank(
            result
        ) and np.linalg.matrix_rank(result) == len(result)
        assert len(np.unique(result, axis=0)) == len(result)
        for arr in result:
            assert np.any(np.all(a == arr, axis=1))
    else:
        assert (
            np.linalg.matrix_rank(a) == np.linalg.matrix_rank(result)
            and np.linalg.matrix_rank(result) == result.shape[1]
        )
        assert np.unique(result, axis=1).shape[1] == result.shape[1]
        for i in range(result.shape[1]):
            assert np.any(np.all(a == result[:, i].reshape(-1, 1), axis=0))

    return 1
