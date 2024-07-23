import pandas as pd
import numpy as np


def test(result, ans=None):
    try:
        assert len(result) == len(ans)
        for i in range(len(result)):
            for j in range(len(result[i])):
                if np.isnan(result[i][j]) or np.isnan(ans[i][j]):
                    assert np.isnan(result[i][j]) and np.isnan(ans[i][j])
                else:
                    assert result[i][j] == ans[i][j]
        return 1
    except:
        return 0
