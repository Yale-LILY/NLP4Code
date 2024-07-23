import pandas as pd
import numpy as np


def test(result, ans=None):
    try:
        assert result[0] == ans[0]
        assert result[1] == ans[1]
        return 1
    except:
        return 0
