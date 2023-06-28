import numpy as np
import pandas as pd


def test(result, ans):
    pd.testing.assert_frame_equal(result, ans)
    return 1
