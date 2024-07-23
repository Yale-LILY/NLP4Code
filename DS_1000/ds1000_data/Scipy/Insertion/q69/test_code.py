import numpy as np
import pandas as pd
import io
from scipy import integrate, stats


def test(result, ans):
    pd.testing.assert_frame_equal(result, ans, atol=1e-3, check_dtype=False)
    return 1
