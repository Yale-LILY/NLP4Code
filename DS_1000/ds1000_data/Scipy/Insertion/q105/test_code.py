import numpy as np
import pandas as pd
import io
from scipy import integrate, stats


def test(result, ans):
    pd.testing.assert_frame_equal(result, ans, check_dtype=False, atol=1e-5)
    return 1
