import numpy as np
import pandas as pd


def test(result, ans):
    assert abs(ans - result) <= 1e-5
    return 1
