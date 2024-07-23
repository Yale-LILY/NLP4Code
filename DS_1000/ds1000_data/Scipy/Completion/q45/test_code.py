import numpy as np
import pandas as pd
import io
from scipy import integrate, sparse


def test(result, ans):
    assert result == ans
    return 1
