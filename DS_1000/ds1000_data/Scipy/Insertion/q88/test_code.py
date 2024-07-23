import numpy as np
import pandas as pd
import io
from scipy import integrate, sparse


def test(result, ans):
    assert type(result) == sparse.csc.csc_matrix
    assert len(sparse.find(result != ans)[0]) == 0

    return 1
