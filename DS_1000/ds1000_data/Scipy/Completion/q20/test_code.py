import numpy as np
import pandas as pd
import io
from scipy import integrate, sparse


def test(result, ans):
    assert type(result) == sparse.csr_matrix
    assert len(sparse.find(ans != result)[0]) == 0
    return 1
