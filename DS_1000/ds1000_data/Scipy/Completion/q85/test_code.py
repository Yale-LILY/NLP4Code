import numpy as np
import pandas as pd
import io
from scipy import integrate, sparse


def test(V, S):
    assert len(sparse.find(S != V)[0]) == 0
    return 1
