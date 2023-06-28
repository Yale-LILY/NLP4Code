import numpy as np
import pandas as pd
import io
from scipy import integrate, optimize


def test(result, ans):
    def g(params):
        a, b, c = params
        return (
            ((a + b - c) - 2) ** 2 + ((3 * a - b - c)) ** 2 + np.sin(b) + np.cos(b) + 4
        )

    assert abs(g(result) - g(ans)) < 1e-2
    return 1
