import numpy as np
import pandas as pd
import io
import scipy.optimize as sciopt


def test(result, ans):
    fp = lambda p, x: p[0] * x[0] + p[1] * x[1]
    e = lambda p, x, y: ((fp(p, x) - y) ** 2).sum()
    pmin, pmax = ans
    x = np.array(
        [
            [
                1247.04,
                1274.9,
                1277.81,
                1259.51,
                1246.06,
                1230.2,
                1207.37,
                1192.0,
                1180.84,
                1182.76,
                1194.76,
                1222.65,
            ],
            [
                589.0,
                581.29,
                576.1,
                570.28,
                566.45,
                575.99,
                601.1,
                620.6,
                637.04,
                631.68,
                611.79,
                599.19,
            ],
        ]
    )
    y = np.array(
        [
            1872.81,
            1875.41,
            1871.43,
            1865.94,
            1854.8,
            1839.2,
            1827.82,
            1831.73,
            1846.68,
            1856.56,
            1861.02,
            1867.15,
        ]
    )
    assert result[0] >= pmin[0] and result[0] <= pmax[0]
    assert result[1] >= pmin[1] and result[1] <= pmax[1]
    assert e(result, x, y) <= 3000
    return 1
