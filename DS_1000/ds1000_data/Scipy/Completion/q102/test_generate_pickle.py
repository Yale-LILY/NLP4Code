import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy.optimize as sciopt


def define_test_input(args):
    if args.test_case == 1:
        pmin = np.array([0.5, 0.7])  # mimimum bounds
        pmax = np.array([1.5, 1.8])  # maximum bounds
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
    elif args.test_case == 2:
        pmin = np.array([1.0, 0.7])  # mimimum bounds
        pmax = np.array([1.5, 1.1])  # maximum bounds
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
    return pmin, pmax, x, y


def ans1(_a):
    pmin, pmax, x, y = _a
    # p_guess = (pmin + pmax)/2
    # bounds = np.c_[pmin, pmax]
    # fp   = lambda p, x: p[0]*x[0]+p[1]*x[1]
    # e    = lambda p, x, y: ((fp(p,x)-y)**2).sum()
    # sol = sciopt.minimize(e, p_guess, bounds=bounds, args=(x,y))
    # result = sol.x

    return pmin, pmax


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
