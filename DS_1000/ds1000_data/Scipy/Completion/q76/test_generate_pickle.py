import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import scipy.optimize
import numpy as np


def define_test_input(args):
    if args.test_case == 1:
        np.random.seed(42)
        a = np.random.rand(3, 5)
        x_true = np.array([10, 13, 5, 8, 40])
        y = a.dot(x_true**2)
        x0 = np.array([2, 3, 1, 4, 20])
        x_bounds = x_true / 2
    return a, x_true, y, x0, x_bounds


def ans1(_a):
    a, x_true, y, x0, x_lower_bounds = _a

    def residual_ans(x, a, y):
        s = ((y - a.dot(x**2)) ** 2).sum()
        return s

    bounds = [[x, None] for x in x_lower_bounds]
    out = scipy.optimize.minimize(
        residual_ans, x0=x0, args=(a, y), method="L-BFGS-B", bounds=bounds
    ).x
    return out


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
