import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy.stats as ss


def define_test_input(args):
    if args.test_case == 1:
        x1 = [38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0]
        x2 = [39.2, 39.3, 39.7, 41.4, 41.8, 42.9, 43.3, 45.8]
        x3 = [34.0, 35.0, 39.0, 40.0, 43.0, 43.0, 44.0, 45.0]
        x4 = [34.0, 34.8, 34.8, 35.4, 37.2, 37.8, 41.2, 42.8]

    elif args.test_case == 2:
        np.random.seed(42)
        x1, x2, x3, x4 = np.random.randn(4, 20)

    return x1, x2, x3, x4


def ans1(_a):
    x1, x2, x3, x4 = _a
    statistic, critical_values, significance_level = ss.anderson_ksamp([x1, x2, x3, x4])

    return [statistic, critical_values, significance_level]


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
