import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy import stats


def define_test_input(args):
    if args.test_case == 1:
        stddev = 2.0785
        mu = 1.744
        x = 25
    elif args.test_case == 2:
        stddev = 2
        mu = 1
        x = 20
    return x, mu, stddev


def ans1(_a):
    x, mu, stddev = _a
    result = stats.lognorm(s=stddev, scale=np.exp(mu)).cdf(x)
    return result


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
