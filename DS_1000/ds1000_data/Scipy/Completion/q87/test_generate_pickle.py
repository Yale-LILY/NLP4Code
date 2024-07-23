import numpy as np
import pandas as pd
import scipy
from scipy import sparse


from scipy import sparse


def define_test_input(args):
    if args.test_case == 1:
        V = sparse.random(10, 10, density=0.05, format="coo", random_state=42)
        x = 100
        y = 99

    elif args.test_case == 2:
        V = sparse.coo_matrix(np.diag(-np.arange(5)))
        x = 1
        y = 1
    return V, x, y


def ans1(_a):
    V, x, y = _a
    V = V.copy()
    V.data += x
    V.eliminate_zeros()
    V.data += y
    V.eliminate_zeros()

    return V


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
