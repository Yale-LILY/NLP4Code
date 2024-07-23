import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
from scipy.sparse import lil_matrix
from scipy import sparse


def define_test_input(args):
    if args.test_case == 1:
        sA = sparse.random(10, 10, density=0.1, format="lil", random_state=42)
    return sA


def ans1(_a):
    M = _a
    rows, cols = M.nonzero()
    M[cols, rows] = M[rows, cols]

    return M


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
