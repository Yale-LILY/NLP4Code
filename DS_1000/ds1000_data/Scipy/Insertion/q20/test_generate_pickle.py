import numpy as np
import pandas as pd
import scipy
from scipy import sparse


from scipy import sparse


def define_test_input(args):
    if args.test_case == 1:
        c1 = sparse.csr_matrix([[0, 0, 1, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
        c2 = sparse.csr_matrix([[0, 3, 4, 0], [0, 0, 0, 5], [6, 7, 0, 8]])
    return c1, c2


def ans1(_a):
    c1, c2 = _a
    Feature = sparse.hstack((c1, c2)).tocsr()

    return Feature


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
    print(type(ans))
