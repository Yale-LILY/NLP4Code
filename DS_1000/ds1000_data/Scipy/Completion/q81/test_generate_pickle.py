import numpy as np
import pandas as pd
import scipy
from scipy import sparse


from scipy import sparse


def define_test_input(args):
    if args.test_case == 1:
        sa = sparse.random(10, 10, density=0.01, format="csr", random_state=42)
        sb = sparse.random(10, 10, density=0.01, format="csr", random_state=45)
    return sa, sb


def ans1(_a):
    sa, sb = _a
    result = sparse.vstack((sa, sb)).tocsr()

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
