import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.array([[0, 3, 1, 3], [3, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]])
        zero_rows = 0
        zero_cols = 0

    return a, zero_rows, zero_cols


def ans1(_a):
    a, zero_rows, zero_cols = _a
    a[zero_rows, :] = 0
    a[:, zero_cols] = 0

    return a


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(copy.deepcopy(a))

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
