import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        data = np.array([4, 2, 5, 6, 7, 5, 4, 3, 5, 7])
        width = 3
    elif args.test_case == 2:
        np.random.seed(42)
        data = np.random.rand(np.random.randint(5, 10))
        width = 4

    return data, width


def ans1(_a):
    data, bin_size = _a
    bin_data_max = (
        data[: (data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
    )

    return bin_data_max


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
