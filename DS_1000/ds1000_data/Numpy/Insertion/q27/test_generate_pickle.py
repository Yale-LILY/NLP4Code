import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.array([[10, 20, 30, 40, 50], [6, 7, 8, 9, 10]])
        permutation = [0, 4, 1, 3, 2]

    elif args.test_case == 2:
        np.random.seed(42)
        a = np.random.rand(np.random.randint(5, 10), np.random.randint(6, 10))
        permutation = np.arange(a.shape[1])
        np.random.shuffle(permutation)
    return a, permutation


def ans1(_a):
    a, permutation = _a
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    a = a[:, c]

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
