import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.array([[1, 2], [3, 4]])
        pos = [1, 2]
        element = np.array([[3, 5], [6, 6]])

    elif args.test_case == 2:
        np.random.seed(42)
        a = np.random.rand(100, 10)
        pos = sorted(np.random.randint(0, 99, (5,)))
        element = np.random.rand(5, 10)

    return a, pos, element


def ans1(_a):
    a, pos, element = _a
    pos = np.array(pos) - np.arange(len(element))
    a = np.insert(a, pos, element, axis=0)

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
