import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        x = [-1, 2, 5, 100]
        y = [123, 456, 789, 1255]
    elif args.test_case == 2:
        np.random.seed(42)
        x = (np.random.rand(100) - 0.5) * 10
        y = (np.random.rand(100) - 0.5) * 10
    return x, y


def ans1(_a):
    x, y = _a
    result = np.polyfit(x, y, 2)

    return result


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
