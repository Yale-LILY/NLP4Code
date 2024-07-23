import numpy as np
import pandas as pd
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        x = np.array([-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2])
    elif args.test_case == 2:
        np.random.seed(42)
        x = np.random.rand(10) - 0.5

    return x


def ans1(_a):
    x = _a
    result = x[x >= 0]

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
