import numpy as np
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.array([[1, 0, 2, 3], [0, 5, 3, 4], [2, 3, 2, 10], [3, 4, 10, 7]])

    return a


def ans1(_a):
    a = _a
    result = np.einsum("ii->i", a)
    save = result.copy()
    a[...] = 0
    result[...] = save

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
