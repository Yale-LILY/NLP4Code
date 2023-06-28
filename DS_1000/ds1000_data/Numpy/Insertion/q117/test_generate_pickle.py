import numpy as np
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.arange(1, 11)
        index = np.array([0, 1, 0, 0, 0, -1, -1, 2, 2, 1])

    elif args.test_case == 2:
        np.random.seed(42)
        index = np.random.randint(-2, 5, (100,))
        a = np.random.randint(-100, 100, (100,))

    return a, index


def ans1(_a):
    a, index = _a
    add = np.max(index)
    mask = index < 0
    index[mask] += add + 1
    uni = np.unique(index)
    result = np.zeros(np.amax(index) + 1)
    for i in uni:
        result[i] = np.min(a[index == i])

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
