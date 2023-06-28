import numpy as np
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        a = np.array([[1, 2], [3, 0]])
    elif args.test_case == 2:
        np.random.seed(42)
        a = np.random.rand(5, 6)
    return a


def ans1(_a):
    a = _a
    result = np.unravel_index(a.argmax(), a.shape)

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
