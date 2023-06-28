import numpy as np
import numpy as np
import copy


def define_test_input(args):
    if args.test_case == 1:
        number = 4
    elif args.test_case == 2:
        np.random.seed(43)
        number = np.random.randint(0, 360)
    elif args.test_case == 3:
        np.random.seed(142)
        number = np.random.randint(0, 360)

    return number


def ans1(_a):
    number = _a
    deg = np.sin(np.deg2rad(number))
    rad = np.sin(number)
    result = int(rad > deg)

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
