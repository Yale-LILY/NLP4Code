import numpy as np
from scipy import signal


def define_test_input(args):
    if args.test_case == 1:
        arr = np.array(
            [
                [
                    -624.59309896,
                    -624.59309896,
                    -624.59309896,
                    -625.0,
                    -625.0,
                    -625.0,
                ],
                [3, 0, 0, 1, 2, 4],
            ]
        )
        n = 2
    elif args.test_case == 2:
        np.random.seed(42)
        arr = (np.random.rand(50, 10) - 0.5) * 10
        n = np.random.randint(2, 4)

    return arr, n


def ans1(_a):
    arr, n = _a
    res = signal.argrelextrema(arr, np.less_equal, order=n, axis=1)
    result = np.zeros((res[0].shape[0], 2)).astype(int)
    result[:, 0] = res[0]
    result[:, 1] = res[1]

    return result


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    a = define_test_input(args)
    ans = ans1(a)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(a, f)

    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
