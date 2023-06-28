import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy.interpolate


def define_test_input(args):
    if args.test_case == 1:
        points = np.array(
            [
                [27.827, 18.53, -30.417],
                [24.002, 17.759, -24.782],
                [22.145, 13.687, -33.282],
                [17.627, 18.224, -25.197],
                [29.018, 18.841, -38.761],
                [24.834, 20.538, -33.012],
                [26.232, 22.327, -27.735],
                [23.017, 23.037, -29.23],
                [28.761, 21.565, -31.586],
                [26.263, 23.686, -32.766],
            ]
        )
        values = np.array(
            [0.205, 0.197, 0.204, 0.197, 0.212, 0.208, 0.204, 0.205, 0.211, 0.215]
        )
        request = np.array([[25, 20, -30]])
    return points, values, request


def ans1(_a):
    points, V, request = _a
    result = scipy.interpolate.griddata(points, V, request)
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
