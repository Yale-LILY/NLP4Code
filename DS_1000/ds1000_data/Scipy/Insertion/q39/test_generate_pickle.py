import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy.spatial.distance


def define_test_input(args):
    if args.test_case == 1:
        example_array = np.array(
            [
                [0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 2, 2, 0, 6, 0, 3, 3, 3],
                [0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                [1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 3],
                [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 5, 5, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            ]
        )
    return example_array


def ans1(_a):
    example_array = _a
    import itertools

    n = example_array.max() + 1
    indexes = []
    for k in range(1, n):
        tmp = np.nonzero(example_array == k)
        tmp = np.asarray(tmp).T
        indexes.append(tmp)
    result = np.zeros((n - 1, n - 1), dtype=float)
    for i, j in itertools.combinations(range(n - 1), 2):
        d2 = scipy.spatial.distance.cdist(
            indexes[i], indexes[j], metric="minkowski", p=1
        )
        result[i, j] = result[j, i] = d2.min()

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
