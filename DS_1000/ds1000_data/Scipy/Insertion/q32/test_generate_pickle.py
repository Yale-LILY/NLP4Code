import numpy as np
import pandas as pd
import scipy
from scipy import sparse


import numpy as np
import scipy.ndimage


def define_test_input(args):
    if args.test_case == 1:
        square = np.zeros((32, 32))
        square[10:-10, 10:-10] = 1
        np.random.seed(12)
        x, y = (32 * np.random.random((2, 20))).astype(int)
        square[x, y] = 1
    return square


def ans1(_a):
    square = _a

    def filter_isolated_cells(array, struct):
        filtered_array = np.copy(array)
        id_regions, num_ids = scipy.ndimage.label(filtered_array, structure=struct)
        id_sizes = np.array(scipy.ndimage.sum(array, id_regions, range(num_ids + 1)))
        area_mask = id_sizes == 1
        filtered_array[area_mask[id_regions]] = 0
        return filtered_array

    square = filter_isolated_cells(square, struct=np.ones((3, 3)))
    return square


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
