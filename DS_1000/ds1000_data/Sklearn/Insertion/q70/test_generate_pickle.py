import numpy as np
import pandas as pd
from sklearn import preprocessing


def define_test_input(args):
    if args.test_case == 1:
        data = np.array([[1, 2], [3, 2], [4, 5]])
    elif args.test_case == 2:
        data = np.array([1, 2, 3, 2, 4, 5])
    return data


def ans1(data):
    centered_scaled_data = preprocessing.scale(data)
    return centered_scaled_data


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    data = define_test_input(args)

    if not os.path.exists("input"):
        os.mkdir("input")
    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(data, f)

    ans = ans1(data)
    # print(ans)
    # np.testing.assert_allclose(ans, ans)
    # pd.testing.assert_frame_equal(ans, ans, check_dtype=False, check_less_precise=True)
    if not os.path.exists("ans"):
        os.mkdir("ans")
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(ans, f)
