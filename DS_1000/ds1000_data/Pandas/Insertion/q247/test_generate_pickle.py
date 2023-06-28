import pandas as pd
import numpy as np


def g(s):
    return (
        pd.DataFrame.from_records(s.values, index=s.index)
        .reset_index()
        .rename(columns={"index": "name"})
    )


def define_test_input(args):
    if args.test_case == 1:
        series = pd.Series(
            [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])],
            index=["file1", "file2", "file3"],
        )
    if args.test_case == 2:
        series = pd.Series(
            [
                np.array([11, 12, 13, 14]),
                np.array([5, 6, 7, 8]),
                np.array([9, 10, 11, 12]),
            ],
            index=["file1", "file2", "file3"],
        )
    return series


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    s = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(s, f)

    result = g(s)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
