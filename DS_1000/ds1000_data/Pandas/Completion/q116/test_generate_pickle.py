import pandas as pd
import numpy as np


def g(df):
    df = df.set_index("cat")
    res = df.div(df.sum(axis=0), axis=1)
    return res.reset_index()


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "cat": ["A", "B", "C"],
                "val1": [7, 10, 5],
                "val2": [10, 2, 15],
                "val3": [0, 1, 6],
                "val4": [19, 14, 16],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "cat": ["A", "B", "C"],
                "val1": [1, 1, 4],
                "val2": [10, 2, 15],
                "val3": [0, 1, 6],
                "val4": [19, 14, 16],
            }
        )

    return df


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case", type=int, default=1)
    args = parser.parse_args()

    df = define_test_input(args)

    with open("input/input{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(df, f)

    result = g(df)
    with open("ans/ans{}.pkl".format(args.test_case), "wb") as f:
        pickle.dump(result, f)
