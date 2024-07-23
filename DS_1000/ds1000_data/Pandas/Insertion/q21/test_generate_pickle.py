import pandas as pd
import numpy as np


def g(df):
    df["category"] = df.idxmin(axis=1)
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "A": [0, 1, 1, 1, 0, 1],
                "B": [1, 0, 1, 1, 1, 0],
                "C": [1, 1, 0, 1, 1, 1],
                "D": [1, 1, 1, 0, 1, 1],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "A": [1, 1, 1, 0, 1, 1],
                "B": [1, 1, 0, 1, 1, 1],
                "C": [1, 0, 1, 1, 1, 0],
                "D": [0, 1, 1, 1, 0, 1],
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
