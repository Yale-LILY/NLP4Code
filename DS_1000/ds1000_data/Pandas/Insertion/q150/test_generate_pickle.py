import pandas as pd
import numpy as np


def g(df):
    return df.groupby("l")["v"].apply(pd.Series.sum, skipna=False).reset_index()


def define_test_input(args):
    if args.test_case == 1:
        d = {
            "l": ["left", "right", "left", "right", "left", "right"],
            "r": ["right", "left", "right", "left", "right", "left"],
            "v": [-1, 1, -1, 1, -1, np.nan],
        }
        df = pd.DataFrame(d)
    if args.test_case == 2:
        d = {
            "l": ["left", "right", "left", "left", "right", "right"],
            "r": ["right", "left", "right", "right", "left", "left"],
            "v": [-1, 1, -1, -1, 1, np.nan],
        }
        df = pd.DataFrame(d)
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
