import pandas as pd
import numpy as np


def g(df):
    categories = []
    for i in range(len(df)):
        l = []
        for col in df.columns:
            if df[col].iloc[i] == 1:
                l.append(col)
        categories.append(l)
    df["category"] = categories
    return df


def define_test_input(args):
    if args.test_case == 1:
        df = pd.DataFrame(
            {
                "A": [1, 0, 0, 0, 1, 0],
                "B": [0, 1, 0, 0, 1, 1],
                "C": [1, 1, 1, 0, 1, 0],
                "D": [0, 0, 0, 1, 1, 0],
            }
        )
    if args.test_case == 2:
        df = pd.DataFrame(
            {
                "A": [0, 1, 1, 1, 0, 0],
                "B": [1, 0, 1, 1, 0, 1],
                "C": [0, 0, 0, 1, 1, 0],
                "D": [1, 1, 1, 0, 1, 0],
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
